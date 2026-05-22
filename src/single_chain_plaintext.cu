#include "single_chain_plaintext.h"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_wrapper.cuh"
#include "ntt.cuh"
#include "rns.cuh"

namespace phantom {

    // Forward declarations of kernels defined in src/bootstrap.cu (non-static,
    // namespace phantom). Reused here so we avoid duplicating the kernel
    // bodies and keep one source of truth.
    __global__ void light_pt_signed_center_kernel(
            const std::uint64_t *src_tower0,
            std::int64_t *dst_signed,
            std::uint64_t q0,
            std::size_t N);

    // Int64 expand (full-scale SCPs): per-tower mod-q_j reduction, scale_2 == 1.
    __global__ void light_pt_expand_per_tower_kernel(
            const std::int64_t *src_signed,
            std::uint64_t *dst,
            const DModulus *moduli,
            std::int64_t scale_2,
            std::size_t num_towers,
            std::size_t N);

    // Int16 expand (quantized IRP weight SCPs): multiplies each int16 coeff
    // (stored at coeff_scale) by scale_2 to restore the full message scale,
    // then reduces mod q_j per RNS tower.
    __global__ void light_pt_expand_per_tower_i16_kernel(
            const std::int16_t *src_signed,
            std::uint64_t *dst,
            const DModulus *moduli,
            std::int64_t scale_2,
            std::size_t num_towers,
            std::size_t N);

    // Block-floating-point int8 expand (Q8_0 / MXFP8 style): recovers
    // coeff_i = round(int8[i] * block_scale[i / block_size]) at the full 2^40
    // message scale, then reduces mod q_j per RNS tower.
    __global__ void light_pt_expand_per_tower_i8_bfp_kernel(
            const std::int8_t *src_mantissa,
            const float *block_scales,
            std::uint64_t *dst,
            const DModulus *moduli,
            std::size_t block_size,
            std::size_t num_towers,
            std::size_t N);

    namespace {
        inline void check_cuda(cudaError_t err, const char *what) {
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("SingleChainPlaintext: ") + what +
                                         " failed: " + cudaGetErrorString(err));
            }
        }

        // Narrow signed int64 coefficients (already centered in [-q0/2, q0/2))
        // to int16. Only invoked when the host range-check confirms every coeff
        // fits int16 (quantized IRP weight SCPs at coeff_scale=2^16).
        __global__ void narrow_i64_to_i16_kernel(
                const std::int64_t *__restrict__ src,
                std::int16_t *__restrict__ dst,
                std::size_t N) {
            for (std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                 tid < N;
                 tid += blockDim.x * gridDim.x) {
                dst[tid] = static_cast<std::int16_t>(src[tid]);
            }
        }
    }

    // ===== PinnedHostInt64Buffer =====

    PinnedHostInt64Buffer::PinnedHostInt64Buffer(std::size_t n) : n_(n) {
        if (n == 0) return;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n * sizeof(std::int64_t)),
                   "cudaMallocHost");
    }

    PinnedHostInt64Buffer::~PinnedHostInt64Buffer() {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
    }

    PinnedHostInt64Buffer::PinnedHostInt64Buffer(const PinnedHostInt64Buffer &other) : n_(other.n_) {
        if (n_ == 0) return;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n_ * sizeof(std::int64_t)),
                   "cudaMallocHost (copy)");
        std::memcpy(ptr_, other.ptr_, n_ * sizeof(std::int64_t));
    }

    PinnedHostInt64Buffer &PinnedHostInt64Buffer::operator=(const PinnedHostInt64Buffer &other) {
        if (this == &other) return *this;
        if (ptr_ != nullptr) { cudaFreeHost(ptr_); ptr_ = nullptr; }
        n_ = other.n_;
        if (n_ == 0) return *this;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n_ * sizeof(std::int64_t)),
                   "cudaMallocHost (copy-assign)");
        std::memcpy(ptr_, other.ptr_, n_ * sizeof(std::int64_t));
        return *this;
    }

    PinnedHostInt64Buffer::PinnedHostInt64Buffer(PinnedHostInt64Buffer &&other) noexcept
            : ptr_(other.ptr_), n_(other.n_) {
        other.ptr_ = nullptr;
        other.n_ = 0;
    }

    PinnedHostInt64Buffer &PinnedHostInt64Buffer::operator=(PinnedHostInt64Buffer &&other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) cudaFreeHost(ptr_);
            ptr_ = other.ptr_;
            n_ = other.n_;
            other.ptr_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }

    // ===== PinnedHostInt16Buffer =====

    PinnedHostInt16Buffer::PinnedHostInt16Buffer(std::size_t n) : n_(n) {
        if (n == 0) return;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n * sizeof(std::int16_t)),
                   "cudaMallocHost");
    }

    PinnedHostInt16Buffer::~PinnedHostInt16Buffer() {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
    }

    PinnedHostInt16Buffer::PinnedHostInt16Buffer(const PinnedHostInt16Buffer &other) : n_(other.n_) {
        if (n_ == 0) return;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n_ * sizeof(std::int16_t)),
                   "cudaMallocHost (copy)");
        std::memcpy(ptr_, other.ptr_, n_ * sizeof(std::int16_t));
    }

    PinnedHostInt16Buffer &PinnedHostInt16Buffer::operator=(const PinnedHostInt16Buffer &other) {
        if (this == &other) return *this;
        if (ptr_ != nullptr) { cudaFreeHost(ptr_); ptr_ = nullptr; }
        n_ = other.n_;
        if (n_ == 0) return *this;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n_ * sizeof(std::int16_t)),
                   "cudaMallocHost (copy-assign)");
        std::memcpy(ptr_, other.ptr_, n_ * sizeof(std::int16_t));
        return *this;
    }

    PinnedHostInt16Buffer::PinnedHostInt16Buffer(PinnedHostInt16Buffer &&other) noexcept
            : ptr_(other.ptr_), n_(other.n_) {
        other.ptr_ = nullptr;
        other.n_ = 0;
    }

    PinnedHostInt16Buffer &PinnedHostInt16Buffer::operator=(PinnedHostInt16Buffer &&other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) cudaFreeHost(ptr_);
            ptr_ = other.ptr_;
            n_ = other.n_;
            other.ptr_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }

    // ===== PinnedHostInt8Buffer =====

    PinnedHostInt8Buffer::PinnedHostInt8Buffer(std::size_t n) : n_(n) {
        if (n == 0) return;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n * sizeof(std::int8_t)),
                   "cudaMallocHost");
    }

    PinnedHostInt8Buffer::~PinnedHostInt8Buffer() {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
    }

    PinnedHostInt8Buffer::PinnedHostInt8Buffer(const PinnedHostInt8Buffer &other) : n_(other.n_) {
        if (n_ == 0) return;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n_ * sizeof(std::int8_t)),
                   "cudaMallocHost (copy)");
        std::memcpy(ptr_, other.ptr_, n_ * sizeof(std::int8_t));
    }

    PinnedHostInt8Buffer &PinnedHostInt8Buffer::operator=(const PinnedHostInt8Buffer &other) {
        if (this == &other) return *this;
        if (ptr_ != nullptr) { cudaFreeHost(ptr_); ptr_ = nullptr; }
        n_ = other.n_;
        if (n_ == 0) return *this;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n_ * sizeof(std::int8_t)),
                   "cudaMallocHost (copy-assign)");
        std::memcpy(ptr_, other.ptr_, n_ * sizeof(std::int8_t));
        return *this;
    }

    PinnedHostInt8Buffer::PinnedHostInt8Buffer(PinnedHostInt8Buffer &&other) noexcept
            : ptr_(other.ptr_), n_(other.n_) {
        other.ptr_ = nullptr;
        other.n_ = 0;
    }

    PinnedHostInt8Buffer &PinnedHostInt8Buffer::operator=(PinnedHostInt8Buffer &&other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) cudaFreeHost(ptr_);
            ptr_ = other.ptr_;
            n_ = other.n_;
            other.ptr_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }

    // ===== PinnedHostFloatBuffer =====

    PinnedHostFloatBuffer::PinnedHostFloatBuffer(std::size_t n) : n_(n) {
        if (n == 0) return;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n * sizeof(float)),
                   "cudaMallocHost");
    }

    PinnedHostFloatBuffer::~PinnedHostFloatBuffer() {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
    }

    PinnedHostFloatBuffer::PinnedHostFloatBuffer(const PinnedHostFloatBuffer &other) : n_(other.n_) {
        if (n_ == 0) return;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n_ * sizeof(float)),
                   "cudaMallocHost (copy)");
        std::memcpy(ptr_, other.ptr_, n_ * sizeof(float));
    }

    PinnedHostFloatBuffer &PinnedHostFloatBuffer::operator=(const PinnedHostFloatBuffer &other) {
        if (this == &other) return *this;
        if (ptr_ != nullptr) { cudaFreeHost(ptr_); ptr_ = nullptr; }
        n_ = other.n_;
        if (n_ == 0) return *this;
        check_cuda(cudaMallocHost(reinterpret_cast<void **>(&ptr_), n_ * sizeof(float)),
                   "cudaMallocHost (copy-assign)");
        std::memcpy(ptr_, other.ptr_, n_ * sizeof(float));
        return *this;
    }

    PinnedHostFloatBuffer::PinnedHostFloatBuffer(PinnedHostFloatBuffer &&other) noexcept
            : ptr_(other.ptr_), n_(other.n_) {
        other.ptr_ = nullptr;
        other.n_ = 0;
    }

    PinnedHostFloatBuffer &PinnedHostFloatBuffer::operator=(PinnedHostFloatBuffer &&other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) cudaFreeHost(ptr_);
            ptr_ = other.ptr_;
            n_ = other.n_;
            other.ptr_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }

    // ===== encode_single_chain_plaintext =====

    SingleChainPlaintext encode_single_chain_plaintext(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const std::vector<std::complex<double>> &slots,
            double scale,
            double coeff_scale,
            std::size_t block_size) {
        const auto &stream = cudaStreamPerThread;

        // Block-floating-point int8 encodes the coeffs at the FULL message scale
        // (the per-block fp32 scale absorbs the entire magnitude). For the int16/
        // int64 adaptive path, coeff_scale <= 0 means "use scale".
        const double enc_scale = (block_size > 0)
                ? scale
                : ((coeff_scale > 0.0) ? coeff_scale : scale);

        // Encode at chain_index = 1 (first usable index) to drive q_0 selection.
        // Single-chain storage is level-agnostic; the picked level only fixes
        // which prime sets the centering modulus q_0.
        const std::size_t chain_index = 1;

        std::vector<cuDoubleComplex> v(slots.size());
        for (std::size_t i = 0; i < slots.size(); ++i) {
            v[i] = make_cuDoubleComplex(slots[i].real(), slots[i].imag());
        }

        PhantomPlaintext full_pt;
        encoder.encode<cuDoubleComplex>(ctx, v, enc_scale, full_pt, chain_index);

        const auto &cd = ctx.get_context_data(chain_index);
        const auto &mods = cd.parms().coeff_modulus();
        const std::size_t N = cd.parms().poly_modulus_degree();
        const std::uint64_t q0 = mods.front().value();

        // INTT tower 0 (in a scratch copy — must not mutate full_pt's storage)
        // back to coefficient form before signed-centering.
        auto tower0 = phantom::util::make_cuda_auto_ptr<std::uint64_t>(N, stream);
        check_cuda(cudaMemcpyAsync(tower0.get(), full_pt.data(),
                                   N * sizeof(std::uint64_t),
                                   cudaMemcpyDeviceToDevice, stream),
                   "D2D tower0 copy");
        nwt_2d_radix8_backward_inplace(tower0.get(), ctx.gpu_rns_tables(),
                                       /*coeff_modulus_size=*/1,
                                       /*start_modulus_idx=*/0, stream);

        // Signed-center on device into an int64 buffer, then D2H to host.
        auto d_signed = phantom::util::make_cuda_auto_ptr<std::int64_t>(N, stream);
        const std::size_t threads = 256;
        const std::size_t blocks = (N + threads - 1) / threads;
        light_pt_signed_center_kernel<<<blocks, threads, 0, stream>>>(
                tower0.get(), d_signed.get(), q0, N);

        std::vector<std::int64_t> host64(N);
        check_cuda(cudaMemcpyAsync(host64.data(), d_signed.get(),
                                   N * sizeof(std::int64_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "D2H signed coeffs");
        check_cuda(cudaStreamSynchronize(stream), "stream sync after encode");

        // Block-floating-point int8 storage (Q8_0 / MXFP8 style). Partition the
        // N coeffs (now at the full 2^40 message scale) into blocks of
        // `block_size`; per block store one fp32 scale = absmax/127 and
        // `block_size` int8 mantissas q = round(coeff / scale) clamped to
        // [-127, 127]. The expand kernel recovers round(q * block_scale).
        if (block_size > 0) {
            if (N % block_size != 0) {
                throw std::invalid_argument(
                        "encode_single_chain_plaintext: N must be divisible by "
                        "block_size for block-floating-point int8");
            }
            const std::size_t num_blocks = N / block_size;
            SingleChainPlaintext out;
            out.scale = scale;
            out.coeff_scale = scale;   // BFP block scales already restore 2^40
            out.is_int16 = false;
            out.is_int8_bfp = true;
            out.block_size = block_size;
            out.coeffs8 = PinnedHostInt8Buffer(N);
            out.block_scales = PinnedHostFloatBuffer(num_blocks);
            for (std::size_t b = 0; b < num_blocks; ++b) {
                std::int64_t absmax = 0;
                for (std::size_t k = 0; k < block_size; ++k) {
                    std::int64_t a = std::llabs(host64[b * block_size + k]);
                    if (a > absmax) absmax = a;
                }
                // fp32 scale: a Q8_0-style fp16 d would overflow (absmax ~ 2^28,
                // d ~ 2^21 > fp16 max ~2^16). All-zero block -> scale 1.
                const float bscale = (absmax > 0)
                        ? static_cast<float>(static_cast<double>(absmax) / 127.0)
                        : 1.0f;
                out.block_scales.data()[b] = bscale;
                const double inv = 1.0 / static_cast<double>(bscale);
                for (std::size_t k = 0; k < block_size; ++k) {
                    const std::size_t i = b * block_size + k;
                    long q = std::lround(static_cast<double>(host64[i]) * inv);
                    if (q > 127) q = 127;
                    else if (q < -127) q = -127;
                    out.coeffs8.data()[i] = static_cast<std::int8_t>(q);
                }
            }
            return out;
        }

        // Adaptive storage: int16 when every coeff fits (quantized IRP weight
        // SCPs at coeff_scale=2^16), int64 otherwise (full-scale SCPs whose
        // coeffs at the 2^40 message scale far exceed int16). Lossless in both.
        bool fits_i16 = true;
        for (std::size_t i = 0; i < N; ++i) {
            if (host64[i] < INT16_MIN || host64[i] > INT16_MAX) {
                fits_i16 = false;
                break;
            }
        }

        SingleChainPlaintext out;
        out.scale = scale;
        out.coeff_scale = enc_scale;
        out.is_int16 = fits_i16;
        if (fits_i16) {
            // Narrow on device then D2H the compact int16 buffer.
            auto d_i16 = phantom::util::make_cuda_auto_ptr<std::int16_t>(N, stream);
            narrow_i64_to_i16_kernel<<<blocks, threads, 0, stream>>>(
                    d_signed.get(), d_i16.get(), N);
            out.coeffs = PinnedHostInt16Buffer(N);
            check_cuda(cudaMemcpyAsync(out.coeffs.data(), d_i16.get(),
                                       N * sizeof(std::int16_t),
                                       cudaMemcpyDeviceToHost, stream),
                       "D2H int16 coeffs");
            check_cuda(cudaStreamSynchronize(stream), "stream sync after narrow");
        } else {
            out.coeffs64 = PinnedHostInt64Buffer(N);
            std::memcpy(out.coeffs64.data(), host64.data(), N * sizeof(std::int64_t));
        }
        return out;
    }

    // ===== expand_single_chain_to_full =====

    PhantomPlaintext expand_single_chain_to_full(
            const PhantomContext &ctx,
            const SingleChainPlaintext &scp,
            std::size_t target_chain_index) {
        const auto &stream = cudaStreamPerThread;
        const auto &cd = ctx.get_context_data(target_chain_index);
        const auto &mods = cd.parms().coeff_modulus();
        const std::size_t coeff_modulus_size = mods.size();
        const std::size_t N = cd.parms().poly_modulus_degree();

        if (scp.N() != N) {
            throw std::invalid_argument(
                    "expand_single_chain_to_full: coeff length != N");
        }

        PhantomPlaintext pt;
        pt.set_chain_index(target_chain_index);
        pt.set_scale(scp.scale);
        pt.resize(coeff_modulus_size, N, stream);

        const DModulus *moduli = ctx.gpu_rns_tables().modulus();
        const std::size_t threads = 256;
        const std::size_t total = coeff_modulus_size * N;
        const std::size_t blocks = (total + threads - 1) / threads;

        if (scp.is_int8_bfp) {
            // Block-floating-point int8: dequant per block via the fp32 scales.
            const std::size_t num_blocks = N / scp.block_size;
            auto d_mant = phantom::util::make_cuda_auto_ptr<std::int8_t>(N, stream);
            auto d_bscale = phantom::util::make_cuda_auto_ptr<float>(num_blocks, stream);
            check_cuda(cudaMemcpyAsync(d_mant.get(), scp.coeffs8.data(),
                                       N * sizeof(std::int8_t),
                                       cudaMemcpyHostToDevice, stream),
                       "H2D int8 mantissas");
            check_cuda(cudaMemcpyAsync(d_bscale.get(), scp.block_scales.data(),
                                       num_blocks * sizeof(float),
                                       cudaMemcpyHostToDevice, stream),
                       "H2D block scales");
            light_pt_expand_per_tower_i8_bfp_kernel<<<blocks, threads, 0, stream>>>(
                    d_mant.get(), d_bscale.get(), pt.data(), moduli,
                    scp.block_size, coeff_modulus_size, N);
        } else if (scp.is_int16) {
            // scale_2 restores the full message scale (scp.scale) from the
            // coeffs' quantization scale (scp.coeff_scale); 1 if they are equal.
            const std::int64_t scale_2 = (scp.coeff_scale > 0.0)
                    ? static_cast<std::int64_t>(std::llround(scp.scale / scp.coeff_scale))
                    : 1;
            auto d_signed = phantom::util::make_cuda_auto_ptr<std::int16_t>(N, stream);
            check_cuda(cudaMemcpyAsync(d_signed.get(), scp.coeffs.data(),
                                       N * sizeof(std::int16_t),
                                       cudaMemcpyHostToDevice, stream),
                       "H2D int16 coeffs");
            light_pt_expand_per_tower_i16_kernel<<<blocks, threads, 0, stream>>>(
                    d_signed.get(), pt.data(), moduli, scale_2,
                    coeff_modulus_size, N);
        } else {
            const std::int64_t scale_2 = (scp.coeff_scale > 0.0)
                    ? static_cast<std::int64_t>(std::llround(scp.scale / scp.coeff_scale))
                    : 1;
            auto d_signed = phantom::util::make_cuda_auto_ptr<std::int64_t>(N, stream);
            check_cuda(cudaMemcpyAsync(d_signed.get(), scp.coeffs64.data(),
                                       N * sizeof(std::int64_t),
                                       cudaMemcpyHostToDevice, stream),
                       "H2D int64 coeffs");
            light_pt_expand_per_tower_kernel<<<blocks, threads, 0, stream>>>(
                    d_signed.get(), pt.data(), moduli, scale_2,
                    coeff_modulus_size, N);
        }

        nwt_2d_radix8_forward_inplace(pt.data(), ctx.gpu_rns_tables(),
                                      coeff_modulus_size, 0, stream);
        return pt;
    }

}
