#include "single_chain_plaintext.h"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

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

    __global__ void light_pt_expand_per_tower_kernel(
            const std::int64_t *src_signed,
            std::uint64_t *dst,
            const DModulus *moduli,
            std::size_t num_towers,
            std::size_t N);

    namespace {
        inline void check_cuda(cudaError_t err, const char *what) {
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("SingleChainPlaintext: ") + what +
                                         " failed: " + cudaGetErrorString(err));
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

    // ===== encode_single_chain_plaintext =====

    SingleChainPlaintext encode_single_chain_plaintext(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const std::vector<std::complex<double>> &slots,
            double scale) {
        const auto &stream = cudaStreamPerThread;

        // Encode at chain_index = 1 (first usable index) to drive q_0 selection.
        // Single-chain storage is level-agnostic; the picked level only fixes
        // which prime sets the centering modulus q_0.
        const std::size_t chain_index = 1;

        std::vector<cuDoubleComplex> v(slots.size());
        for (std::size_t i = 0; i < slots.size(); ++i) {
            v[i] = make_cuDoubleComplex(slots[i].real(), slots[i].imag());
        }

        PhantomPlaintext full_pt;
        encoder.encode<cuDoubleComplex>(ctx, v, scale, full_pt, chain_index);

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

        // Signed-center on device into a temporary int64 buffer, then D2H
        // copy into pinned host memory.
        auto d_signed = phantom::util::make_cuda_auto_ptr<std::int64_t>(N, stream);
        const std::size_t threads = 256;
        const std::size_t blocks = (N + threads - 1) / threads;
        light_pt_signed_center_kernel<<<blocks, threads, 0, stream>>>(
                tower0.get(), d_signed.get(), q0, N);

        SingleChainPlaintext out;
        out.scale = scale;
        out.coeffs = PinnedHostInt64Buffer(N);
        check_cuda(cudaMemcpyAsync(out.coeffs.data(), d_signed.get(),
                                   N * sizeof(std::int64_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "D2H signed coeffs");
        check_cuda(cudaStreamSynchronize(stream), "stream sync after encode");
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

        if (scp.coeffs.size() != N) {
            throw std::invalid_argument(
                    "expand_single_chain_to_full: coeff length != N");
        }

        // H2D async into a device scratch buffer first, since the source lives
        // in pinned host memory.
        auto d_signed = phantom::util::make_cuda_auto_ptr<std::int64_t>(N, stream);
        check_cuda(cudaMemcpyAsync(d_signed.get(), scp.coeffs.data(),
                                   N * sizeof(std::int64_t),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D signed coeffs");

        PhantomPlaintext pt;
        pt.set_chain_index(target_chain_index);
        pt.set_scale(scp.scale);
        pt.resize(coeff_modulus_size, N, stream);

        const DModulus *moduli = ctx.gpu_rns_tables().modulus();
        const std::size_t threads = 256;
        const std::size_t total = coeff_modulus_size * N;
        const std::size_t blocks = (total + threads - 1) / threads;
        light_pt_expand_per_tower_kernel<<<blocks, threads, 0, stream>>>(
                d_signed.get(),
                pt.data(),
                moduli,
                coeff_modulus_size,
                N);

        nwt_2d_radix8_forward_inplace(pt.data(), ctx.gpu_rns_tables(),
                                      coeff_modulus_size, 0, stream);
        return pt;
    }

}
