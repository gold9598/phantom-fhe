#include "bsgs.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "evaluate.cuh"
#include "polymath.cuh"
#include "single_chain_plaintext.h"
#include "uintmath.cuh"
#include "uintmodmath.cuh"

namespace phantom {

    // Forward declarations of kernels defined in src/bootstrap.cu (non-static,
    // namespace phantom). Reused here to share one source of truth with the
    // single-chain expand path. The int16 variant applies scale_2 to restore
    // the full message scale from the SCP's quantization scale; the int64
    // variant is used for full-scale SCPs (scale_2 implicitly 1).
    __global__ void light_pt_expand_per_tower_kernel(
            const std::int64_t *src_signed,
            std::uint64_t *dst,
            const DModulus *moduli,
            std::size_t num_towers,
            std::size_t N);

    __global__ void light_pt_expand_per_tower_i16_kernel(
            const std::int16_t *src_signed,
            std::uint64_t *dst,
            const DModulus *moduli,
            std::int64_t scale_2,
            std::size_t num_towers,
            std::size_t N);

    namespace {

        inline bool is_power_of_two(std::size_t x) {
            return x != 0 && (x & (x - 1)) == 0;
        }

        // Highest power-of-2 <= x (for x >= 1).
        inline std::size_t msb_pow2(std::size_t x) {
            std::size_t r = 1;
            while ((r << 1) <= x) r <<= 1;
            return r;
        }

        // Fused batched MAC for one BSGS giant.
        //
        // For each (tower j, slot i) in [0, num_towers) x [0, N):
        //   acc0 = sum_{b in [0, M)} pooled[b][j][i] * babies[b].c0[j][i]   mod q_j
        //   acc1 = sum_{b in [0, M)} pooled[b][j][i] * babies[b].c1[j][i]   mod q_j
        //
        // Bit-identical to running M back-to-back (multiply_plain_ntt + add_inplace)
        // calls: each iteration computes a 128-bit product, adds it to a 128-bit
        // running accumulator, then Barrett-reduces to [0, q_j) before the next
        // accumulate. This matches the per-step modular reduction semantics of
        // the unfused path (which Barrett-reduces inside multiply_rns_poly and
        // then conditional-subtracts inside add_uint64_uint64_mod).
        __global__ void mac_accumulate_kernel(
            const std::uint64_t *__restrict__ pooled,         // M * num_towers * N
            const std::uint64_t *const *__restrict__ babies_c0_ptrs,  // M pointers
            const std::uint64_t *const *__restrict__ babies_c1_ptrs,  // M pointers
            std::uint64_t *__restrict__ out_c0,               // num_towers * N
            std::uint64_t *__restrict__ out_c1,
            const DModulus *__restrict__ moduli,
            std::size_t M,
            std::size_t num_towers,
            std::size_t N) {
            using namespace phantom::arith;

            const std::size_t total = num_towers * N;
            const std::size_t pt_stride = num_towers * N;  // per-baby stride in pooled

            for (std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                 tid < total;
                 tid += blockDim.x * gridDim.x) {
                const std::size_t j = tid / N;
                const std::size_t i = tid - j * N;
                const DModulus mod = moduli[j];
                const std::uint64_t qj = mod.value();
                const std::uint64_t *const ratio = mod.const_ratio();
                const std::size_t off = j * N + i;

                std::uint64_t acc0 = 0;
                std::uint64_t acc1 = 0;
                for (std::size_t b = 0; b < M; ++b) {
                    const std::uint64_t pt_v = pooled[b * pt_stride + off];
                    const std::uint64_t b0_v = babies_c0_ptrs[b][off];
                    const std::uint64_t b1_v = babies_c1_ptrs[b][off];

                    // 128-bit product + add accumulator + Barrett reduce.
                    // (acc + pt*baby) < q_j + q_j^2 fits in 128 bits since q_j < 2^60.
                    uint128_t prod0 = multiply_uint64_uint64(pt_v, b0_v);
                    uint128_t prod1 = multiply_uint64_uint64(pt_v, b1_v);
                    uint128_t sum0 = add_uint128_uint64(prod0, acc0);
                    uint128_t sum1 = add_uint128_uint64(prod1, acc1);
                    acc0 = barrett_reduce_uint128_uint64(sum0, qj, ratio);
                    acc1 = barrett_reduce_uint128_uint64(sum1, qj, ratio);
                }

                out_c0[off] = acc0;
                out_c1[off] = acc1;
            }
        }

    } // namespace

    BsgsDiagonals pre_encode_bsgs_diagonals(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const std::vector<double> &matrix,
            std::size_t num_rows,
            std::size_t num_cols,
            std::size_t d_pad,
            std::size_t baby_steps,
            double scale) {
        if (num_rows == 0 || num_cols == 0) {
            throw std::invalid_argument("pre_encode_bsgs_diagonals: dimensions must be non-zero");
        }
        if (matrix.size() != num_rows * num_cols) {
            throw std::invalid_argument("pre_encode_bsgs_diagonals: matrix size mismatch");
        }
        if (!is_power_of_two(d_pad)) {
            throw std::invalid_argument("pre_encode_bsgs_diagonals: d_pad must be a power of 2");
        }
        if (d_pad < num_rows || d_pad < num_cols) {
            throw std::invalid_argument("pre_encode_bsgs_diagonals: d_pad must be >= max(num_rows, num_cols)");
        }
        if (baby_steps == 0 || d_pad % baby_steps != 0) {
            throw std::invalid_argument("pre_encode_bsgs_diagonals: baby_steps must divide d_pad");
        }
        const std::size_t giant_steps = d_pad / baby_steps;

        const std::size_t num_slots = encoder.slot_count();
        if (num_slots % d_pad != 0) {
            throw std::invalid_argument("pre_encode_bsgs_diagonals: num_slots must be multiple of d_pad");
        }

        // diag[d][t] = padded[t][(t + d) mod d_pad], with padded zero-extended
        // beyond [num_rows x num_cols].
        std::vector<std::vector<double>> diagonals(d_pad, std::vector<double>(d_pad, 0.0));
        for (std::size_t t = 0; t < num_rows; ++t) {
            const double *row = matrix.data() + t * num_cols;
            for (std::size_t d = 0; d < d_pad; ++d) {
                const std::size_t j = (t + d) % d_pad;
                if (j < num_cols) {
                    diagonals[d][t] = row[j];
                }
            }
        }

        BsgsDiagonals out;
        out.d_pad = d_pad;
        out.baby_steps = baby_steps;
        out.giant_steps = giant_steps;

        std::vector<double> pt_real(num_slots, 0.0);
        std::vector<std::complex<double>> slots(num_slots);

        out.diagonals.reserve(d_pad);
        for (std::size_t g = 0; g < giant_steps; ++g) {
            const std::size_t g_shift = g * baby_steps;
            for (std::size_t b = 0; b < baby_steps; ++b) {
                const auto &src = diagonals[g_shift + b];
                // Left-rotate src by g_shift positions into the first period:
                // pt_real[i] = src[(i + g_shift) mod d_pad] for i in [0, d_pad).
                if (g_shift == 0) {
                    std::copy(src.begin(), src.end(), pt_real.begin());
                } else {
                    // first[..g_shift] = src[d_pad - g_shift..]
                    // first[g_shift..] = src[..d_pad - g_shift]
                    std::copy(src.begin() + (d_pad - g_shift), src.end(), pt_real.begin());
                    std::copy(src.begin(), src.begin() + (d_pad - g_shift), pt_real.begin() + g_shift);
                }
                // Tile across remaining periods.
                for (std::size_t off = d_pad; off < num_slots; off += d_pad) {
                    std::copy(pt_real.begin(), pt_real.begin() + d_pad, pt_real.begin() + off);
                }
                for (std::size_t i = 0; i < num_slots; ++i) {
                    slots[i] = std::complex<double>(pt_real[i], 0.0);
                }
                out.diagonals.push_back(encode_single_chain_plaintext(ctx, encoder, slots, scale));
            }
        }
        return out;
    }

    std::vector<int> bsgs_required_steps(std::size_t baby_steps) {
        std::vector<int> steps;
        if (baby_steps <= 1) {
            return steps;
        }
        // Powers of 2 in [1, baby_steps): 1, 2, 4, ..., baby_steps/2 (chained baby rotations).
        for (std::size_t s = 1; s < baby_steps; s <<= 1) {
            steps.push_back(static_cast<int>(s));
        }
        // baby_steps itself for giant Horner stride.
        steps.push_back(static_cast<int>(baby_steps));
        return steps;
    }

    std::vector<PhantomCiphertext> compute_bsgs_babies(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            std::size_t baby_steps) {
        if (baby_steps == 0) {
            throw std::invalid_argument("compute_bsgs_babies: baby_steps must be > 0");
        }

        // ---- Babies: chained power-of-2 rotations. ----
        // babies[0] = x; babies[b] = rotate(babies[b - msb(b)], msb(b)).
        std::vector<PhantomCiphertext> babies;
        babies.reserve(baby_steps);
        babies.push_back(x);
        for (std::size_t b = 1; b < baby_steps; ++b) {
            const std::size_t bit = msb_pow2(b);
            const std::size_t rem = b - bit;
            babies.push_back(rotate(ctx, babies[rem], static_cast<int>(bit), galois_key));
        }
        return babies;
    }

    PhantomCiphertext bsgs_apply_giants_with_babies(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const std::vector<PhantomCiphertext> &babies,
            const BsgsDiagonals &diags) {
        const std::size_t M = diags.baby_steps;
        const std::size_t G = diags.giant_steps;
        const std::size_t d_pad = diags.d_pad;
        if (M == 0 || G == 0 || M * G != d_pad) {
            throw std::invalid_argument("bsgs_apply_giants_with_babies: M*G must equal d_pad");
        }
        if (babies.size() != M) {
            throw std::invalid_argument("bsgs_apply_giants_with_babies: babies size != baby_steps");
        }
        if (diags.diagonals.size() != d_pad) {
            throw std::invalid_argument("bsgs_apply_giants_with_babies: diagonals size mismatch");
        }

        const auto &stream = cudaStreamPerThread;
        const PhantomCiphertext &x = babies[0];
        const double nominal = x.scale();
        const std::size_t target_ci = x.chain_index();

        // ---- Per-giant fused MAC accumulation ----
        //
        // Setup invariants: all babies share x.chain_index() (rotate preserves
        // chain_index) and are size-2 NTT-form ciphertexts. Cache the babies'
        // c0/c1 device pointers in a host vector once, then upload to a device
        // pointer-array buffer reused across giants.
        const auto &cd = ctx.get_context_data(target_ci);
        const auto &mods = cd.parms().coeff_modulus();
        const std::size_t num_towers = mods.size();
        const std::size_t N = cd.parms().poly_modulus_degree();
        const std::size_t per_poly = num_towers * N;
        const auto *base_rns = ctx.gpu_rns_tables().modulus();

        std::vector<const std::uint64_t *> h_babies_c0_ptrs(M);
        std::vector<const std::uint64_t *> h_babies_c1_ptrs(M);
        for (std::size_t b = 0; b < M; ++b) {
            const auto *base_ptr = babies[b].data();
            h_babies_c0_ptrs[b] = base_ptr;
            h_babies_c1_ptrs[b] = base_ptr + per_poly;
        }
        auto d_babies_c0_ptrs = phantom::util::make_cuda_auto_ptr<const std::uint64_t *>(M, stream);
        auto d_babies_c1_ptrs = phantom::util::make_cuda_auto_ptr<const std::uint64_t *>(M, stream);
        cudaMemcpyAsync(d_babies_c0_ptrs.get(), h_babies_c0_ptrs.data(),
                        M * sizeof(const std::uint64_t *),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_babies_c1_ptrs.get(), h_babies_c1_ptrs.data(),
                        M * sizeof(const std::uint64_t *),
                        cudaMemcpyHostToDevice, stream);

        // Pool of M expanded plaintexts, reused across all giants.
        // At LLaMA scale (M=128, num_towers=12, N=32768): ~400 MB.
        auto pooled = phantom::util::make_cuda_auto_ptr<std::uint64_t>(
                M * per_poly, stream);

        // Build a partial PhantomCiphertext with the fused MAC. The result is
        // size-2, NTT-form, at the babies' chain_index, with scale = babies'
        // scale * pt scale (matches multiply_plain_ntt's scale update).
        auto build_partial = [&](std::size_t g) -> PhantomCiphertext {
            const std::size_t base = g * M;

            // Step 1a: expand each plaintext into the pooled buffer (per-tower
            // expansion only; NTT is launched once for all babies after the
            // expand sweep finishes).
            for (std::size_t b = 0; b < M; ++b) {
                const auto &scp = diags.diagonals[base + b];
                if (scp.N() != N) {
                    throw std::invalid_argument(
                            "bsgs_apply_giants_with_babies: scp coeff length != N");
                }

                std::uint64_t *pt_dst = pooled.get() + b * per_poly;

                const std::size_t threads = 256;
                const std::size_t total = num_towers * N;
                const std::size_t blocks = (total + threads - 1) / threads;

                // H2D async into a device scratch buffer first, since the source
                // lives in pinned host memory. Dispatch on the SCP's storage
                // width: int16 (quantized) applies scale_2; int64 (full-scale).
                if (scp.is_int16) {
                    const std::int64_t scale_2 = (scp.coeff_scale > 0.0)
                            ? static_cast<std::int64_t>(std::llround(scp.scale / scp.coeff_scale))
                            : 1;
                    auto d_signed = phantom::util::make_cuda_auto_ptr<std::int16_t>(N, stream);
                    cudaMemcpyAsync(d_signed.get(), scp.coeffs.data(),
                                    N * sizeof(std::int16_t),
                                    cudaMemcpyHostToDevice, stream);
                    light_pt_expand_per_tower_i16_kernel<<<blocks, threads, 0, stream>>>(
                            d_signed.get(), pt_dst, base_rns, scale_2,
                            num_towers, N);
                } else {
                    auto d_signed = phantom::util::make_cuda_auto_ptr<std::int64_t>(N, stream);
                    cudaMemcpyAsync(d_signed.get(), scp.coeffs64.data(),
                                    N * sizeof(std::int64_t),
                                    cudaMemcpyHostToDevice, stream);
                    light_pt_expand_per_tower_kernel<<<blocks, threads, 0, stream>>>(
                            d_signed.get(), pt_dst, base_rns,
                            num_towers, N);
                }

                // Forward NTT per baby. Empirically faster than the batched
                // variant at LLaMA scale because a single-poly NTT already
                // saturates the GPU; batching inflates working set into L2
                // thrashing. The batched API is kept for future use cases
                // where M is small enough for twiddle-cache reuse to win.
                nwt_2d_radix8_forward_inplace(pt_dst, ctx.gpu_rns_tables(),
                                              num_towers, 0, stream);
            }

            // Step 2: allocate output ciphertext (size=2, NTT-form), then fused MAC.
            PhantomCiphertext acc;
            acc.resize(ctx, target_ci, /*size=*/2, stream);
            acc.set_ntt_form(true);
            acc.set_scale(x.scale() * diags.diagonals[base + 0].scale);
            acc.set_correction_factor(x.correction_factor());

            std::uint64_t *out_c0 = acc.data_ptr().get();
            std::uint64_t *out_c1 = out_c0 + per_poly;

            const std::size_t threads = 256;
            const std::size_t total = num_towers * N;
            const std::size_t blocks = (total + threads - 1) / threads;
            mac_accumulate_kernel<<<blocks, threads, 0, stream>>>(
                    pooled.get(),
                    d_babies_c0_ptrs.get(),
                    d_babies_c1_ptrs.get(),
                    out_c0,
                    out_c1,
                    base_rns,
                    M,
                    num_towers,
                    N);

            return acc;
        };

        // Seed with partial_{G-1} (no rotate; Horner ends at absolute shift (G-1)*M).
        PhantomCiphertext acc = build_partial(G - 1);

        // For g = G-2 .. 0: acc = rotate(acc, M); acc += partial_g.
        for (std::size_t gi = G - 1; gi-- > 0;) {
            acc = rotate(ctx, acc, static_cast<int>(M), galois_key);
            PhantomCiphertext partial_g = build_partial(gi);
            add_inplace(ctx, acc, partial_g);
        }

        acc = rescale_to_next(ctx, acc);
        acc.set_scale(nominal);
        // Suppress unused warning for target_ci on builds that strip the assert.
        (void)target_ci;
        return acc;
    }

    PhantomCiphertext bsgs_matmul_preencoded(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            const BsgsDiagonals &diags) {
        auto babies = compute_bsgs_babies(ctx, galois_key, x, diags.baby_steps);
        return bsgs_apply_giants_with_babies(ctx, galois_key, babies, diags);
    }

    // ---- Complex-folded BSGS ----

    namespace {
        // Build (w_real, w_imag) of shape (folded_rows x folded_cols) from
        // the input real matrix according to the requested fold mode.
        void apply_complex_fold(
                const std::vector<double> &matrix,
                std::size_t num_rows,
                std::size_t num_cols,
                ComplexFoldMode mode,
                std::vector<double> &w_real,
                std::vector<double> &w_imag,
                std::size_t &folded_rows,
                std::size_t &folded_cols) {
            if (mode == ComplexFoldMode::Rows) {
                const std::size_t num_rows_half = (num_rows + 1) / 2;
                folded_rows = num_rows_half;
                folded_cols = num_cols;
                w_real.assign(num_rows_half * num_cols, 0.0);
                w_imag.assign(num_rows_half * num_cols, 0.0);
                for (std::size_t i = 0; i < num_rows_half; ++i) {
                    for (std::size_t j = 0; j < num_cols; ++j) {
                        w_real[i * num_cols + j] = matrix[i * num_cols + j];
                    }
                    const std::size_t bot_row = i + num_rows_half;
                    if (bot_row < num_rows) {
                        for (std::size_t j = 0; j < num_cols; ++j) {
                            w_imag[i * num_cols + j] = matrix[bot_row * num_cols + j];
                        }
                    }
                }
            } else {  // ColsConj
                const std::size_t num_cols_half = (num_cols + 1) / 2;
                folded_rows = num_rows;
                folded_cols = num_cols_half;
                w_real.assign(num_rows * num_cols_half, 0.0);
                w_imag.assign(num_rows * num_cols_half, 0.0);
                for (std::size_t i = 0; i < num_rows; ++i) {
                    for (std::size_t j = 0; j < num_cols_half; ++j) {
                        w_real[i * num_cols_half + j] = matrix[i * num_cols + j];
                        const std::size_t right = j + num_cols_half;
                        if (right < num_cols) {
                            w_imag[i * num_cols_half + j] = -matrix[i * num_cols + right];
                        }
                    }
                }
            }
        }
    } // namespace

    ComplexBsgsDiagonals pre_encode_bsgs_diagonals_complex(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const std::vector<double> &matrix,
            std::size_t num_rows,
            std::size_t num_cols,
            std::size_t d_pad,
            std::size_t baby_steps,
            double scale,
            ComplexFoldMode mode) {
        if (num_rows == 0 || num_cols == 0) {
            throw std::invalid_argument("pre_encode_bsgs_diagonals_complex: dimensions must be non-zero");
        }
        if (matrix.size() != num_rows * num_cols) {
            throw std::invalid_argument("pre_encode_bsgs_diagonals_complex: matrix size mismatch");
        }
        if (!is_power_of_two(d_pad)) {
            throw std::invalid_argument("pre_encode_bsgs_diagonals_complex: d_pad must be a power of 2");
        }
        if (baby_steps == 0 || d_pad % baby_steps != 0) {
            throw std::invalid_argument("pre_encode_bsgs_diagonals_complex: baby_steps must divide d_pad");
        }

        // Fold the input matrix.
        std::vector<double> w_real;
        std::vector<double> w_imag;
        std::size_t folded_rows = 0;
        std::size_t folded_cols = 0;
        apply_complex_fold(matrix, num_rows, num_cols, mode,
                           w_real, w_imag, folded_rows, folded_cols);

        if (d_pad < folded_rows || d_pad < folded_cols) {
            throw std::invalid_argument(
                    "pre_encode_bsgs_diagonals_complex: d_pad must be >= max(folded_rows, folded_cols)");
        }

        const std::size_t giant_steps = d_pad / baby_steps;
        const std::size_t num_slots = encoder.slot_count();
        if (num_slots % d_pad != 0) {
            throw std::invalid_argument(
                    "pre_encode_bsgs_diagonals_complex: num_slots must be multiple of d_pad");
        }

        // Build per-diagonal real/imag arrays of length d_pad.
        // diag_re[d][t] = padded[t][(t+d) mod d_pad]; same for diag_im.
        std::vector<std::vector<double>> diag_real(d_pad, std::vector<double>(d_pad, 0.0));
        std::vector<std::vector<double>> diag_imag(d_pad, std::vector<double>(d_pad, 0.0));
        for (std::size_t t = 0; t < folded_rows; ++t) {
            const double *row_re = w_real.data() + t * folded_cols;
            const double *row_im = w_imag.data() + t * folded_cols;
            for (std::size_t d = 0; d < d_pad; ++d) {
                const std::size_t j = (t + d) % d_pad;
                if (j < folded_cols) {
                    diag_real[d][t] = row_re[j];
                    diag_imag[d][t] = row_im[j];
                }
            }
        }

        ComplexBsgsDiagonals out;
        out.inner.d_pad = d_pad;
        out.inner.baby_steps = baby_steps;
        out.inner.giant_steps = giant_steps;

        std::vector<double> pt_real(num_slots, 0.0);
        std::vector<double> pt_imag(num_slots, 0.0);
        std::vector<std::complex<double>> slots(num_slots);

        out.inner.diagonals.reserve(d_pad);
        for (std::size_t g = 0; g < giant_steps; ++g) {
            const std::size_t g_shift = g * baby_steps;
            for (std::size_t b = 0; b < baby_steps; ++b) {
                const auto &src_re = diag_real[g_shift + b];
                const auto &src_im = diag_imag[g_shift + b];
                if (g_shift == 0) {
                    std::copy(src_re.begin(), src_re.end(), pt_real.begin());
                    std::copy(src_im.begin(), src_im.end(), pt_imag.begin());
                } else {
                    std::copy(src_re.begin() + (d_pad - g_shift), src_re.end(), pt_real.begin());
                    std::copy(src_re.begin(), src_re.begin() + (d_pad - g_shift), pt_real.begin() + g_shift);
                    std::copy(src_im.begin() + (d_pad - g_shift), src_im.end(), pt_imag.begin());
                    std::copy(src_im.begin(), src_im.begin() + (d_pad - g_shift), pt_imag.begin() + g_shift);
                }
                for (std::size_t off = d_pad; off < num_slots; off += d_pad) {
                    std::copy(pt_real.begin(), pt_real.begin() + d_pad, pt_real.begin() + off);
                    std::copy(pt_imag.begin(), pt_imag.begin() + d_pad, pt_imag.begin() + off);
                }
                for (std::size_t i = 0; i < num_slots; ++i) {
                    slots[i] = std::complex<double>(pt_real[i], pt_imag[i]);
                }
                out.inner.diagonals.push_back(
                        encode_single_chain_plaintext(ctx, encoder, slots, scale));
            }
        }
        return out;
    }

    PhantomCiphertext bsgs_matmul_preencoded_complex(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            const ComplexBsgsDiagonals &diags) {
        // BSGS structure is unchanged for complex-encoded diagonals.
        return bsgs_matmul_preencoded(ctx, galois_key, x, diags.inner);
    }

    PhantomCiphertext bsgs_apply_giants_with_babies_complex(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const std::vector<PhantomCiphertext> &babies,
            const ComplexBsgsDiagonals &diags) {
        return bsgs_apply_giants_with_babies(ctx, galois_key, babies, diags.inner);
    }

}
