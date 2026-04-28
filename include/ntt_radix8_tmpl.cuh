#pragma once

// =============================================================================
// Templated radix-8 2D NTT kernels (basic in-place path).
//
// These four kernels are the workhorses for both the standalone
// nwt_2d_radix8_forward_inplace / _backward_inplace launchers and the
// keyswitching-fused launchers in ntt_modup.cu / ntt_moddown.cu. Centralizing
// them here keeps a single source of truth.
//
// Design (mirrors lapis ntt_2d_tmpl.cu):
//   - LOG_N1 / LOG_N2 are compile-time so n1, n2, group, pad, smem_stride and
//     loop tail counts collapse into constexpr expressions.
//   - All power-of-two divisions/modulos are shifts/masks.
//   - Forward butterflies use the lazy primitives (ct_butterfly_lazy /
//     fntt8_lazy / fntt4_lazy + lazy_reduce_to_2q) when the modulus is below
//     2^kNttLazyPrimeBits = 2^59. Inverse stays on non-lazy GS — matches
//     lapis intt_2d_tmpl which doesn't lazy the GS path.
// =============================================================================

#include "ntt.cuh"
#include "butterfly.cuh"
#include "common.h"

namespace phantom::ntt::radix8 {

using phantom::arith::ct_butterfly;
using phantom::arith::ct_butterfly_lazy;
using phantom::arith::gs_butterfly;
using phantom::arith::fntt8;
using phantom::arith::fntt8_lazy;
using phantom::arith::fntt4;
using phantom::arith::fntt4_lazy;
using phantom::arith::intt8;
using phantom::arith::intt4;
using phantom::arith::lazy_reduce_to_2q;
using phantom::arith::csub_q;
using phantom::arith::multiply_and_reduce_shoup_lazy;
using phantom::arith::kNttLazyPrimeBits;
using phantom::util::per_block_pad;

__host__ __device__ constexpr int ct_log2_pow2(size_t v) {
    int r = 0;
    while ((static_cast<size_t>(1) << r) < v) ++r;
    return r;
}

// Bank-conflict avoidance for 64-bit smem accesses (32 banks * 8B). Inserts
// one padding slot every 16 entries; breaks stride-16 / stride-32 conflicts
// on the row-NTT phase. Used for the n2-axis layout in fnwt_phase2 and
// inwt_phase1_oop. The column-NTT phase uses the existing per-block_pad
// padding (smem_row = n1 + pad) and doesn't need this.
__device__ __forceinline__ size_t smem_pad_addr(size_t addr) {
    return addr + (addr >> 4);
}

__host__ __device__ constexpr size_t smem_padded_per_set(size_t n2) {
    return n2 + (n2 >> 4);
}

// Total smem (in uint64s) the n2-axis kernels need per block. block_size = 128
// threads, num_sets = 128/group = 128/(n2/8) = 1024/n2.
__host__ __device__ constexpr size_t smem_padded_total_uint64(size_t n2) {
    return (1024 / n2) * smem_padded_per_set(n2);
}

// -----------------------------------------------------------------------------
// Forward NTT phase 1 (column NTTs of size n1). In-place.
// -----------------------------------------------------------------------------
template <int LOG_N1, int LOG_N2>
__global__ __launch_bounds__(((1 << LOG_N1) >> 3) * per_block_pad, 1)
void fnwt_phase1(uint64_t *inout,
                 const uint64_t *twiddles,
                 const uint64_t *twiddles_shoup,
                 const DModulus *modulus,
                 size_t coeff_mod_size,
                 size_t start_mod_idx) {
    constexpr int LOG_N             = LOG_N1 + LOG_N2;
    constexpr size_t n              = 1ULL << LOG_N;
    constexpr size_t n1             = 1ULL << LOG_N1;
    constexpr int log_group         = LOG_N1 - 3;
    constexpr size_t group          = 1ULL << log_group;
    constexpr size_t pad            = per_block_pad;
    constexpr int log_pad           = ct_log2_pow2(pad);
    constexpr size_t smem_row       = n1 + pad;
    constexpr int log_t_quarter     = LOG_N - 3;
    constexpr int log_stride_per_pi = log_t_quarter - log_group;

    extern __shared__ uint64_t buffer[];

    const size_t pad_tid = threadIdx.x & (pad - 1);
    const size_t pad_idx = threadIdx.x >> log_pad;
    uint64_t samples[8];

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n >> 3) * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {

        const size_t twr_idx = (tid >> log_t_quarter) + start_mod_idx;
        const size_t n_idx   = tid & ((n >> 3) - 1);

        uint64_t *data_ptr        = inout + twr_idx * n;
        const uint64_t *psi       = twiddles       + twr_idx * n;
        const uint64_t *psi_shoup = twiddles_shoup + twr_idx * n;
        const uint64_t mod        = modulus[twr_idx].value();
        const bool use_lazy       = (mod < (1ULL << kNttLazyPrimeBits));

        const size_t n_init = (pad_idx << log_stride_per_pi) + pad_tid +
                              ((n_idx >> (log_group + log_pad)) << log_pad);

        #pragma unroll
        for (int j = 0; j < 8; j++)
            samples[j] = *(data_ptr + n_init + ((size_t)j << log_t_quarter));

        const size_t tw_idx = 1;
        if (use_lazy) fntt8_lazy(samples, psi, psi_shoup, tw_idx, mod);
        else          fntt8     (samples, psi, psi_shoup, tw_idx, mod);

        #pragma unroll
        for (int j = 0; j < 8; j++)
            buffer[pad_tid * smem_row + pad_idx + group * j] = samples[j];
        __syncthreads();

        size_t remain_iters = 0;
        #pragma unroll
        for (size_t j = 8, k = group >> 1, log_k = log_group - 1;
             j < group + 1;
             j *= 8, k >>= 3, log_k -= 3) {
            const size_t k4     = k >> 2;
            const size_t m_idx2 = (log_k >= 2) ? (pad_idx >> (log_k - 2)) : 0;
            const size_t t_idx2 = pad_idx & (k4 - 1);
            #pragma unroll
            for (int l = 0; l < 8; l++)
                samples[l] = buffer[smem_row * pad_tid + 2 * m_idx2 * k + t_idx2 + k4 * l];
            const size_t tw_idx2 = j * tw_idx + m_idx2;
            if (use_lazy) fntt8_lazy(samples, psi, psi_shoup, tw_idx2, mod);
            else          fntt8     (samples, psi, psi_shoup, tw_idx2, mod);
            #pragma unroll
            for (int l = 0; l < 8; l++)
                buffer[smem_row * pad_tid + 2 * m_idx2 * k + t_idx2 + k4 * l] = samples[l];
            if (j == (group >> 1)) remain_iters = 1;
            if (j == (group >> 2)) remain_iters = 2;
            __syncthreads();
        }
        if constexpr (group < 8) {
            remain_iters = (group == 4) ? 2 : 1;
        }

        #pragma unroll
        for (int l = 0; l < 8; l++)
            samples[l] = buffer[smem_row * pad_tid + 8 * pad_idx + l];

        if (remain_iters == 1) {
            const size_t tw_idx2 = (group << 2) * tw_idx + (pad_idx << 2);
            if (use_lazy) {
                ct_butterfly_lazy(samples[0], samples[1], psi[tw_idx2],     psi_shoup[tw_idx2],     mod);
                ct_butterfly_lazy(samples[2], samples[3], psi[tw_idx2 + 1], psi_shoup[tw_idx2 + 1], mod);
                ct_butterfly_lazy(samples[4], samples[5], psi[tw_idx2 + 2], psi_shoup[tw_idx2 + 2], mod);
                ct_butterfly_lazy(samples[6], samples[7], psi[tw_idx2 + 3], psi_shoup[tw_idx2 + 3], mod);
            } else {
                ct_butterfly(samples[0], samples[1], psi[tw_idx2],     psi_shoup[tw_idx2],     mod);
                ct_butterfly(samples[2], samples[3], psi[tw_idx2 + 1], psi_shoup[tw_idx2 + 1], mod);
                ct_butterfly(samples[4], samples[5], psi[tw_idx2 + 2], psi_shoup[tw_idx2 + 2], mod);
                ct_butterfly(samples[6], samples[7], psi[tw_idx2 + 3], psi_shoup[tw_idx2 + 3], mod);
            }
        } else if (remain_iters == 2) {
            const size_t tw_idx2 = (group << 1) * tw_idx + (pad_idx << 1);
            if (use_lazy) {
                fntt4_lazy(samples,     psi, psi_shoup, tw_idx2,     mod);
                fntt4_lazy(samples + 4, psi, psi_shoup, tw_idx2 + 1, mod);
            } else {
                fntt4(samples,     psi, psi_shoup, tw_idx2,     mod);
                fntt4(samples + 4, psi, psi_shoup, tw_idx2 + 1, mod);
            }
        }

        if (use_lazy) {
            #pragma unroll
            for (int l = 0; l < 8; l++)
                samples[l] = lazy_reduce_to_2q(samples[l], mod);
        }

        #pragma unroll
        for (int l = 0; l < 8; l++)
            buffer[smem_row * pad_tid + 8 * pad_idx + l] = samples[l];
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < 8; j++)
            *(data_ptr + n_init + ((size_t)j << log_t_quarter))
                = buffer[pad_tid * smem_row + pad_idx + group * j];
    }
}

// -----------------------------------------------------------------------------
// Forward NTT phase 2 (row NTTs of size n2). In-place. The Epilogue functor is
// invoked per element with (uint64_t reduced_value, size_t global_idx,
// size_t twr_idx) and is responsible for writing back to global memory.
//
// The default Store epilogue just writes to data_ptr[n_init + j*(n/8)].
// The keyswitching kernels use a custom epilogue that fuses the moddown.
// -----------------------------------------------------------------------------
struct StoreEpilogue {
    template <int LOG_N>
    __device__ __forceinline__ void operator()(uint64_t v, uint64_t *data_ptr,
                                               size_t local_idx,
                                               size_t /*twr_idx*/,
                                               uint64_t /*mod*/) const {
        data_ptr[local_idx] = v;
    }
};

// Epilogue for the keyswitch moddown kernel. Performs
//   ct[twr][i] = (cx[twr][i] - NTT(delta)[twr][i]) * bigPInv[twr] mod q[twr]
// directly from the post-NTT register value, avoiding a write-back to delta.
struct FuseModDownEpilogue {
    uint64_t *ct;
    const uint64_t *cx;
    const uint64_t *bigPInv;
    const uint64_t *bigPInv_shoup;

    template <int LOG_N>
    __device__ __forceinline__ void operator()(uint64_t v, uint64_t * /*data_ptr*/,
                                               size_t local_idx,
                                               size_t twr_idx,
                                               uint64_t mod) const {
        constexpr size_t n = 1ULL << LOG_N;
        const size_t g = twr_idx * n + local_idx;
        ct[g] = phantom::arith::sub_negate_const_mult(v, cx[g],
                                                      bigPInv[twr_idx],
                                                      bigPInv_shoup[twr_idx], mod);
    }
};

template <int LOG_N1, int LOG_N2, typename Epilogue = StoreEpilogue>
__global__ __launch_bounds__(128, 1)
void fnwt_phase2(uint64_t *data_ptr_base,
                 const uint64_t *twiddles,
                 const uint64_t *twiddles_shoup,
                 const DModulus *modulus,
                 size_t coeff_mod_size,
                 size_t start_mod_idx,
                 Epilogue epilogue = {}) {
    constexpr int LOG_N         = LOG_N1 + LOG_N2;
    constexpr size_t n          = 1ULL << LOG_N;
    constexpr size_t n1         = 1ULL << LOG_N1;
    constexpr size_t n2         = 1ULL << LOG_N2;
    constexpr int log_t         = LOG_N2 - 1;
    constexpr int log_t_quarter = LOG_N2 - 3;
    constexpr size_t t          = 1ULL << log_t;
    constexpr int log_group     = LOG_N2 - 3;
    constexpr size_t smem_set   = smem_padded_per_set(n2);

    extern __shared__ uint64_t buffer[];
    const size_t set = threadIdx.x >> log_group;
    uint64_t *my_smem = buffer + set * smem_set;
    uint64_t samples[8];

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n >> 3) * coeff_mod_size;
         tid += blockDim.x * gridDim.x) {

        const size_t twr_idx = coeff_mod_size - 1 - (tid >> (LOG_N - 3)) + start_mod_idx;
        const size_t n_idx   = tid & ((n >> 3) - 1);
        const size_t m_idx   = n_idx >> log_t_quarter;
        const size_t t_idx   = n_idx & ((1ULL << log_t_quarter) - 1);

        uint64_t *data_ptr        = data_ptr_base + twr_idx * n;
        const uint64_t mod        = modulus[twr_idx].value();
        const uint64_t *psi       = twiddles       + n * twr_idx;
        const uint64_t *psi_shoup = twiddles_shoup + n * twr_idx;
        const bool use_lazy       = (mod < (1ULL << kNttLazyPrimeBits));

        const size_t n_init = (m_idx << (log_t + 1)) + t_idx;

        #pragma unroll
        for (int j = 0; j < 8; j++)
            samples[j] = *(data_ptr + n_init + ((size_t)j << log_t_quarter));

        const size_t tw_idx = n1 + m_idx;
        if (use_lazy) fntt8_lazy(samples, psi, psi_shoup, tw_idx, mod);
        else          fntt8     (samples, psi, psi_shoup, tw_idx, mod);

        #pragma unroll
        for (int j = 0; j < 8; j++)
            my_smem[smem_pad_addr(t_idx + ((size_t)j << log_t_quarter))] = samples[j];
        __syncthreads();

        size_t tail = 0;
        #pragma unroll
        for (size_t j = 8, k = t >> 3, log_k = log_t - 3;
             j < (t >> 2) + 1;
             j *= 8, k >>= 3, log_k -= 3) {
            const size_t k4     = k >> 2;
            const size_t m_idx2 = (log_k >= 2) ? (t_idx >> (log_k - 2)) : 0;
            const size_t t_idx2 = t_idx & (k4 - 1);
            #pragma unroll
            for (int l = 0; l < 8; l++)
                samples[l] = my_smem[smem_pad_addr(2 * m_idx2 * k + t_idx2 + k4 * l)];
            const size_t tw_idx2 = j * tw_idx + m_idx2;
            if (use_lazy) fntt8_lazy(samples, psi, psi_shoup, tw_idx2, mod);
            else          fntt8     (samples, psi, psi_shoup, tw_idx2, mod);
            #pragma unroll
            for (int l = 0; l < 8; l++)
                my_smem[smem_pad_addr(2 * m_idx2 * k + t_idx2 + k4 * l)] = samples[l];
            if (j == (t >> 3)) tail = 1;
            if (j == (t >> 4)) tail = 2;
            __syncthreads();
        }

        #pragma unroll
        for (int l = 0; l < 8; l++)
            samples[l] = my_smem[smem_pad_addr(8 * t_idx + l)];

        if (tail == 1) {
            const size_t tw_idx2 = t * tw_idx + (t_idx << 2);
            if (use_lazy) {
                ct_butterfly_lazy(samples[0], samples[1], psi[tw_idx2],     psi_shoup[tw_idx2],     mod);
                ct_butterfly_lazy(samples[2], samples[3], psi[tw_idx2 + 1], psi_shoup[tw_idx2 + 1], mod);
                ct_butterfly_lazy(samples[4], samples[5], psi[tw_idx2 + 2], psi_shoup[tw_idx2 + 2], mod);
                ct_butterfly_lazy(samples[6], samples[7], psi[tw_idx2 + 3], psi_shoup[tw_idx2 + 3], mod);
            } else {
                ct_butterfly(samples[0], samples[1], psi[tw_idx2],     psi_shoup[tw_idx2],     mod);
                ct_butterfly(samples[2], samples[3], psi[tw_idx2 + 1], psi_shoup[tw_idx2 + 1], mod);
                ct_butterfly(samples[4], samples[5], psi[tw_idx2 + 2], psi_shoup[tw_idx2 + 2], mod);
                ct_butterfly(samples[6], samples[7], psi[tw_idx2 + 3], psi_shoup[tw_idx2 + 3], mod);
            }
        } else if (tail == 2) {
            const size_t tw_idx2 = (t >> 1) * tw_idx + (t_idx << 1);
            if (use_lazy) {
                fntt4_lazy(samples,     psi, psi_shoup, tw_idx2,     mod);
                fntt4_lazy(samples + 4, psi, psi_shoup, tw_idx2 + 1, mod);
            } else {
                fntt4(samples,     psi, psi_shoup, tw_idx2,     mod);
                fntt4(samples + 4, psi, psi_shoup, tw_idx2 + 1, mod);
            }
        }

        #pragma unroll
        for (int l = 0; l < 8; l++)
            my_smem[smem_pad_addr(8 * t_idx + l)] = samples[l];
        __syncthreads();

        const uint64_t mod2 = mod << 1;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t v = my_smem[smem_pad_addr(t_idx + ((size_t)j << log_t_quarter))];
            if (use_lazy) v = lazy_reduce_to_2q(v, mod);
            csub_q(v, mod2);
            csub_q(v, mod);
            const size_t local_idx = n_init + ((size_t)j << log_t_quarter);
            epilogue.template operator()<LOG_N>(v, data_ptr, local_idx, twr_idx, mod);
        }
    }
}

// -----------------------------------------------------------------------------
// Inverse NTT phase 1 (row INTTs of size n2). Templated source/dest pointers
// support both in-place (src == dst) and out-of-place (modup pre-stage).
// -----------------------------------------------------------------------------
template <int LOG_N1, int LOG_N2>
__global__ __launch_bounds__(128, 1)
void inwt_phase1_oop(const uint64_t *src,
                     uint64_t *dst,
                     const uint64_t *itwiddles,
                     const uint64_t *itwiddles_shoup,
                     const DModulus *modulus,
                     size_t coeff_mod_size,
                     size_t start_mod_idx) {
    constexpr int LOG_N         = LOG_N1 + LOG_N2;
    constexpr size_t n          = 1ULL << LOG_N;
    constexpr size_t n1         = 1ULL << LOG_N1;
    constexpr size_t n2         = 1ULL << LOG_N2;
    constexpr int log_t         = LOG_N2 - 1;
    constexpr int log_t_quarter = LOG_N2 - 3;
    constexpr size_t t          = 1ULL << log_t;
    constexpr int log_group     = LOG_N2 - 3;
    constexpr size_t group      = 1ULL << log_group;
    constexpr size_t smem_set   = smem_padded_per_set(n2);

    extern __shared__ uint64_t buffer[];
    const size_t set = threadIdx.x >> log_group;
    uint64_t *my_smem = buffer + set * smem_set;
    uint64_t samples[8];

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (n >> 3) * coeff_mod_size;
         i += blockDim.x * gridDim.x) {

        const size_t twr_idx = (i >> (LOG_N - 3)) + start_mod_idx;
        const size_t n_idx   = i & ((n >> 3) - 1);
        const size_t m_idx   = n_idx >> log_t_quarter;
        const size_t t_idx   = n_idx & ((1ULL << log_t_quarter) - 1);

        const uint64_t *src_ptr   = src + twr_idx * n;
        uint64_t *dst_ptr         = dst + twr_idx * n;
        const uint64_t *psi       = itwiddles       + n * twr_idx;
        const uint64_t *psi_shoup = itwiddles_shoup + n * twr_idx;
        const uint64_t mod        = modulus[twr_idx].value();

        const size_t n_init = (m_idx << (log_t + 1)) + t_idx;

        #pragma unroll
        for (int j = 0; j < 8; j++)
            my_smem[smem_pad_addr(t_idx + ((size_t)j << log_t_quarter))]
                = src_ptr[n_init + ((size_t)j << log_t_quarter)];
        __syncthreads();

        #pragma unroll
        for (int l = 0; l < 8; l++)
            samples[l] = my_smem[smem_pad_addr(8 * t_idx + l)];

        const size_t tw_idx = n1 + m_idx;
        size_t tw_idx2 = group * tw_idx + t_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, mod);
        #pragma unroll
        for (int l = 0; l < 8; l++)
            my_smem[smem_pad_addr(8 * t_idx + l)] = samples[l];
        __syncthreads();

        size_t tail = 0;
        #pragma unroll
        for (size_t j = t >> 5, k = 32, log_k = 5;
             j > 0;
             j >>= 3, k <<= 3, log_k += 3) {
            const size_t k4     = k >> 2;
            const size_t m_idx2 = (log_k >= 2) ? (t_idx >> (log_k - 2)) : 0;
            const size_t t_idx2 = t_idx & (k4 - 1);
            #pragma unroll
            for (int l = 0; l < 8; l++)
                samples[l] = my_smem[smem_pad_addr(2 * m_idx2 * k + t_idx2 + k4 * l)];
            tw_idx2 = j * tw_idx + m_idx2;
            intt8(samples, psi, psi_shoup, tw_idx2, mod);
            #pragma unroll
            for (int l = 0; l < 8; l++)
                my_smem[smem_pad_addr(2 * m_idx2 * k + t_idx2 + k4 * l)] = samples[l];
            if (j == 2) tail = 1;
            if (j == 4) tail = 2;
            __syncthreads();
        }
        if constexpr (group < 8) {
            tail = (group == 4) ? 2 : 1;
        }

        #pragma unroll
        for (int j = 0; j < 8; j++)
            samples[j] = my_smem[smem_pad_addr(t_idx + ((size_t)j << log_t_quarter))];

        if (tail == 1) {
            gs_butterfly(samples[0], samples[4], psi[tw_idx], psi_shoup[tw_idx], mod);
            gs_butterfly(samples[1], samples[5], psi[tw_idx], psi_shoup[tw_idx], mod);
            gs_butterfly(samples[2], samples[6], psi[tw_idx], psi_shoup[tw_idx], mod);
            gs_butterfly(samples[3], samples[7], psi[tw_idx], psi_shoup[tw_idx], mod);
        } else if (tail == 2) {
            intt4(samples,     psi, psi_shoup, tw_idx, mod);
            intt4(samples + 1, psi, psi_shoup, tw_idx, mod);
        }

        #pragma unroll
        for (int j = 0; j < 8; j++)
            dst_ptr[n_init + ((size_t)j << log_t_quarter)] = samples[j];
    }
}

// -----------------------------------------------------------------------------
// Inverse NTT phase 2 (column INTT of size n1) with N^{-1} scaling and an
// optional extra Shoup-multiplied scale (e.g. for keyswitch_scale variants).
// When ScaleEnabled = false, the extra scale arrays are ignored.
// -----------------------------------------------------------------------------
template <int LOG_N1, int LOG_N2, bool ScaleEnabled>
__global__ __launch_bounds__(((1 << LOG_N1) >> 3) * per_block_pad, 1)
void inwt_phase2(uint64_t *inout,
                 const uint64_t *itwiddles,
                 const uint64_t *itwiddles_shoup,
                 const uint64_t *inv_degree_modulo,
                 const uint64_t *inv_degree_modulo_shoup,
                 const DModulus *modulus,
                 size_t coeff_mod_size,
                 size_t start_mod_idx,
                 const uint64_t *scale,
                 const uint64_t *scale_shoup) {
    constexpr int LOG_N             = LOG_N1 + LOG_N2;
    constexpr size_t n              = 1ULL << LOG_N;
    constexpr size_t n1             = 1ULL << LOG_N1;
    constexpr int log_group         = LOG_N1 - 3;
    constexpr size_t group          = 1ULL << log_group;
    constexpr size_t pad            = per_block_pad;
    constexpr int log_pad           = ct_log2_pow2(pad);
    constexpr size_t smem_row       = n1 + pad;
    constexpr int log_t_quarter     = LOG_N - 3;
    constexpr int log_stride_per_pi = log_t_quarter - log_group;
    constexpr int log_stride_init   = LOG_N + 3 - LOG_N1;

    extern __shared__ uint64_t buffer[];
    const size_t pad_tid = threadIdx.x & (pad - 1);
    const size_t pad_idx = threadIdx.x >> log_pad;
    uint64_t samples[8];

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (n >> 3) * coeff_mod_size;
         i += blockDim.x * gridDim.x) {

        const size_t twr_idx = (i >> log_t_quarter) + start_mod_idx;
        const size_t n_idx   = i & ((n >> 3) - 1);

        uint64_t *data_ptr        = inout + twr_idx * n;
        const uint64_t *psi       = itwiddles       + n * twr_idx;
        const uint64_t *psi_shoup = itwiddles_shoup + n * twr_idx;
        const uint64_t mod        = modulus[twr_idx].value();
        const uint64_t inv_n      = inv_degree_modulo[twr_idx];
        const uint64_t inv_n_shp  = inv_degree_modulo_shoup[twr_idx];

        size_t n_init = (pad_idx << log_stride_init) + pad_tid +
                        ((n_idx >> (log_group + log_pad)) << log_pad);

        #pragma unroll
        for (int j = 0; j < 8; j++)
            samples[j] = *(data_ptr + n_init + ((size_t)j << log_stride_per_pi));

        const size_t tw_idx = 1;
        size_t tw_idx2 = group * tw_idx + pad_idx;
        intt8(samples, psi, psi_shoup, tw_idx2, mod);
        #pragma unroll
        for (int j = 0; j < 8; j++)
            buffer[pad_tid * smem_row + 8 * pad_idx + j] = samples[j];
        __syncthreads();

        size_t tail = 0;
        #pragma unroll
        for (size_t j = group >> 3, k = 32, log_k = 5;
             j > 0;
             j >>= 3, k <<= 3, log_k += 3) {
            const size_t k4     = k >> 2;
            const size_t m_idx2 = (log_k >= 2) ? (pad_idx >> (log_k - 2)) : 0;
            const size_t t_idx2 = pad_idx & (k4 - 1);
            #pragma unroll
            for (int l = 0; l < 8; l++)
                samples[l] = buffer[smem_row * pad_tid + 2 * m_idx2 * k + t_idx2 + k4 * l];
            tw_idx2 = j * tw_idx + m_idx2;
            intt8(samples, psi, psi_shoup, tw_idx2, mod);
            #pragma unroll
            for (int l = 0; l < 8; l++)
                buffer[smem_row * pad_tid + 2 * m_idx2 * k + t_idx2 + k4 * l] = samples[l];
            if (j == 2) tail = 1;
            if (j == 4) tail = 2;
            __syncthreads();
        }
        if constexpr (group < 8) {
            tail = (group == 4) ? 2 : 1;
        }

        #pragma unroll
        for (int l = 0; l < 8; l++)
            samples[l] = buffer[pad_tid * smem_row + pad_idx + group * l];

        if (tail == 1) {
            gs_butterfly(samples[0], samples[4], psi[tw_idx], psi_shoup[tw_idx], mod);
            gs_butterfly(samples[1], samples[5], psi[tw_idx], psi_shoup[tw_idx], mod);
            gs_butterfly(samples[2], samples[6], psi[tw_idx], psi_shoup[tw_idx], mod);
            gs_butterfly(samples[3], samples[7], psi[tw_idx], psi_shoup[tw_idx], mod);
        } else if (tail == 2) {
            intt4(samples,     psi, psi_shoup, tw_idx, mod);
            intt4(samples + 1, psi, psi_shoup, tw_idx, mod);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++)
            samples[j] = multiply_and_reduce_shoup_lazy(samples[j], inv_n, inv_n_shp, mod);

        n_init = (pad_idx << log_stride_per_pi) + pad_tid +
                 ((n_idx >> (log_group + log_pad)) << log_pad);

        if constexpr (ScaleEnabled) {
            const uint64_t s_q = scale[twr_idx];
            const uint64_t s_q_shoup = scale_shoup[twr_idx];
            #pragma unroll
            for (int j = 0; j < 8; j++)
                *(data_ptr + n_init + ((size_t)j << log_t_quarter))
                    = phantom::arith::multiply_and_reduce_shoup(samples[j], s_q, s_q_shoup, mod);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                csub_q(samples[j], mod);
                *(data_ptr + n_init + ((size_t)j << log_t_quarter)) = samples[j];
            }
        }
    }
}

} // namespace phantom::ntt::radix8
