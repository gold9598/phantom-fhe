#pragma once

#include "uintmodmath.cuh"

namespace phantom::arith {
    // Threshold (in bits) below which lazy butterflies are safe.
    // Lapis' rule: starting from K=2 (Shoup output range), each radix-8 lazy pass
    // grows the upper bound by +6q. With q < 2^59, K can grow to ~31 without
    // overflowing u64 (since (K+2)*q must fit in 2^64). For 50/54-bit primes that
    // is plenty for 16K-degree NTTs (<= 5 lazy radix-8 passes).
    constexpr int kNttLazyPrimeBits = 59;

    /** Computer one butterfly in forward NTT
     * x[0] = x[0] + pow * x[1] % mod
     * x[1] = x[0] - pow * x[1] % mod
     */
    __device__ __forceinline__ void ct_butterfly(uint64_t& x, uint64_t& y,
                                                 const uint64_t& tw, const uint64_t& tw_shoup,
                                                 const uint64_t& mod) {
        // const uint64_t tw_y = multiply_and_reduce_shoup_lazy(y, tw, tw_shoup, mod);
        const uint64_t hi = __umul64hi(y, tw_shoup);
        const uint64_t tw_y = y * tw - hi * mod;
        // csub_q(x, mod2);
        const uint64_t mod2 = mod << 1;
        const uint64_t tmp = x - mod2;
        x = tmp + (tmp >> 63) * mod2;
        y = x + mod2 - tw_y;
        x += tw_y;
    }

    // Lazy CT butterfly: skips the input csub_q. Input range [0, Kq); output [0, (K+2)q).
    // Caller must apply lazy_reduce_to_2q at phase boundary to bring K back to 2.
    __device__ __forceinline__ void ct_butterfly_lazy(uint64_t& x, uint64_t& y,
                                                      const uint64_t& tw, const uint64_t& tw_shoup,
                                                      const uint64_t& mod) {
        const uint64_t hi = __umul64hi(y, tw_shoup);
        const uint64_t tw_y = y * tw - hi * mod; // [0, 2q)
        const uint64_t mod2 = mod << 1;
        y = x + mod2 - tw_y;                      // [0, (K+2)q)
        x += tw_y;                                // [0, (K+2)q)
    }

    // Reduce x in [0, Kq) to [0, 2q) for K <= 32 using a 4-step branchless cascade.
    // Cost: 4 sub + 4 select + 4 shift; no divisions.
    __device__ __forceinline__ uint64_t lazy_reduce_to_2q(uint64_t x, uint64_t mod) {
        uint64_t v = mod << 4; // 16q
        int64_t t;
        t = static_cast<int64_t>(x - v); x = (t < 0) ? x : static_cast<uint64_t>(t);
        v >>= 1;               // 8q
        t = static_cast<int64_t>(x - v); x = (t < 0) ? x : static_cast<uint64_t>(t);
        v >>= 1;               // 4q
        t = static_cast<int64_t>(x - v); x = (t < 0) ? x : static_cast<uint64_t>(t);
        v >>= 1;               // 2q
        t = static_cast<int64_t>(x - v); x = (t < 0) ? x : static_cast<uint64_t>(t);
        return x;
    }

    /** Computer one butterfly in inverse NTT
     * x[0] = (x[0] + pow * x[1]) / 2 % mod
     * x[1] = (x[0] - pow * x[1]) / 2 % mod
     */
    __device__ __forceinline__ void gs_butterfly(uint64_t& x, uint64_t& y,
                                                 const uint64_t& tw, const uint64_t& tw_shoup,
                                                 const uint64_t& mod) {
        const uint64_t mod2 = 2 * mod;
        const uint64_t t = x + mod2 - y; // [0, 4q)
        uint64_t s = x + y; // [0, 4q)
        csub_q(s, mod2); // [0, 2q)
        x = s;
        y = multiply_and_reduce_shoup_lazy(t, tw, tw_shoup, mod);
    }

    __device__ __forceinline__ void fntt8(uint64_t* s,
                                          const uint64_t* tw,
                                          const uint64_t* tw_shoup,
                                          uint64_t tw_idx,
                                          uint64_t mod) {
        // stage 1
        ct_butterfly(s[0], s[4], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly(s[1], s[5], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly(s[2], s[6], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly(s[3], s[7], tw[tw_idx], tw_shoup[tw_idx], mod);
        // stage 2
        ct_butterfly(s[0], s[2], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        ct_butterfly(s[1], s[3], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        ct_butterfly(s[4], s[6], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
        ct_butterfly(s[5], s[7], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
        // stage 3
        ct_butterfly(s[0], s[1], tw[4 * tw_idx], tw_shoup[4 * tw_idx], mod);
        ct_butterfly(s[2], s[3], tw[4 * tw_idx + 1], tw_shoup[4 * tw_idx + 1], mod);
        ct_butterfly(s[4], s[5], tw[4 * tw_idx + 2], tw_shoup[4 * tw_idx + 2], mod);
        ct_butterfly(s[6], s[7], tw[4 * tw_idx + 3], tw_shoup[4 * tw_idx + 3], mod);
    }

    __device__ __forceinline__ void fntt4(uint64_t* s,
                                          const uint64_t* tw,
                                          const uint64_t* tw_shoup,
                                          uint64_t tw_idx,
                                          uint64_t mod) {
        // stage 1
        ct_butterfly(s[0], s[2], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly(s[1], s[3], tw[tw_idx], tw_shoup[tw_idx], mod);
        // stage 2
        ct_butterfly(s[0], s[1], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        ct_butterfly(s[2], s[3], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
    }

    // Lazy radix-8 forward (one pass). Input s[0..7] in [0, Kq); output in [0, (K+6)q).
    __device__ __forceinline__ void fntt8_lazy(uint64_t* s,
                                               const uint64_t* tw,
                                               const uint64_t* tw_shoup,
                                               uint64_t tw_idx,
                                               uint64_t mod) {
        // stage 1 (stride 4, single twiddle)
        ct_butterfly_lazy(s[0], s[4], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly_lazy(s[1], s[5], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly_lazy(s[2], s[6], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly_lazy(s[3], s[7], tw[tw_idx], tw_shoup[tw_idx], mod);
        // stage 2 (stride 2)
        const uint64_t tw1 = tw[(tw_idx << 1)];
        const uint64_t tw1s = tw_shoup[(tw_idx << 1)];
        const uint64_t tw1b = tw[(tw_idx << 1) + 1];
        const uint64_t tw1bs = tw_shoup[(tw_idx << 1) + 1];
        ct_butterfly_lazy(s[0], s[2], tw1, tw1s, mod);
        ct_butterfly_lazy(s[1], s[3], tw1, tw1s, mod);
        ct_butterfly_lazy(s[4], s[6], tw1b, tw1bs, mod);
        ct_butterfly_lazy(s[5], s[7], tw1b, tw1bs, mod);
        // stage 3 (stride 1)
        const uint64_t i2 = tw_idx << 2;
        ct_butterfly_lazy(s[0], s[1], tw[i2],     tw_shoup[i2],     mod);
        ct_butterfly_lazy(s[2], s[3], tw[i2 + 1], tw_shoup[i2 + 1], mod);
        ct_butterfly_lazy(s[4], s[5], tw[i2 + 2], tw_shoup[i2 + 2], mod);
        ct_butterfly_lazy(s[6], s[7], tw[i2 + 3], tw_shoup[i2 + 3], mod);
    }

    __device__ __forceinline__ void fntt4_lazy(uint64_t* s,
                                               const uint64_t* tw,
                                               const uint64_t* tw_shoup,
                                               uint64_t tw_idx,
                                               uint64_t mod) {
        ct_butterfly_lazy(s[0], s[2], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly_lazy(s[1], s[3], tw[tw_idx], tw_shoup[tw_idx], mod);
        const uint64_t i2 = tw_idx << 1;
        ct_butterfly_lazy(s[0], s[1], tw[i2],     tw_shoup[i2],     mod);
        ct_butterfly_lazy(s[2], s[3], tw[i2 + 1], tw_shoup[i2 + 1], mod);
    }

    __device__ __forceinline__ void intt8(uint64_t* s,
                                          const uint64_t* tw,
                                          const uint64_t* tw_shoup,
                                          uint64_t tw_idx,
                                          uint64_t mod) {
        // stage 1
        gs_butterfly(s[0], s[1], tw[4 * tw_idx], tw_shoup[4 * tw_idx], mod);
        gs_butterfly(s[2], s[3], tw[4 * tw_idx + 1], tw_shoup[4 * tw_idx + 1], mod);
        gs_butterfly(s[4], s[5], tw[4 * tw_idx + 2], tw_shoup[4 * tw_idx + 2], mod);
        gs_butterfly(s[6], s[7], tw[4 * tw_idx + 3], tw_shoup[4 * tw_idx + 3], mod);

        // stage 2
        gs_butterfly(s[0], s[2], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        gs_butterfly(s[1], s[3], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        gs_butterfly(s[4], s[6], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
        gs_butterfly(s[5], s[7], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);

        // stage 3
        gs_butterfly(s[0], s[4], tw[tw_idx], tw_shoup[tw_idx], mod);
        gs_butterfly(s[1], s[5], tw[tw_idx], tw_shoup[tw_idx], mod);
        gs_butterfly(s[2], s[6], tw[tw_idx], tw_shoup[tw_idx], mod);
        gs_butterfly(s[3], s[7], tw[tw_idx], tw_shoup[tw_idx], mod);
    }

    __device__ __forceinline__ void intt4(uint64_t* s,
                                          const uint64_t* tw,
                                          const uint64_t* tw_shoup,
                                          uint64_t tw_idx,
                                          uint64_t mod) {
        // stage 1
        gs_butterfly(s[0], s[2], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        gs_butterfly(s[4], s[6], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
        // stage 2
        gs_butterfly(s[0], s[4], tw[tw_idx], tw_shoup[tw_idx], mod);
        gs_butterfly(s[2], s[6], tw[tw_idx], tw_shoup[tw_idx], mod);
    }
}
