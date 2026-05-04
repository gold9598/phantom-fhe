#pragma once

#include <cstddef>
#include <vector>

#include "ciphertext.h"
#include "ckks.h"
#include "context.cuh"
#include "secretkey.h"

namespace phantom {

    // Compute attention scores per packed K-cache ct.
    //
    // Inputs:
    //   q: ciphertext of Q, replicated across all slots (caller responsibility).
    //      Layout: slot[t_local * d_total + h * d_head + i] = Q[h][i] for every t_local.
    //   packed_k: each ct holds K for one or more token positions. Per ct, layout:
    //      slot[t_local * d_total + h * d_head + i] = K[t_pos+t_local][h][i].
    //   d_head: dimension per head (must be a power of 2).
    //   relin_key, galois_key: keys for ct×ct + rescale and inner-sum rotations.
    //
    // Output: one ciphertext per input K ct. Per output ct:
    //   slot[t_local * d_total + h * d_head] = Q[h] · K[t_pos+t_local][h].
    //   Other positions within each d_head block hold inner-sum junk.
    std::vector<PhantomCiphertext> compute_qkt(
            const PhantomContext &ctx,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &q,
            const std::vector<PhantomCiphertext> &packed_k,
            std::size_t d_head);

    // Compute Σ_t score[t] · V[t] across all packed-position cts.
    // Output: ciphertext with attention output in slots [0..d_total),
    // organized as slot[h * d_head + i] = Σ_t score[t][h] · V[t][h][i].
    // The result is replicated across all d_total-wide blocks (a side
    // effect of the per-chunk accumulator broadcast).
    PhantomCiphertext score_times_v(
            const PhantomContext &ctx,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const std::vector<PhantomCiphertext> &score_cts,
            const std::vector<PhantomCiphertext> &v_cts,
            const PhantomPlaintext &mask_pt,
            std::size_t d_head,
            std::size_t d_total,
            std::size_t positions_per_ct);
}
