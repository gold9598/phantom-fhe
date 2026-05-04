#pragma once

#include <cstddef>
#include <vector>

#include "ciphertext.h"
#include "ckks.h"
#include "context.cuh"
#include "secretkey.h"

namespace phantom {

    // Tree-reduce sum across each block_size-wide block. Result: each block's
    // slot-0 holds Σ block; other slots within the block hold partial-sum junk.
    // block_size must be a power of 2. Required galois steps: {1, 2, 4, ..., block_size/2}.
    PhantomCiphertext inner_sum(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &ct,
            std::size_t block_size);

    // Broadcast slot[0..period) of `ct` to fill all slots in periods of `period`.
    // Returns a new ct with `slot[k*period + i] = ct_original[i]` for all valid k, i.
    PhantomCiphertext replicate(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &ct,
            std::size_t period,
            std::size_t num_slots);

}
