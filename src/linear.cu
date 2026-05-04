#include "linear.h"

#include <stdexcept>

#include "evaluate.cuh"

namespace phantom {

    PhantomCiphertext inner_sum(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &ct,
            std::size_t block_size) {
        if (block_size == 0 || (block_size & (block_size - 1)) != 0) {
            throw std::invalid_argument("inner_sum: block_size must be a power of 2");
        }
        PhantomCiphertext acc = ct;
        std::size_t stride = 1;
        while (stride < block_size) {
            PhantomCiphertext rotated = rotate(ctx, acc, static_cast<int>(stride), galois_key);
            add_inplace(ctx, acc, rotated);
            stride <<= 1;
        }
        return acc;
    }

    PhantomCiphertext replicate(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &ct,
            std::size_t period,
            std::size_t num_slots) {
        if (period == 0 || (period & (period - 1)) != 0) {
            throw std::invalid_argument("replicate: period must be a power of 2");
        }
        if (num_slots == 0 || (num_slots & (num_slots - 1)) != 0) {
            throw std::invalid_argument("replicate: num_slots must be a power of 2");
        }
        if (period > num_slots) {
            throw std::invalid_argument("replicate: period exceeds num_slots");
        }
        PhantomCiphertext acc = ct;
        std::size_t stride = period;
        while (stride < num_slots) {
            PhantomCiphertext rotated = rotate(ctx, acc, -static_cast<int>(stride), galois_key);
            add_inplace(ctx, acc, rotated);
            stride <<= 1;
        }
        return acc;
    }

}
