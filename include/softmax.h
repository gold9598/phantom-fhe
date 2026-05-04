#pragma once

#include <cstddef>
#include <vector>

#include "ciphertext.h"
#include "ckks.h"
#include "context.cuh"
#include "secretkey.h"

namespace phantom {

    // Compute (extra_scale * T^(-1/2^k)) * sum_{i=0..deg} c_i * y^i where y = x/2^k,
    // c_i are the deg-5 Chebyshev fit of exp(y) on [-2, 2].
    PhantomCiphertext ps_exp_init(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomCiphertext &scores,
            std::size_t num_tokens,
            std::size_t num_squarings,
            double extra_scale);

    // Square ct k times in place (mul_and_relin + rescale per iter).
    void square_iterations_inplace(
            const PhantomContext &ctx,
            const PhantomRelinKey &relin_key,
            PhantomCiphertext &ct,
            std::size_t num_squarings);

    // Damped variant of square_iterations_inplace. After each ct² + rescale,
    // multiplies by damps[i] (also +1 level for the scalar mul). Total levels:
    // 2 * num_squarings (vs num_squarings for the un-damped variant).
    // Mathematically equivalent to un-damped after Goldschmidt cancels the
    // accumulated damping constant — see softmax.rs:90 for the cancellation
    // derivation.
    void square_iterations_damped_inplace(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            PhantomCiphertext &ct,
            const std::vector<double> &damps);

    // Joint Goldschmidt iteration: refines (x, a) -> (e_hat/a_hat, 1).
    PhantomCiphertext softmax_correct(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomCiphertext &e_ct,
            const PhantomCiphertext &a_ct,
            std::size_t iters);

    // End-of-pipeline: e_ct holds (extra_scale^(2^k) * exp(x)/T); finalize via
    // sum_reduce + softmax_correct.
    PhantomCiphertext finalize_softmax(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &e_ct,
            std::size_t num_tokens,
            std::size_t stride,
            std::size_t iters);

}
