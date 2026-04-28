#pragma once

// ----------------------------------------------------------------------------
// CKKS EvalMod (K=7, R=3): evaluate sin(2π·K·x)/(2π) under encryption.
//
// Ported from lapis/fhe/src/engine/evalmod.rs (sine_chebyshev_k7 + R=3 DA).
// Uses phantom's public CKKS API (multiply_and_relin / multiply_plain /
// rescale_to_next / add / sub / add_plain).
//
// Algorithm (K=7, degree-15 Chebyshev sine approximation + 3 double-angle):
//   1. Build Chebyshev basis T_1..T_8 (3 levels)
//   2. low  = c_0 + Σ_{i=1..7} K7_LOW[i] T_i + K7_LOW[8] T_8       (1 level)
//   3. high = Σ_{j=1..7} K7_HIGH[j] T_j                            (1 level, parallel)
//   4. result = 2·T_8·high + low                                   (1 level)
//   5. for j in 1..=R: result ← 2·result² − (2π)^{−2^{j−R}}        (R = 3 levels)
//
// Total: 5 + 3 = 8 levels. Input ct must be at chain_index ≤ L−8 with
// magnitude in [-1, 1] (Chebyshev range).
// ----------------------------------------------------------------------------

#include "phantom.h"

namespace phantom {

    PhantomCiphertext evalmod_k7_r3(const PhantomContext &ctx,
                                    PhantomCKKSEncoder &encoder,
                                    const PhantomCiphertext &ct,
                                    const PhantomRelinKey &relin_keys);

    // K=16 R=3: degree-31 polynomial, 33.9-bit polynomial precision, 9 levels.
    // After R=3 DA the result approximates sin(2π·16·x)/(2π).
    PhantomCiphertext evalmod_k16_r3(const PhantomContext &ctx,
                                     PhantomCKKSEncoder &encoder,
                                     const PhantomCiphertext &ct,
                                     const PhantomRelinKey &relin_keys);

}  // namespace phantom
