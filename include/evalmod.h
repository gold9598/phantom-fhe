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
//
// Precision regime: a uniform 40-bit chain leaves no headroom for the 8-level
// EvalMod (~10 bits at logN=16). Lapis's production setup uses a heterogeneous
// chain — `boot_setup_4section`:
//   q_list = [msg(58) | scale(40)×N | ER(58)×9 | special(58)]
// with `engine.scale = q_msg ≈ 2^58` and a sparse hw=128 secret. EvalMod runs
// against the 58-bit ER segment; the 40-bit "scale" segment is the
// post-bootstrap user space. K=16 then reaches ~26–31 bits, K=7 ~17 bits.
// See test/evalmod_test.cu (`run_lapis_4section`).
// ----------------------------------------------------------------------------

#include "phantom.h"

#include <vector>

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

    // K=28 R=3: degree-49 polynomial (ported from the_lib bootstrap.cpp sine()).
    // Baby T_1..T_7, giant T_14/T_28, PS basis T_49. 9 levels total (6 sine + 3 DA).
    // Approximates sin(2π·28·x)/(2π) with higher precision than K=16.
    PhantomCiphertext evalmod_k28_r3(const PhantomContext &ctx,
                                     PhantomCKKSEncoder &encoder,
                                     const PhantomCiphertext &ct,
                                     const PhantomRelinKey &relin_keys);

    // K=28 R=4: same degree-49 polynomial as K=28 R=3 but with one extra
    // double-angle iteration. 10 levels total (6 sine + 4 DA). Effective K
    // amplification is 28·2^4 = 448 (vs 28·2^3 = 224 for R=3). Reaches ~30
    // bits per-slot precision vs ~27 bits for R=3.
    PhantomCiphertext evalmod_k28_r4(const PhantomContext &ctx,
                                     PhantomCKKSEncoder &encoder,
                                     const PhantomCiphertext &ct,
                                     const PhantomRelinKey &relin_keys);

#ifdef EVALMOD_STAGE_DEBUG
    // DEBUG-ONLY: when phantom is built with -DEVALMOD_STAGE_DEBUG=1, the
    // K=28 R=3 sine kernel decrypts and measures bits-of-precision at each
    // stage (basis, T_14, T_28, T_21, T_49, aux_ct, quotient, remainder,
    // aux*quotient, sine_final). Call set_stage_probe with the secret key
    // and the analytical input vector (one value per slot in [0..N-1]) BEFORE
    // calling evalmod_k28_r3, then clear with clear_stage_probe.
    void evalmod_set_stage_probe(const PhantomSecretKey *sk,
                                 const std::vector<double> &vals);
    void evalmod_clear_stage_probe();
#endif

}  // namespace phantom
