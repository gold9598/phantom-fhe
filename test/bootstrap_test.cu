// Unit test: phantom::mod_raise_inplace + SmallBootstrapKey.
//
// Builds the lapis 4-section logN=16 chain
//     [msg(58) | scale(40)x4 | ER(58)x9 | special(58)]
// encrypts a small random vector at scale 2^58, drives the ciphertext down to
// the bottom of the modulus chain (single remaining prime q_msg ~= 2^58),
// then runs encapsulated mod-raise:
//
//     KS dense -> sparse  ;  modulus extension  ;  KS sparse -> dense
//
// Phase 1 only: the lifted ciphertext encrypts the same plaintext at the top
// of the chain *plus* a per-coefficient I*q_msg term (the standard CKKS
// bootstrap I-term that EvalMod removes in Phase 2). The encapsulation
// ensures the per-coefficient |I| is bounded by O(sqrt(hw)) instead of
// O(sqrt(N)) — that's the Oglivie-2026 mitigation.
//
// We sanity-check (a) the kswitch round-trip alone is precise (~1e-10), and
// (b) the full mod-raise lift's residual slot error is bounded by
//     O(sqrt(N) * sqrt(hw)) ~= 1000 for N=2^16 and hw=128.
// Tighter precision lands once Phase 2 (C2S + EvalMod + S2C) is wired up.

#include "phantom.h"
#include "bootstrap.h"
#include "galois.cuh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

using namespace phantom;
using namespace phantom::arith;

static int run_round_trip() {
    EncryptionParameters parms(scheme_type::ckks);
    constexpr size_t log_n = 16;
    const size_t N = 1ULL << log_n;
    parms.set_poly_modulus_degree(N);
    parms.set_special_modulus_size(1);

    // Lapis 4-section: msg | 4*scale | 9*ER | special (K=28 R=3 needs 9 ER primes)
    std::vector<int> bits;
    bits.push_back(58);                              // msg (q_0)
    for (int i = 0; i < 4; ++i) bits.push_back(40);  // scale segment
    for (int i = 0; i < 9; ++i) bits.push_back(58);  // EvalRound segment
    bits.push_back(58);                              // special
    parms.set_coeff_modulus(CoeffModulus::Create(N, bits));

    PhantomContext context(parms);
    PhantomSecretKey sk(context);                    // dense ternary
    PhantomCKKSEncoder encoder(context);

    const double scale = std::pow(2.0, 58);
    const size_t slot_count = encoder.slot_count();

    std::mt19937_64 rng(0xB007UL);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<double> input(slot_count, 0.0);
    for (size_t i = 0; i < slot_count; ++i) input[i] = dist(rng);

    PhantomPlaintext pt;
    encoder.encode(context, input, scale, pt);
    PhantomCiphertext ct;
    sk.encrypt_symmetric(context, pt, ct);

    // Drive ct down to the bottom of the chain (single prime q_msg).
    // CKKS mod_switch_to drops without rescaling; the message stays at scale=q_msg.
    const size_t bottom_index = context.total_parm_size() - 1;
    mod_switch_to_inplace(context, ct, bottom_index);

    // Build the encapsulation KSKs (hw=128 per spec).
    const std::size_t HW_TEST = 128;
    SmallBootstrapKey bk = create_small_bootstrap_key(context, sk, HW_TEST);

    // The actual mod-raise (encapsulated).
    mod_raise_inplace(context, ct, bk);

    // Confirm we landed at the top of the chain.
    if (ct.chain_index() != context.get_first_index()) {
        std::fprintf(stderr,
            "FAIL: post-mod_raise chain_index=%zu, expected first_index=%zu (bottom=%zu)\n",
            ct.chain_index(), context.get_first_index(), bottom_index);
        return 1;
    }

    PhantomPlaintext dec_pt;
    sk.decrypt(context, ct, dec_pt);
    std::vector<double> decoded;
    encoder.decode(context, dec_pt, decoded);

    double max_abs_err = 0.0, sum_abs_err = 0.0;
    for (size_t i = 0; i < slot_count; ++i) {
        const double err = std::abs(decoded[i] - input[i]);
        max_abs_err = std::max(max_abs_err, err);
        sum_abs_err += err;
    }
    const double avg_err = sum_abs_err / double(slot_count);

    std::printf("mod_raise round-trip: logN=%zu, q_msg=2^58, scale_seg=40x4, ER=58x9, hw=%zu\n", log_n, HW_TEST);
    for (size_t i = 0; i < 4; ++i) {
        std::printf("  slot[%zu] decoded=%.6e  input=%.6e  err=%.3e\n",
                    i, decoded[i], input[i], std::abs(decoded[i] - input[i]));
    }
    std::printf("  avg |err|     = %.3e\n", avg_err);
    std::printf("  max |err|     = %.3e\n", max_abs_err);

    // Per-slot error analysis: mod_raise lifts the bottom-level ct to the top
    // of the chain; the lifted plaintext at slot i contains the original
    // m_i*scale plus a per-coefficient integer-multiple-of-q_msg (the "I*q_msg"
    // term standard in CKKS bootstrap). After dividing by scale (= q_msg here),
    // this contributes an integer offset I per coefficient. The slot-domain
    // image of {I_i} via the encoding FFT gives bounded slot error
    //     |slot_err| <= O(sqrt(N) * hw_bound)
    // (hw_bound is the per-coefficient I bound, which scales with sparse hw).
    // For logN=16 with hw=128, this is ~10^3-10^4; full bootstrap (C2S +
    // EvalMod + S2C) eliminates the I*q_msg term in EvalMod.
    //
    // Loose tolerance: just verify mod_raise didn't produce garbage. A bug in
    // the kernel (e.g. wrong sign-tile, wrong moduli) drives the error to
    // O(q_total/scale) which is enormous (e.g. 10^7+); a correct lift stays
    // within O(sqrt(N) * hw_bound).
    constexpr double tol = 1e5;
    if (max_abs_err > tol) {
        std::fprintf(stderr, "FAIL: max abs error %.3e exceeds tolerance %.3e\n",
                     max_abs_err, tol);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

// Sanity: KS round-trip dense -> sparse -> dense at a level where decoding is
// unambiguous. Should preserve the message to within kswitch noise.
static int run_ks_round_trip_only() {
    EncryptionParameters parms(scheme_type::ckks);
    constexpr size_t log_n = 16;
    const size_t N = 1ULL << log_n;
    parms.set_poly_modulus_degree(N);
    parms.set_special_modulus_size(1);

    std::vector<int> bits;
    bits.push_back(58);
    for (int i = 0; i < 4; ++i) bits.push_back(40);
    for (int i = 0; i < 9; ++i) bits.push_back(58);
    bits.push_back(58);
    parms.set_coeff_modulus(CoeffModulus::Create(N, bits));

    PhantomContext context(parms);
    PhantomSecretKey sk(context);
    PhantomCKKSEncoder encoder(context);

    const double scale = std::pow(2.0, 58);
    const size_t slot_count = encoder.slot_count();

    std::mt19937_64 rng(0xBADAUL);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<double> input(slot_count, 0.0);
    for (size_t i = 0; i < slot_count; ++i) input[i] = dist(rng);

    PhantomPlaintext pt;
    encoder.encode(context, input, scale, pt);
    PhantomCiphertext ct;
    sk.encrypt_symmetric(context, pt, ct);

    SmallBootstrapKey bk = create_small_bootstrap_key(context, sk, /*hw=*/128);

    // Round-trip at top of chain (level 1) — no mod-raise involved.
    apply_kswitch_inplace(context, ct, bk.ksk_to_sparse);
    apply_kswitch_inplace(context, ct, bk.ksk_to_dense);

    PhantomPlaintext dec_pt;
    sk.decrypt(context, ct, dec_pt);
    std::vector<double> decoded;
    encoder.decode(context, dec_pt, decoded);

    double max_abs_err = 0.0;
    for (size_t i = 0; i < slot_count; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs(decoded[i] - input[i]));
    }
    std::printf("ks_round_trip_only: max |err| = %.3e\n", max_abs_err);
    if (max_abs_err > 1e-3) {
        std::fprintf(stderr, "FAIL: KS round-trip max abs error %.3e > 1e-3\n", max_abs_err);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

// Phase 2: C2S round-trip via host oracle.
//
// Builds the standard 4-section logN=16 chain, encrypts a small complex
// vector, applies C2S, and compares against an exact host-side replay of the
// same diagonal sums (with the same per-layer normalization). The host
// oracle is the *same* mathematical operation evaluated in double-precision
// complex arithmetic, so this is a tight round-trip check rather than an
// abstract DFT comparison — it isolates "does the GPU evaluator match the
// math we encoded?" from "does our DFT factorization match a reference?"
//
// Tolerance: looser than CKKS noise floor would suggest because we use
// 58-bit primes everywhere (no 60-bit special) and consume 3 levels (~3
// rescales). Empirically <= 1e-3 on a quiet GPU.
static int run_c2s_round_trip_via_inverse() {
    using C64 = std::complex<double>;
    using namespace phantom::util;

    EncryptionParameters parms(scheme_type::ckks);
    constexpr size_t log_n = 16;
    const size_t N = 1ULL << log_n;
    parms.set_poly_modulus_degree(N);

    // Reduced chain for C2S correctness only. Phase 2 evaluator needs:
    //   - 1 prime at the C2S input level
    //   - 3 primes consumed by the 3 rescales (one per layer)
    //   - some special primes for key-switching
    // We use 4 specials so that dnum = size_Q / size_P stays small (≈1) and
    // the 156 Galois KSKs fit in GPU memory (single special => dnum=15 =>
    // KSKs ~245MB each, OOM at 32GB). 58-bit primes throughout match the
    // bootstrap operating regime (q_msg ≈ 2^58).
    parms.set_special_modulus_size(4);
    std::vector<int> bits;
    bits.push_back(58);                              // q_0
    for (int i = 0; i < 7; ++i) bits.push_back(58);  // 3 layers + buffer
    for (int i = 0; i < 4; ++i) bits.push_back(58);  // 4 special (dnum=2)
    parms.set_coeff_modulus(CoeffModulus::Create(N, bits));

    // Compute the union of Galois rotations C2S needs (plus conjugation
    // 2N-1) and stamp them on the parms BEFORE constructing the context.
    // PhantomGaloisTool is built when context is initialized.
    LinearTransformDiagonals host_diags = build_c2s_diagonals(static_cast<int>(log_n), {5, 5, 5});
    const int num_slots = static_cast<int>(N >> 1);
    std::vector<int> steps = c2s_required_rotation_steps(host_diags, num_slots);
    auto galois_elts = phantom::util::get_elts_from_steps(steps, N);
    galois_elts.push_back(static_cast<uint32_t>(2 * N - 1)); // conjugation
    parms.set_galois_elts(galois_elts);

    PhantomContext context(parms);
    PhantomSecretKey sk(context);
    PhantomCKKSEncoder encoder(context);

    const std::size_t HW_TEST = 128;
    BootstrapKey bk = create_bootstrap_key(context, encoder, sk, HW_TEST);

    // Build a small complex slot vector.
    const size_t slot_count = encoder.slot_count();
    std::mt19937_64 rng(0xC2C0DEUL);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<C64> z(slot_count);
    std::vector<cuDoubleComplex> z_cu(slot_count);
    for (size_t i = 0; i < slot_count; ++i) {
        const double re = dist(rng);
        const double im = dist(rng);
        z[i] = C64(re, im);
        z_cu[i] = make_cuDoubleComplex(re, im);
    }

    // Encode at the C2S input level (= chain index of layer-0 plaintexts).
    const size_t c2s_in_chain =
        bk.c2s.layers[0].diagonals.begin()->second.target_chain_index;
    const auto &lvl_data = context.get_context_data(c2s_in_chain);
    const double encode_scale = static_cast<double>(
        lvl_data.parms().coeff_modulus().back().value());

    PhantomPlaintext pt;
    encoder.encode(context, z_cu, encode_scale, pt, c2s_in_chain);
    PhantomCiphertext ct;
    sk.encrypt_symmetric(context, pt, ct);

    // Sanity: ct should already be at c2s_in_chain after symmetric encrypt.
    if (ct.chain_index() != c2s_in_chain) {
        // Encrypt drops to the encode level; if behaviour ever differs, mod_switch.
        mod_switch_to_inplace(context, ct, c2s_in_chain);
    }

    apply_c2s_inplace(context, ct, bk);

    PhantomPlaintext dec_pt;
    sk.decrypt(context, ct, dec_pt);
    std::vector<cuDoubleComplex> decoded_cu;
    encoder.decode(context, dec_pt, decoded_cu);
    std::vector<C64> decoded(slot_count);
    for (size_t i = 0; i < slot_count; ++i) {
        decoded[i] = C64(decoded_cu[i].x, decoded_cu[i].y);
    }

    // Host oracle: apply the same C2S operation on z directly.
    const double last_layer_norm = static_cast<double>(num_slots);
    std::vector<C64> reference = apply_c2s_host(host_diags, z, last_layer_norm);

    double max_abs_err = 0.0;
    double sum_abs_err = 0.0;
    double max_abs_ref = 0.0;
    for (size_t i = 0; i < slot_count; ++i) {
        const double err = std::abs(decoded[i] - reference[i]);
        max_abs_err = std::max(max_abs_err, err);
        sum_abs_err += err;
        max_abs_ref = std::max(max_abs_ref, std::abs(reference[i]));
    }
    const double avg_err = sum_abs_err / static_cast<double>(slot_count);

    std::printf("c2s_round_trip_via_inverse: logN=%zu, stages=[5,5,5], hw=%zu\n",
                log_n, HW_TEST);
    for (size_t i = 0; i < 4; ++i) {
        std::printf("  slot[%zu] dec=(%.4e,%.4e)  ref=(%.4e,%.4e)\n",
                    i, decoded[i].real(), decoded[i].imag(),
                    reference[i].real(), reference[i].imag());
    }
    std::printf("  avg |err| = %.3e\n", avg_err);
    std::printf("  max |err| = %.3e\n", max_abs_err);
    std::printf("  max |ref| = %.3e\n", max_abs_ref);

    constexpr double tol = 1e-3;
    if (max_abs_err > tol) {
        std::fprintf(stderr,
            "FAIL: c2s max abs error %.3e exceeds tolerance %.3e\n",
            max_abs_err, tol);
        return 1;
    }
    // Structural sanity: reference must not be identically zero (catches a
    // bug where diagonals collapse to all-zero).
    if (max_abs_ref < 1e-6) {
        std::fprintf(stderr,
            "FAIL: c2s reference vector is suspiciously small (%.3e)\n",
            max_abs_ref);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

// Phase 3: C2S followed by S2C should round-trip the slot vector.
//
// With C2S `last_layer_norm = num_slots` and S2C `last_layer_norm = 1.0` (the
// lapis convention), the per-iteration normalizations are 1/num_slots and 1
// respectively, so the host composition `S2C(C2S(z)) = DFT(IDFT(z)) = z`. We
// run that pipeline on the GPU and check the decoded output against the
// original input directly. (We don't traverse EvalMod yet — Phase 4 — so we
// `mod_switch_to` from the post-C2S level straight to S2C's input level.)
//
// Tolerance: this is a CKKS noise check across 6 rescales (3 C2S + 3 S2C) at
// 58-bit primes plus key-switching noise. ~1e-3 is comfortable; we tighten if
// observed precision is much better.
static int run_c2s_then_s2c_round_trip() {
    using C64 = std::complex<double>;
    using namespace phantom::util;

    EncryptionParameters parms(scheme_type::ckks);
    constexpr size_t log_n = 16;
    const size_t N = 1ULL << log_n;
    parms.set_poly_modulus_degree(N);

    // Reduced chain (mirrors Phase 2's special_modulus_size=4 trick to keep
    // the Galois KSK bundle in GPU memory). 8 ordinary primes are enough:
    // C2S consumes chain 0..2, S2C consumes chain 3..5, leaving chain 6..7
    // for decryption headroom.
    parms.set_special_modulus_size(4);
    std::vector<int> bits;
    bits.push_back(58);                              // q_0
    for (int i = 0; i < 7; ++i) bits.push_back(58);  // 3 C2S + 3 S2C + 1 buffer
    for (int i = 0; i < 4; ++i) bits.push_back(58);  // 4 special
    parms.set_coeff_modulus(CoeffModulus::Create(N, bits));

    // Union of C2S and S2C rotation steps + conjugation Galois elt.
    LinearTransformDiagonals c2s_h = build_c2s_diagonals(static_cast<int>(log_n), {5, 5, 5});
    LinearTransformDiagonals s2c_h = build_s2c_diagonals(static_cast<int>(log_n), {5, 5, 5});
    const int num_slots = static_cast<int>(N >> 1);

    std::vector<int> c2s_steps = c2s_required_rotation_steps(c2s_h, num_slots);
    std::vector<int> s2c_steps = c2s_required_rotation_steps(s2c_h, num_slots);
    std::vector<int> all_steps = c2s_steps;
    all_steps.insert(all_steps.end(), s2c_steps.begin(), s2c_steps.end());
    std::sort(all_steps.begin(), all_steps.end());
    all_steps.erase(std::unique(all_steps.begin(), all_steps.end()), all_steps.end());

    auto galois_elts = phantom::util::get_elts_from_steps(all_steps, N);
    galois_elts.push_back(static_cast<uint32_t>(2 * N - 1)); // conjugation
    parms.set_galois_elts(galois_elts);

    PhantomContext context(parms);
    PhantomSecretKey sk(context);
    PhantomCKKSEncoder encoder(context);

    const std::size_t HW_TEST = 128;
    BootstrapKey bk = create_bootstrap_key(context, encoder, sk, HW_TEST);

    // Sanity-check derived chain indices.
    const size_t c2s_in_chain =
        bk.c2s.layers[0].diagonals.begin()->second.target_chain_index;
    const size_t s2c_in_chain =
        bk.s2c.layers[0].diagonals.begin()->second.target_chain_index;
    if (c2s_in_chain != context.get_first_index()) {
        std::fprintf(stderr,
            "FAIL: C2S layer 0 chain %zu != first_index %zu\n",
            c2s_in_chain, context.get_first_index());
        return 1;
    }
    if (s2c_in_chain != context.get_first_index() + bk.c2s.layers.size()) {
        std::fprintf(stderr,
            "FAIL: S2C layer 0 chain %zu != first_index + num_c2s_layers %zu\n",
            s2c_in_chain,
            context.get_first_index() + bk.c2s.layers.size());
        return 1;
    }

    // Encode + encrypt at C2S input level.
    const auto &lvl_data = context.get_context_data(c2s_in_chain);
    const double encode_scale = static_cast<double>(
        lvl_data.parms().coeff_modulus().back().value());

    std::mt19937_64 rng(0xC2C5UL);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    const size_t slot_count = encoder.slot_count();
    std::vector<C64> z(slot_count);
    std::vector<cuDoubleComplex> z_cu(slot_count);
    for (size_t i = 0; i < slot_count; ++i) {
        const double re = dist(rng);
        const double im = dist(rng);
        z[i] = C64(re, im);
        z_cu[i] = make_cuDoubleComplex(re, im);
    }

    PhantomPlaintext pt;
    encoder.encode(context, z_cu, encode_scale, pt, c2s_in_chain);
    PhantomCiphertext ct;
    sk.encrypt_symmetric(context, pt, ct);
    if (ct.chain_index() != c2s_in_chain) {
        mod_switch_to_inplace(context, ct, c2s_in_chain);
    }

    // Optional sanity: host oracle round-trip should be ≈ identity.
    {
        std::vector<C64> mid_host = apply_c2s_host(c2s_h, z, /*last_layer_norm=*/static_cast<double>(num_slots));
        std::vector<C64> back_host = apply_s2c_host(s2c_h, mid_host, /*last_layer_norm=*/1.0);
        double max_host_err = 0.0;
        for (size_t i = 0; i < slot_count; ++i) {
            max_host_err = std::max(max_host_err, std::abs(back_host[i] - z[i]));
        }
        std::printf("c2s_then_s2c host oracle round-trip max |err| = %.3e\n", max_host_err);
        if (max_host_err > 1e-9) {
            std::fprintf(stderr,
                "FAIL: host oracle S2C∘C2S not identity (%.3e); diagonal math is off\n",
                max_host_err);
            return 1;
        }
    }

    // GPU: C2S, then mod_switch (no-op if s2c_in_chain == post-C2S chain), then S2C.
    apply_c2s_inplace(context, ct, bk);
    if (ct.chain_index() != s2c_in_chain) {
        mod_switch_to_inplace(context, ct, s2c_in_chain);
    }
    apply_s2c_inplace(context, ct, bk);

    PhantomPlaintext dec_pt;
    sk.decrypt(context, ct, dec_pt);
    std::vector<cuDoubleComplex> decoded_cu;
    encoder.decode(context, dec_pt, decoded_cu);
    std::vector<C64> decoded(slot_count);
    for (size_t i = 0; i < slot_count; ++i) {
        decoded[i] = C64(decoded_cu[i].x, decoded_cu[i].y);
    }

    double max_abs_err = 0.0, sum_abs_err = 0.0, max_abs_in = 0.0;
    for (size_t i = 0; i < slot_count; ++i) {
        const double err = std::abs(decoded[i] - z[i]);
        max_abs_err = std::max(max_abs_err, err);
        sum_abs_err += err;
        max_abs_in = std::max(max_abs_in, std::abs(z[i]));
    }
    const double avg_err = sum_abs_err / static_cast<double>(slot_count);

    std::printf("c2s_then_s2c_round_trip: logN=%zu, stages=[5,5,5], hw=%zu\n",
                log_n, HW_TEST);
    for (size_t i = 0; i < 4; ++i) {
        std::printf("  slot[%zu] dec=(%.4e,%.4e)  in=(%.4e,%.4e)\n",
                    i, decoded[i].real(), decoded[i].imag(),
                    z[i].real(), z[i].imag());
    }
    std::printf("  avg |err| = %.3e\n", avg_err);
    std::printf("  max |err| = %.3e\n", max_abs_err);
    std::printf("  max |in|  = %.3e\n", max_abs_in);

    constexpr double tol = 1e-3;
    if (max_abs_err > tol) {
        std::fprintf(stderr,
            "FAIL: c2s_then_s2c max abs error %.3e exceeds tolerance %.3e\n",
            max_abs_err, tol);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

// Phase 4: full end-to-end bootstrap round-trip.
//
// Chain layout (bottom→top in bits[] order, matching the lapis heterogeneous
// "4-section" spec for K=28 R=3 EvalMod = 9 levels):
//
//   bits[] = [msg(58) | scale(40)×4 | S2C(58)×3 | ER(58)×9 | C2S(29)×3 | special(58)×3]
//
// With set_special_modulus_size(3) and first_idx=1 this gives chain indices:
//   C2S:     1,  2,  3   (29-bit, consumed first by C2S rescales)
//   ER:      4 ..12      (58-bit, consumed by EvalMod K=28 R=3, 9 levels)
//   S2C:    13, 14, 15   (58-bit, consumed by S2C rescales)
//   scale:  16 ..19      (40-bit, user-space primes)
//   msg:    20           (= total_parm_size()-1, single 58-bit q_0)
// size_Q = 1+4+3+9+3 = 20; 20%4 = 0 (dnum=5) with num_special=4... but we use
// num_special=3: 20%3 != 0. Use num_special=4: 20%4=0 ✓, or adjust.
// Actually 1+4+3+9+3=20, num_special=4: size_Q=20, 20%4=0 ✓.
// Alternatively keep num_special=3 and use 9+3+3+4+1=20: 20%3=2 ✗.
// Use num_special=4: size_Q=20, dnum=5. ✓
//
// The test encodes at scale=2^40 (in the scale segment), depletes ct to the
// bottom (single prime q_msg ≈ 2^58), then calls bootstrap() and checks that
// the decoded output matches the original within 5e-6.
static int run_bootstrap_round_trip() {
    using C64 = std::complex<double>;
    using namespace phantom::util;

    EncryptionParameters parms(scheme_type::ckks);
    constexpr size_t log_n = 16;
    const size_t N = 1ULL << log_n;
    parms.set_poly_modulus_degree(N);
    // size_Q = 1+4+3+9+3 = 20; num_special=4: 20%4=0 (dnum=5). ✓
    parms.set_special_modulus_size(4);

    // Build the heterogeneous chain (bottom→top, special last).
    std::vector<int> bits;
    bits.push_back(58);                              // msg (q_0, bottom)
    for (int i = 0; i < 4; ++i) bits.push_back(40); // scale segment
    for (int i = 0; i < 3; ++i) bits.push_back(58); // S2C (58-bit)
    for (int i = 0; i < 9; ++i) bits.push_back(58); // ER / EvalMod (K=28 R=3: 6 sine + 3 DA = 9 levels)
    for (int i = 0; i < 3; ++i) bits.push_back(29); // C2S (29-bit, consumed first)
    for (int i = 0; i < 4; ++i) bits.push_back(58); // special
    parms.set_coeff_modulus(CoeffModulus::Create(N, bits));

    // Compute the union of C2S and S2C rotation steps + conjugation Galois elt.
    LinearTransformDiagonals c2s_h = build_c2s_diagonals(static_cast<int>(log_n), {5, 5, 5});
    LinearTransformDiagonals s2c_h = build_s2c_diagonals(static_cast<int>(log_n), {5, 5, 5});
    const int num_slots = static_cast<int>(N >> 1);

    std::vector<int> c2s_steps = c2s_required_rotation_steps(c2s_h, num_slots);
    std::vector<int> s2c_steps = c2s_required_rotation_steps(s2c_h, num_slots);
    std::vector<int> all_steps = c2s_steps;
    all_steps.insert(all_steps.end(), s2c_steps.begin(), s2c_steps.end());
    std::sort(all_steps.begin(), all_steps.end());
    all_steps.erase(std::unique(all_steps.begin(), all_steps.end()), all_steps.end());

    auto galois_elts = phantom::util::get_elts_from_steps(all_steps, N);
    galois_elts.push_back(static_cast<uint32_t>(2 * N - 1)); // conjugation
    parms.set_galois_elts(galois_elts);

    PhantomContext context(parms);
    PhantomSecretKey sk(context);
    PhantomCKKSEncoder encoder(context);

    const std::size_t HW_TEST = 128;
    const double user_scale = std::pow(2.0, 40);
    // eval_mod_levels=9: K=28 R=3 EvalMod consumes 9 levels (6 sine + 3 DA).
    // S2C plaintexts are pre-encoded 9 levels below C2S end.
    // user_scale: the LAST S2C layer is encoded at this scale to bake the
    // q_msg → user_scale scale-down into the linear transform.
    BootstrapKey bk = create_bootstrap_key(context, encoder, sk, HW_TEST,
                                           /*eval_mod_levels=*/9,
                                           /*user_scale=*/user_scale);

    // Encode at user scale=2^40 (in the scale segment, just below S2C).
    const size_t slot_count = encoder.slot_count();

    std::mt19937_64 rng(0xB007B007UL);
    // CKKS native message bound: |m| < 0.5. Larger application ranges
    // require user-level preprocessing (scale-down before encrypt,
    // scale-back-up after decrypt).
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<double> input_re(slot_count);
    for (size_t i = 0; i < slot_count; ++i) input_re[i] = dist(rng);

    // Encode as real (imaginary parts zero).
    std::vector<cuDoubleComplex> z_cu(slot_count);
    for (size_t i = 0; i < slot_count; ++i) {
        z_cu[i] = make_cuDoubleComplex(input_re[i], 0.0);
    }

    PhantomPlaintext pt;
    encoder.encode(context, z_cu, user_scale, pt);
    PhantomCiphertext ct;
    sk.encrypt_symmetric(context, pt, ct);

    // Deplete ct to ONE level above the bottom (2 primes: [q_msg, q_scale]).
    // scale_up_for_bootstrap needs this extra level to do multiply + rescale.
    // bottom_index = total_parm_size()-1 (single prime); one above = bottom-1.
    const size_t bottom_index = context.total_parm_size() - 1;
    const size_t pre_boot_index = bottom_index - 1; // one level above bottom
    mod_switch_to_inplace(context, ct, pre_boot_index);

    std::printf("bootstrap_round_trip: logN=%zu, user_scale=2^40, hw=%zu\n",
                log_n, HW_TEST);
    std::printf("  ct before bootstrap: chain_index=%zu, scale=2^%.1f\n",
                ct.chain_index(), std::log2(ct.scale()));

    // Run the full bootstrap pipeline.
    PhantomCiphertext out = bootstrap(context, encoder, ct, bk, user_scale);

    std::printf("  ct after  bootstrap: chain_index=%zu, scale=2^%.1f\n",
                out.chain_index(), std::log2(out.scale()));

    PhantomPlaintext dec_pt;
    sk.decrypt(context, out, dec_pt);
    std::vector<cuDoubleComplex> decoded_cu;
    encoder.decode(context, dec_pt, decoded_cu);

    std::vector<std::pair<double, size_t>> err_slot(slot_count);
    double max_abs_err = 0.0, sum_abs_err = 0.0;
    for (size_t i = 0; i < slot_count; ++i) {
        const double err = std::abs(decoded_cu[i].x - input_re[i]);
        err_slot[i] = {err, i};
        max_abs_err = std::max(max_abs_err, err);
        sum_abs_err += err;
    }
    const double avg_err = sum_abs_err / static_cast<double>(slot_count);

    // Sort descending by error magnitude.
    std::sort(err_slot.begin(), err_slot.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    std::printf("  top 10 worst slots (slot, input, decoded, err, |input|):\n");
    for (int k = 0; k < 10; ++k) {
        const size_t i = err_slot[k].second;
        std::printf("    [%zu]  in=%+.6f  dec=%+.6f  err=%.3e  |in|=%.4f\n",
                    i, input_re[i], decoded_cu[i].x, err_slot[k].first, std::abs(input_re[i]));
    }
    // Histogram by error magnitude (log-bins).
    int bins[8] = {0};  // 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, < 1e-8
    for (auto &p : err_slot) {
        int b = (p.first >= 1e-2) ? 0
              : (p.first >= 1e-3) ? 1
              : (p.first >= 1e-4) ? 2
              : (p.first >= 1e-5) ? 3
              : (p.first >= 1e-6) ? 4
              : (p.first >= 1e-7) ? 5
              : (p.first >= 1e-8) ? 6
              :                     7;
        bins[b]++;
    }
    std::printf("  error histogram: ≥1e-2:%d, 1e-3:%d, 1e-4:%d, 1e-5:%d, 1e-6:%d, 1e-7:%d, 1e-8:%d, <1e-8:%d\n",
                bins[0], bins[1], bins[2], bins[3], bins[4], bins[5], bins[6], bins[7]);
    // Median + 99th percentile for distribution shape.
    const double med = err_slot[slot_count / 2].first;
    const double p99 = err_slot[slot_count / 100].first;
    std::printf("  avg |err| = %.3e   median = %.3e   p99 = %.3e   max = %.3e\n",
                avg_err, med, p99, max_abs_err);

    constexpr double tol = 5e-6;
    if (max_abs_err > tol) {
        std::fprintf(stderr,
            "FAIL: bootstrap max abs error %.3e exceeds tolerance %.3e\n",
            max_abs_err, tol);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

int main() {
    int rc = 0;
    rc |= run_ks_round_trip_only();
    rc |= run_round_trip();
    rc |= run_c2s_round_trip_via_inverse();
    rc |= run_c2s_then_s2c_round_trip();
    rc |= run_bootstrap_round_trip();
    return rc;
}
