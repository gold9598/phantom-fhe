#pragma once

// CKKS bootstrap support: encapsulation key (sparse↔dense KSK pair) and
// modulus-extension primitive `mod_raise_inplace`.
//
// Phase 1: SmallBootstrapKey + mod_raise_inplace.
// Phase 2: BootstrapKey + LinearTransformParams + apply_c2s_inplace
//          (this file's recent additions). C2S maps the slot vector embedded
//          in the coefficient form (post-mod-raise) back into the slot
//          domain so EvalMod can act element-wise.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "ciphertext.h"
#include "context.cuh"
#include "cuda_wrapper.cuh"
#include "plaintext.h"
#include "secretkey.h"

class PhantomCKKSEncoder;

namespace phantom {

    // Encapsulation KSK pair used by `mod_raise_inplace`.
    //
    // `ksk_to_sparse` switches a ciphertext under the user-facing dense secret
    // into one under a temporary sparse secret (Hamming-weight-bounded);
    // `ksk_to_dense` switches it back. The sparse secret itself is
    // generated transiently inside `create_small_bootstrap_key` and then
    // discarded — only the two KSKs are retained.
    struct SmallBootstrapKey {
        PhantomRelinKey ksk_to_sparse; // KSK(dense → sparse)
        PhantomRelinKey ksk_to_dense;  // KSK(sparse → dense)
    };

    // Build the encapsulation KSK pair for `dense_sk`. The sparse secret has
    // exactly `sparse_hamming_weight` non-zero ternary coefficients (typical:
    // 128). The sparse secret is local to this call and is destroyed before
    // returning.
    [[nodiscard]] SmallBootstrapKey
    create_small_bootstrap_key(const PhantomContext &ctx,
                               const PhantomSecretKey &dense_sk,
                               std::size_t sparse_hamming_weight = 128);

    // Encapsulated mod-raise.
    //
    // Caller must have brought `ct` down to the bottom of the modulus chain
    // (single remaining prime q_msg). After this call, `ct` lives at the top
    // of the chain (chain_index = first index, all Q primes), with each
    // coefficient re-encoded into every higher tower from its centered
    // representative in [-q_msg/2, q_msg/2). The dense↔sparse encapsulation
    // is applied transparently.
    void mod_raise_inplace(const PhantomContext &ctx,
                           PhantomCiphertext &ct,
                           const SmallBootstrapKey &bk);

    // Compact pre-encoded plaintext for bootstrap diagonals. Stores the
    // polynomial in coefficient (INTT) form as a single tower of signed-centered
    // int64_t values on device, instead of as a full-RNS NTT-form
    // PhantomPlaintext. Storage: N * 8 bytes (e.g. 512 KB at logN=16) versus
    // ~14 MB for the full PhantomPlaintext. Expanded on demand at multiply
    // time via `expand_light_plaintext`.
    struct LightPlaintext {
        // Polynomial coefficients in [-q_0/2, q_0/2). Length N. Lives on device.
        phantom::util::cuda_auto_ptr<std::int64_t> coeffs_int64;
        // Chain index that this diagonal will be expanded to and multiplied at.
        std::size_t target_chain_index = 0;
        // CKKS scale Δ used at encode time; copied into the expanded plaintext.
        double scale = 1.0;
    };

    // Diagonal layout for one BSGS layer of C2S (or S2C — same structure).
    // `diagonals` maps signed diagonal index -> pre-encoded LightPlaintext at the
    // chain_index that layer evaluates at. Stored once at key generation; the
    // BSGS evaluator expands the whole layer to PhantomPlaintexts on demand.
    struct C2SLayerDiagonals {
        std::unordered_map<int, LightPlaintext> diagonals;
        int n1;             // baby-step count = 2^(stages-1)
        int rotation_unit;  // 2^(sum of stages in subsequent layers)
    };

    // Forward / inverse linear-transform parameters and pre-encoded diagonals.
    // Built once at key gen; immutable after.
    struct LinearTransformParams {
        std::vector<int> stages_per_layer;     // e.g. [5,5,5] for logN=16
        std::vector<C2SLayerDiagonals> layers; // one entry per layer
        int n2 = 4;                            // giant-step count (placeholder)
    };

    // Per-layer KSK slot: either OWNS a generated KSK or DELEGATES to a
    // canonical owner elsewhere in the BootstrapKey. The fallback pointer is
    // a non-owning view into another slot's `owned` field; the canonical
    // owner's lifetime is tied to the same BootstrapKey, so the raw pointer
    // is safe.
    //
    // Canonical-owner principle: for every Galois element used anywhere in
    // the engine, exactly ONE physical KSK is generated — at the shallowest
    // chain at which it is needed. Deeper-chain uses register a fallback to
    // that single canonical KSK. Phantom's keyswitching kernel drops unused
    // primes at runtime, so a shallower-chain KSK is a superset that serves
    // all deeper uses (matching the relin-key pattern: relin keys live at
    // chain 0 and are consumed at every chain).
    struct PerLayerKSKSlot {
        PhantomRelinKey owned;              // valid iff fallback == nullptr
        const PhantomRelinKey* fallback = nullptr; // non-null => use this instead

        // Returns the effective KSK: the fallback if registered, else owned.
        // The `owned` member is valid even when default-constructed; callers
        // must only invoke `get()` after the canonical builder has run.
        const PhantomRelinKey& get() const {
            return fallback ? *fallback : owned;
        }
    };

    // BootstrapKey: full key bundle for the four-stage CKKS bootstrap pipeline
    // (ModRaise → C2S → EvalMod → S2C). C2S is filled in this phase; S2C and
    // EvalMod live in later phases.
    //
    // Galois key storage is split across three buckets to enable per-stage
    // level-aware truncation of bootstrap KSKs (see `mod_drop_to_inplace`):
    //
    //   * `user_galois_keys`: full-Q KSKs for arbitrary user rotations
    //     (called from any user level via `CKKSEngine::rotate_inplace`).
    //   * `c2s_galois_keys[layer][step]`: KSK slot for the rotation `step`
    //     used by C2S layer `layer`. The slot either owns a KSK truncated
    //     to the chain_index where that layer evaluates, or holds a
    //     fallback pointer to a canonical owner elsewhere in the bundle
    //     (see `PerLayerKSKSlot` for the canonical-owner principle).
    //   * `s2c_galois_keys[layer][step]`: same idea for S2C layers, which
    //     evaluate at deeper chain indices and therefore see much larger
    //     dnum savings (size_Ql shrinks → beta_k drops more entries).
    //     A step that also appears in C2S delegates to the C2S owner
    //     (shallower chain, superset of primes).
    //
    // The conjugation KSK (Galois elt 2N-1, used between C2S and EvalMod)
    // is kept full-Q under `user_galois_keys` because it's invoked at
    // C2S's output level which has no dnum savings to capture.
    struct BootstrapKey {
        SmallBootstrapKey small;
        PhantomGaloisKey  user_galois_keys;   // full-Q KSKs (user rotations + conjugation)
        std::vector<std::map<int, PerLayerKSKSlot>> c2s_galois_keys; // per-layer step → owned-or-fallback slot
        std::vector<std::map<int, PerLayerKSKSlot>> s2c_galois_keys; // per-layer step → owned-or-fallback slot
        PhantomRelinKey   relin_key;          // for EvalMod (Phase 4)
        LinearTransformParams c2s;
        LinearTransformParams s2c;            // empty in Phase 2
    };

    // ===== Host-side diagonal computation (no encoding, no GPU) =====
    //
    // `LinearTransformDiagonals` holds the raw complex diagonals before
    // encoding into plaintexts; this is what `build_c2s_diagonals` returns.
    // It exists separately from `LinearTransformParams` because the latter
    // owns GPU plaintexts (only constructible once a context exists).
    struct LinearTransformLayerHost {
        std::unordered_map<int, std::vector<std::complex<double>>> diagonals;
        int n1 = 0;
        int rotation_unit = 0;
    };

    struct LinearTransformDiagonals {
        std::vector<int> stages_per_layer;
        std::vector<LinearTransformLayerHost> layers;
        int n2 = 4;
    };

    // Pure-math: build C2S diagonals (DIF butterfly factorization of the
    // inverse-DFT that maps slot vectors to coefficient form).
    //   sum(stages_per_layer) must equal log2(N/2) = log_n - 1.
    [[nodiscard]] LinearTransformDiagonals
    build_c2s_diagonals(int log_n, std::vector<int> stages_per_layer);

    // Pure-math: build S2C diagonals (the inverse linear transform of C2S).
    // Derived from `build_c2s_diagonals(log_n, stages_per_layer)` by reversing
    // the layer order and Hermitian-conjugating each diagonal:
    //   s2c.layers[k] = (c2s.layers[L-1-k])^H
    //   diagH[-d][j] = conj(diag[d][(j + d*R) mod n])  // i.e. roll(vals, +d*R)
    [[nodiscard]] LinearTransformDiagonals
    build_s2c_diagonals(int log_n, std::vector<int> stages_per_layer);

    // Compute the (deduplicated, sorted) set of cyclic rotation steps
    // required to evaluate the C2S transform with the naive
    // rotate-multiply-sum algorithm. Steps are reduced into the slot range
    // (-num_slots/2, num_slots/2]. The conjugation step (= 0 in lapis's
    // shifted convention; here, just `2N-1` Galois elt) is not included —
    // callers should add it manually if needed.
    [[nodiscard]] std::vector<int>
    c2s_required_rotation_steps(const LinearTransformDiagonals &diags,
                                int num_slots);

    // Build the full BootstrapKey:
    //   1. SmallBootstrapKey (ModRaise encapsulation)
    //   2. Relin key (for EvalMod)
    //   3. PhantomGaloisKey — for the rotation steps required by C2S+S2C,
    //      plus the conjugation step (Galois elt 2N-1).
    //   4. C2S diagonals — pre-encoded at chain indices
    //         [first_idx, first_idx + num_c2s_layers).
    //   5. S2C diagonals — pre-encoded at chain indices
    //         [first_idx + num_c2s_layers + eval_mod_levels,
    //          first_idx + num_c2s_layers + eval_mod_levels + num_s2c_layers).
    //
    // `eval_mod_levels` is the number of chain levels consumed by EvalMod
    // between C2S and S2C. For K=16 R=3 that's 9 (default). Pass 0 if you
    // intend to chain C2S directly into S2C (e.g. Phase 3 round-trip test).
    //
    // `user_scale`: when > 0, the LAST S2C layer is encoded at this scale
    // (instead of the chain prime at that level). This bakes the scale-down
    // from q_msg back to user_scale into the linear transform — after the
    // last S2C multiply+rescale, ct.scale ≈ user_scale. Pass 0 to encode
    // every S2C layer at its chain prime (Phase 2/3 uniform-58-bit usage).
    //
    // `split_scale_down`: when true, encode all S2C layers at their chain
    // primes (no user_scale baking, regardless of the `user_scale` arg) so
    // that bootstrap()'s post-S2C scale matches saved's q_msg-aligned scale.
    // The q_msg → user_scale step is then applied as a single integer
    // multiply + rescale AFTER the saved-out subtraction, on the small
    // residual `m` (rather than the large mod-raised `m + K·I`). Costs 1
    // extra user level but recovers ~5 bits of bootstrap precision (the
    // scale-down rounding no longer compounds with K·I noise).
    [[nodiscard]] BootstrapKey
    create_bootstrap_key(const PhantomContext &ctx,
                         PhantomCKKSEncoder &encoder,
                         const PhantomSecretKey &dense_sk,
                         std::size_t sparse_hamming_weight = 128,
                         std::size_t eval_mod_levels = 0,
                         double user_scale = 0.0,
                         bool split_scale_down = false);

    // Apply the pre-encoded C2S linear transform in place.
    //
    // Naive rotate-multiply-sum across each layer's diagonals. Before calling,
    // `ct` must satisfy `ct.chain_index() == bk.c2s.layers[0].diagonals.begin()->second.target_chain_index`
    // (i.e. live at the input level of layer 0). After return, `ct` has
    // consumed `num_layers` levels (= 3 for stages={5,5,5}).
    void apply_c2s_inplace(const PhantomContext &ctx,
                           PhantomCiphertext &ct,
                           const BootstrapKey &bk);

    // Single-stage C2S variant matching the_lib's `coeff_to_slot_complex_for_
    // 17_levels` (src/ckks/engine/bootstrap.cpp:714): applies all layer
    // butterflies WITHOUT per-layer rescale (OVER_SCALED accumulation), then
    // a single `rescale_after_multiply` consumes one chain prime — the 42-bit
    // C2S prime in the BootstrapTo17Levels chain.
    //
    // The bootstrap key's C2S diagonals must be pre-encoded with all layers
    // sharing the same `target_chain_index` (see Phase-3 build path in
    // create_bootstrap_key when use_bootstrap_to_17_levels is set).
    void apply_c2s_inplace_single_stage(const PhantomContext &ctx,
                                        PhantomCiphertext &ct,
                                        const BootstrapKey &bk);

    // Apply the pre-encoded S2C linear transform in place.
    //
    // Same naive rotate-multiply-sum-rescale evaluator as C2S, but iterates
    // over `bk.s2c` (which is empty before Phase 3). Before calling, `ct` must
    // live at the chain index of `bk.s2c.layers[0]`'s plaintexts. After
    // return, `ct` has consumed `num_layers` levels.
    void apply_s2c_inplace(const PhantomContext &ctx,
                           PhantomCiphertext &ct,
                           const BootstrapKey &bk);

    // Host-side reference: apply the same C2S transform (diagonals +
    // per-layer normalization) to a host complex slot vector. Used by the
    // test as an oracle. `last_layer_norm` must match what
    // `create_bootstrap_key` used (default: num_slots).
    [[nodiscard]] std::vector<std::complex<double>>
    apply_c2s_host(const LinearTransformDiagonals &diags,
                   const std::vector<std::complex<double>> &slot_input,
                   double last_layer_norm);

    // Host-side reference for S2C (same math as C2S but iterates over the
    // S2C diagonals). For the lapis-default norms (C2S last_layer_norm =
    // num_slots, S2C last_layer_norm = 1.0), the round-trip
    // `apply_s2c_host(apply_c2s_host(z, c2s, num_slots), s2c, 1.0)` is the
    // identity on slot vectors.
    [[nodiscard]] std::vector<std::complex<double>>
    apply_s2c_host(const LinearTransformDiagonals &diags,
                   const std::vector<std::complex<double>> &slot_input,
                   double last_layer_norm);

    // Phase 4: full bootstrap pipeline.
    //
    // Pipeline (all internal):
    //   scale_up → mod_raise → C2S → eval_round (= K·ct − EvalMod(ct))
    //                        → S2C → final scale-down to user_scale
    //
    // Caller's `ct` must be at the bottom of the chain (single remaining
    // prime q_msg), holding a message originally encoded at `user_scale`
    // (e.g. 2^40). On return, the bootstrapped ciphertext encodes the same
    // plaintext at the same `user_scale`, fresh in the chain.
    //
    // `split_scale_down` MUST match the value passed to create_bootstrap_key
    // (the S2C diagonals are pre-encoded for one path or the other). When
    // true, the post-S2C ct lives at the s2c_in chain prime; an extra
    // multiply+rescale brings it down to user_scale at one chain index
    // deeper than the no-split path.
    [[nodiscard]] PhantomCiphertext
    bootstrap(const PhantomContext &ctx,
              PhantomCKKSEncoder &encoder,
              const PhantomCiphertext &ct,
              const BootstrapKey &bk,
              double user_scale,
              bool split_scale_down = false);

} // namespace phantom
