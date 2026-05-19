#pragma once

// phantom::CKKSEngine — user-facing CKKS facade with bootstrap support.
//
// Phase 5 of the CKKS bootstrap port. Owns:
//   * a heterogeneous-chain PhantomContext matching the lapis 4-section spec
//     used by `bootstrap_round_trip` (msg | scale x N | S2C x 3 | ER x 9 |
//     C2S x 3 | special),
//   * a sparse-hw=128 dense secret key,
//   * relin / Galois keys, pre-encoded C2S+S2C diagonals (BootstrapKey).
//
// The user never touches bootstrap primes. They see only a `user_level`
// counter: 0 == freshest (top of the user-scale segment), increments by 1
// per `rescale_inplace`, max == num_scale_levels - 1 (pre_boot level, two
// primes: q_scale_last + q_msg). `bootstrap_inplace` resets user_level to 0.

#include <complex>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "phantom.h"
#include "bootstrap.h"

namespace phantom {

    // Configuration knobs the user CAN set. Everything else is fixed by the
    // lapis-port chain layout.
    struct CKKSEngineConfig {
        int    log_n           = 16;                 // pinned for this port
        double user_scale      = std::pow(2.0, 40);  // 40-bit scale segment
        // num_scale_levels=14 + num_special_primes=6 → mul depth 13. size_Q
        // = 1 + num_scale_levels + 3(S2C) + 9(ER) + 3(C2S) = 16 + num_scale_levels;
        // must stay divisible by num_special_primes. 16+14=30, 30%6=0. ✓
        // (K=28 R=3 uses 9 ER primes, same as K=16 R=3.)
        int    num_scale_levels = 14;
        int    sparse_hw       = 128;                // bootstrap encapsulation hw
        int    num_special_primes = 6;               // for KSK hybrid
        // If true, the engine generates Galois keys for a small fixed set of
        // user rotations (±1, ±2, conjugation) in addition to the C2S/S2C
        // bootstrap rotations. When `user_rotation_steps` is non-empty, this
        // flag is ignored and only the explicit list is used (plus conjugation,
        // which is always added).
        bool   include_user_rotations = true;
        // Explicit user rotation step list. When non-empty, overrides the
        // default ±1/±2 set. Conjugation (galois_elt = 2N-1) is always added.
        // Steps used by C2S/S2C bootstrap are added automatically — callers
        // need only list the rotations their pipeline needs.
        std::vector<int> user_rotation_steps;
        // Optional: per-step target chain indices for user_rotation_steps.
        // Must be empty OR same length as user_rotation_steps.
        // If empty, all keys are generated at fresh user-scale (current default).
        // If provided, the i-th step's key is generated at the i-th target chain index.
        std::vector<std::size_t> user_rotation_target_chain_indices;
        // When true, the bootstrap performs the q_msg → user_scale scale-down
        // as a separate integer multiply + rescale AFTER the saved-out
        // subtraction (rather than baking it into the last S2C layer). This
        // costs 1 extra user level — bootstrap_inplace returns ct at
        // user_level == 1 instead of user_level == 0 — but recovers ~5 bits
        // of bootstrap precision (the scale-down rounding no longer
        // compounds with the K·I noise from mod-raise + S2C). Mirrors
        // the_lib's BootstrapTo14Levels compact-mode fix that splits
        // scale_down_ratio out of the bootstrap.
        bool   split_scale_down = false;

        // Phase 1 of the_lib BootstrapTo17Levels port: build per-level
        // ckks_scale[] (squared) + ckks_rescaled_scale[] (single) arrays
        // mirroring the_lib's `precomputed_.ckks_scale_` / `ckks_rescaled_scale_`
        // (src/ckks/scale.cpp:1420 `make_ckks_scales`). When true the engine
        // populates `ckks_scale()` / `ckks_rescaled_scale()` accessors so
        // downstream phases (encode-then-rescale, rescale-first butterfly)
        // can read the per-level scale precisely. Currently does NOT change
        // the chain layout or bootstrap algorithm — Phase 1 is pure metadata.
        bool   build_two_scale_arrays = false;

        // Phase 2: opt into the_lib's BootstrapTo17Levels chain layout
        // (CKKS_42_54_29_40_60_BOOTSTRAP — see the_lib src/ckks/config.cpp:168-178).
        // When true the engine builds the chain
        //     bits = [40×NSL | 29×1 | 54×12 | 42×1 | 60×NSP]
        //   instead of our default
        //     bits = [58 | 40×NSL | 58×3 | 58×9 | 29×3 | 58×NSP]
        //
        // The new layout uses the_lib's distinct prime pools — 40-bit small
        // (user-scale), 29-bit scale_down, 54-bit bootstrap segment (ER+S2C),
        // 42-bit coeff-to-slot, 60-bit large (special) — which is what enables
        // the_lib's 19-21 bit precision target.
        //
        // NOTE: Phase 2 changes the chain ONLY. The bootstrap algorithm still
        // assumes the legacy chain shape and will mis-execute on this layout
        // until Phase 3+ (single-stage C2S + encode-then-rescale + rescale-
        // first butterfly + final scale-down) is implemented. A test that
        // boots a ct on this chain WILL produce wrong values; the Phase 2
        // verification is engine init + scale-array correctness only.
        bool   use_bootstrap_to_17_levels = false;

        // EvalMod K=28 number of double-angle iterations.
        //   R=3 (default): 9 ER primes,  ~27-bit polynomial precision,
        //                  good on |input| <= ~0.5–0.7.
        //   R=4          : 10 ER primes, ~30-bit polynomial precision,
        //                  extends precision regime up to |input| ~ 1.0.
        // R=4 adds ONE extra prime to the chain: size_Q grows by 1, so
        // NSL+16+(R-3) must be divisible by num_special_primes.
        // Other values are not supported (only K=28 R=3 and K=28 R=4 are
        // implemented — see evalmod.cu evalmod_k28_r3 / evalmod_k28_r4).
        int    evalmod_r = 3;

        // PROBE-ONLY (do not commit): when true, build ER segment with 54-bit
        // primes instead of 58-bit. Used by probe_evalmod_calibration.py to
        // empirically determine whether evalmod_k28_r3's coefficients are
        // chain-prime-agnostic or calibrated for 58-bit.
        // Only honoured on the legacy (use_bootstrap_to_17_levels=false) path.
        bool   probe_er_54bit = false;
    };

    class CKKSEngine {
    public:
        explicit CKKSEngine(const CKKSEngineConfig &cfg = {});

        // ---- User-facing properties ----
        [[nodiscard]] std::size_t slot_count() const noexcept;
        [[nodiscard]] double user_scale() const noexcept { return cfg_.user_scale; }
        // max_user_level() = num_scale_levels - 1.
        // The deepest accessible level before bootstrap is required.
        [[nodiscard]] int    max_user_level() const noexcept { return cfg_.num_scale_levels - 1; }
        // Chain index of user_level=0 (the freshest user-accessible chain
        // immediately above the bootstrap output). Invariant to NSL; depends
        // on bootstrap pipeline (1+NSL+S2C+ER+C2S layout) — currently 16
        // for evalmod_r=3 (9 ER primes), 17 for evalmod_r=4 (10 ER primes).
        [[nodiscard]] std::size_t freshest_chain_index() const noexcept { return freshest_chain_index_; }

        // user_level: 0 = freshest, increments with each rescale_inplace,
        // max == num_scale_levels - 1 (pre_boot level; two primes remain).
        [[nodiscard]] int    user_level(const PhantomCiphertext &ct) const;

        // ---- Encoding / encryption ----
        [[nodiscard]] PhantomPlaintext encode(const std::vector<std::complex<double>> &v,
                                              int user_level = 0);
        [[nodiscard]] PhantomCiphertext encrypt(const PhantomPlaintext &pt);
        [[nodiscard]] std::vector<std::complex<double>> decrypt_decode(const PhantomCiphertext &ct);

        // ---- Arithmetic ----
        // Both operands must be at the same user_level.
        void add_inplace(PhantomCiphertext &dst, const PhantomCiphertext &src);
        void sub_inplace(PhantomCiphertext &dst, const PhantomCiphertext &src);
        void mul_and_relin_inplace(PhantomCiphertext &dst, const PhantomCiphertext &src);
        void mul_plain_inplace(PhantomCiphertext &dst, const PhantomPlaintext &pt);

        // ---- Level management ----
        void rescale_inplace(PhantomCiphertext &ct);    // user_level++
        void rotate_inplace(PhantomCiphertext &ct, int step);

        // Reset user_level to 0. Accepts any user_level in [0, max_user_level()].
        // The engine internally mod-switches to pre_boot_index (bottom - 1)
        // before calling bootstrap().
        void bootstrap_inplace(PhantomCiphertext &ct);

        // Diagnostic-only: identical to `bootstrap_inplace` but routes through
        // `phantom::bootstrap_debug` which prints stage-by-stage scale/chain/slot
        // metadata. Used exclusively by the BootstrapTo17Levels bisect probe.
        // Mutates `ct` in place; behavior otherwise matches bootstrap_inplace.
        void bootstrap_inplace_debug(PhantomCiphertext &ct);

        // ---- Read-only views (advanced; do NOT use to touch bootstrap primes) ----
        [[nodiscard]] const PhantomContext &context() const noexcept { return *ctx_; }
        [[nodiscard]] const PhantomCKKSEncoder &encoder() const noexcept { return *enc_; }
        [[nodiscard]] const PhantomSecretKey &secret_key() const noexcept { return *sk_; }
        [[nodiscard]] const PhantomRelinKey &relin_key() const noexcept { return bk_.relin_key; }
        [[nodiscard]] const PhantomGaloisKey &galois_key() const noexcept { return bk_.user_galois_keys; }
        // user_level chain index (for callers that need to encode plaintexts at
        // a specific level matching this engine's user-scale segment).
        [[nodiscard]] std::size_t user_level_chain_index(int user_level_) const {
            if (user_level_ < 0 || user_level_ >= cfg_.num_scale_levels) {
                throw std::invalid_argument(
                    "CKKSEngine::user_level_chain_index: user_level out of range");
            }
            return freshest_chain_index_ + static_cast<std::size_t>(user_level_);
        }
        [[nodiscard]] PhantomCKKSEncoder &mutable_encoder() noexcept { return *enc_; }
        [[nodiscard]] PhantomSecretKey &mutable_secret_key() noexcept { return *sk_; }

        // Per-level CKKS scale arrays (built when cfg.build_two_scale_arrays).
        // These mirror the_lib's `precomputed_.ckks_scale_` (squared / pre-rescale)
        // and `ckks_rescaled_scale_` (single / post-rescale) — see
        // src/ckks/scale.cpp:1420 (`make_ckks_scales`). Indexed by
        // chain_index in [0, total_parm_size()-1]:
        //   - ckks_scale_at(idx) = squared scale (i.e. (rescaled)^2 — the
        //     scale of an over-scaled ct at this level, just before rescale)
        //   - ckks_rescaled_scale_at(idx) = single scale (the scale of a
        //     freshly-rescaled ct at this level)
        // Returns 0.0 when arrays are not populated (build flag is false).
        [[nodiscard]] long double ckks_scale_at(std::size_t chain_index) const {
            return (chain_index < ckks_scale_.size())
                       ? ckks_scale_[chain_index]
                       : 0.0L;
        }
        [[nodiscard]] long double ckks_rescaled_scale_at(std::size_t chain_index) const {
            return (chain_index < ckks_rescaled_scale_.size())
                       ? ckks_rescaled_scale_[chain_index]
                       : 0.0L;
        }
        [[nodiscard]] std::size_t scale_array_size() const {
            return ckks_rescaled_scale_.size();
        }

    private:
        // Phase 1 helper: populate ckks_scale_ / ckks_rescaled_scale_ from
        // the chain primes. Called from the constructor when
        // cfg_.build_two_scale_arrays is true. Both legacy and the
        // BootstrapTo17Levels paths share this implementation.
        void build_scale_arrays();

        CKKSEngineConfig cfg_;
        std::unique_ptr<PhantomContext> ctx_;
        std::unique_ptr<PhantomCKKSEncoder> enc_;
        std::unique_ptr<PhantomSecretKey>   sk_;
        BootstrapKey bk_;

        // Chain-index mappings, computed once in the constructor.
        std::size_t freshest_chain_index_ = 0; // chain index for user_level=0
        std::size_t bottom_chain_index_   = 0; // chain index for user_level=max

        // Phase 1 (the_lib BootstrapTo17Levels port): per-chain-index scale
        // arrays. Built only when cfg_.build_two_scale_arrays is true.
        std::vector<long double> ckks_scale_;          // squared (pre-rescale)
        std::vector<long double> ckks_rescaled_scale_; // single (post-rescale)
    };

}  // namespace phantom
