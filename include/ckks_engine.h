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

    private:
        CKKSEngineConfig cfg_;
        std::unique_ptr<PhantomContext> ctx_;
        std::unique_ptr<PhantomCKKSEncoder> enc_;
        std::unique_ptr<PhantomSecretKey>   sk_;
        BootstrapKey bk_;

        // Chain-index mappings, computed once in the constructor.
        std::size_t freshest_chain_index_ = 0; // chain index for user_level=0
        std::size_t bottom_chain_index_   = 0; // chain index for user_level=max
    };

}  // namespace phantom
