// phantom::CKKSEngine — facade over the lapis 4-section heterogeneous CKKS
// chain + bootstrap pipeline. The user only ever sees a `user_level` counter;
// the C2S/ER/S2C/special primes are managed internally.
//
// Chain layout (bottom->top in bits[] order, special last):
//
//   msg(58) | scale(40) x num_scale_levels | S2C(58) x 3 |
//   ER(58) x 9 | C2S(29) x 3 | special(58) x num_special
//
// K=28 R=3 EvalMod (degree-49, from the_lib) consumes 9 levels (6 sine + 3 DA).
// This is the exact chain used by bootstrap_test's run_bootstrap_round_trip.
// size_Q = 1 + num_scale_levels + 3 + 9 + 3 = 16 + num_scale_levels.
// With num_scale_levels=14 and num_special=6: size_Q=30, 30%6=0 (dnum=5). ✓
//
// The bootstrap pipeline (scale_up_for_bootstrap) expects ct at
// pre_boot_index = bottom_chain_index_ - 1 (two primes: q_scale + q_msg)
// so it can multiply by q_msg/user_scale and rescale internally.
//
// User_level mapping:
//   freshest_chain_index_ = first_idx + num_c2s + eval_mod + num_s2c = 16
//   user_level k -> chain freshest + k
//   user_level 0  -> chain 16 (freshest)
//   user_level max -> chain 16 + num_scale_levels - 1 = 19 (pre_boot_index)
//   bottom          = chain 20 (single q_msg prime; bootstrap internal only)
//
// IMPORTANT: user_level max == num_scale_levels - 1 (NOT num_scale_levels).
// pre_boot_index = freshest + (num_scale_levels - 1) = bottom - 1.
// bootstrap_inplace mod-switches ct to pre_boot_index before calling bootstrap().

#include "ckks_engine.h"

#include "galois.cuh"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>

namespace phantom {

    // ---- Helpers (file-local) ----------------------------------------------

    namespace {

        std::vector<cuDoubleComplex>
        to_cu_complex(const std::vector<std::complex<double>> &v) {
            std::vector<cuDoubleComplex> out(v.size());
            for (std::size_t i = 0; i < v.size(); ++i) {
                out[i] = make_cuDoubleComplex(v[i].real(), v[i].imag());
            }
            return out;
        }

    } // namespace

    // ---- Constructor -------------------------------------------------------

    CKKSEngine::CKKSEngine(const CKKSEngineConfig &cfg) : cfg_(cfg) {
        if (cfg_.log_n != 16) {
            throw std::invalid_argument(
                "CKKSEngine: this port pins log_n=16");
        }
        if (cfg_.num_scale_levels < 2) {
            // We need at least 2 scale primes: 1 for pre_boot and 1 for the
            // user to hold a fresh ciphertext at user_level=0 distinct from
            // the pre_boot level. (user_level max = num_scale_levels - 1,
            // so num_scale_levels >= 2 gives at least user_level 0 and 1.)
            throw std::invalid_argument(
                "CKKSEngine: num_scale_levels must be >= 2");
        }
        if (cfg_.num_special_primes < 1) {
            throw std::invalid_argument(
                "CKKSEngine: num_special_primes must be >= 1");
        }
        // size_Q definition depends on which chain layout we build.
        //   - Default lapis-style: size_Q = 1(msg) + NSL + 3(S2C) + 9(ER) + 3(C2S)
        //   - BootstrapTo17Levels: size_Q = NSL + 1(scale_down) + 12(boot) + 1(C2S)
        //                                = NSL + 14   (no separate q_msg —
        //                                  bits[0] of user-scale segment IS q_msg,
        //                                  matching the_lib's convention)
        const int size_Q = cfg_.use_bootstrap_to_17_levels
            ? cfg_.num_scale_levels + 14
            : 1 + cfg_.num_scale_levels + 3 + 9 + 3;
        if (size_Q % cfg_.num_special_primes != 0) {
            throw std::invalid_argument(
                "CKKSEngine: size_Q must be divisible by num_special_primes "
                "(required by the hybrid KSK decomposition).");
        }

        EncryptionParameters parms(scheme_type::ckks);
        const std::size_t N = std::size_t(1) << cfg_.log_n;
        parms.set_poly_modulus_degree(N);
        parms.set_special_modulus_size(static_cast<std::size_t>(cfg_.num_special_primes));

        std::vector<int> bits;
        if (cfg_.use_bootstrap_to_17_levels) {
            // the_lib BootstrapTo17Levels chain (CKKS_42_54_29_40_60_BOOTSTRAP):
            //   bits[0..NSL-1]    = 40 (user-scale segment, bits[0] = q_msg)
            //   bits[NSL]         = 29 (single scale_down prime)
            //   bits[NSL+1..NSL+12] = 54 (bootstrap segment: 9 ER + 3 S2C)
            //   bits[NSL+13]      = 42 (single C2S prime)
            //   bits[NSL+14..]    = 60 (special primes, NSP)
            // S2C and ER both at 54-bit (the_lib's "bootstrap primes" pool).
            // Drop order from chain 1: bits[NSL+13]=42 (C2S, dropped first),
            // then 54×12 (S2C+ER), then 29 (scale_down), then 40×NSL.
            for (int i = 0; i < cfg_.num_scale_levels; ++i) {
                bits.push_back(40);                       // user-scale (msg = bits[0])
            }
            bits.push_back(29);                           // scale_down
            for (int i = 0; i < 12; ++i) bits.push_back(54); // bootstrap (ER+S2C)
            bits.push_back(42);                           // C2S (single layer)
            for (int i = 0; i < cfg_.num_special_primes; ++i) {
                bits.push_back(60);                       // special (large)
            }
        } else {
            // Lapis-style chain (legacy path).
            bits.push_back(58);                           // msg (q_0, bottom)
            for (int i = 0; i < cfg_.num_scale_levels; ++i) {
                bits.push_back(40);                       // user-scale segment
            }
            for (int i = 0; i < 3; ++i) bits.push_back(58);   // S2C
            for (int i = 0; i < 9; ++i) bits.push_back(58);   // ER / EvalMod (K=28 R=3)
            for (int i = 0; i < 3; ++i) bits.push_back(29);   // C2S
            for (int i = 0; i < cfg_.num_special_primes; ++i) {
                bits.push_back(58);                       // special
            }
        }
        parms.set_coeff_modulus(arith::CoeffModulus::Create(N, bits));

        // Galois rotations: union of C2S+S2C bootstrap steps + small user set
        // + conjugation (2N-1).
        const int num_slots = static_cast<int>(N >> 1);
        LinearTransformDiagonals c2s_h = build_c2s_diagonals(cfg_.log_n, {5, 5, 5});
        LinearTransformDiagonals s2c_h = build_s2c_diagonals(cfg_.log_n, {5, 5, 5});

        std::vector<int> all_steps = c2s_required_rotation_steps(c2s_h, num_slots);
        {
            auto s2c_steps = c2s_required_rotation_steps(s2c_h, num_slots);
            all_steps.insert(all_steps.end(), s2c_steps.begin(), s2c_steps.end());
        }

        if (!cfg_.user_rotation_steps.empty()) {
            // Explicit user rotation list overrides the default.
            for (int s : cfg_.user_rotation_steps) all_steps.push_back(s);
        } else if (cfg_.include_user_rotations) {
            // Minimal rotation set for the test. Production code should pass a
            // full power-of-2 ladder via cfg.user_rotation_steps.
            for (int s : {1, -1, 2, -2}) all_steps.push_back(s);
        }

        std::sort(all_steps.begin(), all_steps.end());
        all_steps.erase(std::unique(all_steps.begin(), all_steps.end()), all_steps.end());

        auto galois_elts = phantom::util::get_elts_from_steps(all_steps, N);
        galois_elts.push_back(static_cast<std::uint32_t>(2 * N - 1));  // conjugation
        parms.set_galois_elts(galois_elts);

        ctx_ = std::make_unique<PhantomContext>(parms);
        sk_  = std::make_unique<PhantomSecretKey>(*ctx_);
        enc_ = std::make_unique<PhantomCKKSEncoder>(*ctx_);

        // Phase 2: when on the BootstrapTo17Levels chain, skip the legacy
        // 3-stage bootstrap key build — its 3-layer C2S would consume into
        // the bootstrap segment instead of the single 42-bit C2S prime.
        // Phase 3+ wires up a single-stage bootstrap key matching the new
        // chain. For now the engine on this chain supports encode / encrypt
        // / decrypt only — bootstrap_inplace will throw.
        if (cfg_.use_bootstrap_to_17_levels) {
            // the_lib's BootstrapTo17Levels pipeline (verified against
            // src/ckks/engine/bootstrap.cpp:1026-1095 and :204-215):
            //   * coeff_to_slot_complex_for_17_levels = 3 OVER_SCALED butterflies
            //     + 1 rescale_after_multiply  →  consumes 1 chain prime (the
            //     42-bit C2S prime)
            //   * round = K·ct − modulo(ct), where modulo = sine + 3 DA
            //     (K=28 R=3 = 6 sine + 3 DA)  →  consumes 9 chain primes
            //     (the 9 × 54-bit "ER" segment)
            //   * slot_to_coeff (regular, rescale-first per stage,
            //     stage_count=3)  →  consumes 3 chain primes (the 3 × 54-bit
            //     "S2C" segment)
            // Total bootstrap-pipeline primes = 1 + 9 + 3 = 13.
            //
            // The 29-bit "scale_down" prime in the chain is NOT consumed by
            // the bootstrap pipeline — it is consumed by the user's FIRST
            // post-bootstrap rescale (level 17 → 16 in the_lib's level
            // numbering). Bootstrap output therefore lands at chain
            //     freshest_chain_index_ = first_idx + 13
            // which corresponds to user_level=0 with the 29-bit scale_down
            // prime as the active back prime; the user's first rescale_inplace
            // drops it and lands at user_level=1 with a 40-bit user-scale back.
            //
            // Sanity: with size_Q = NSL + 14 and bottom_chain = size_Q,
            //   freshest + NSL == bottom
            // becomes  (1 + 13) + NSL == NSL + 14  ✓
            const std::size_t first_idx_new = ctx_->get_first_index();
            freshest_chain_index_ = first_idx_new + 13;
            bottom_chain_index_   = ctx_->total_parm_size() - 1;

            // Sanity: freshest + NSL == bottom (still must hold).
            if (freshest_chain_index_ + static_cast<std::size_t>(cfg_.num_scale_levels)
                    != bottom_chain_index_) {
                throw std::logic_error(
                    "CKKSEngine: BootstrapTo17Levels chain mismatch: "
                    "freshest + num_scale_levels != bottom. "
                    "(scale_down + 13-prime bootstrap pipeline + NSL must equal "
                    "size_Q.)");
            }

            // Build scale arrays (Phase 1 path) and return early.
            if (cfg_.build_two_scale_arrays) {
                build_scale_arrays();
            }
            return;
        }

        bk_ = create_bootstrap_key(*ctx_, *enc_, *sk_,
                                   static_cast<std::size_t>(cfg_.sparse_hw),
                                   /*eval_mod_levels=*/9,
                                   cfg_.user_scale,
                                   cfg_.split_scale_down);

        // create_bootstrap_key partitions galois elts as bootstrap XOR user.
        // Steps that overlap C2S/S2C (e.g. step 1 ↔ galois_elt 5 used by both
        // a C2S layer AND user rms inner_sum) get classified as bootstrap-only
        // and are MISSING from bk_.user_galois_keys. User-code rotations
        // (e.g. phantom::rotate with bk_.user_galois_keys) then look up an
        // empty key and produce garbage. Override the user bundle with a
        // complete one covering conjugation + every configured user step.
        std::vector<int> user_steps_actual;
        if (!cfg_.user_rotation_steps.empty()) {
            user_steps_actual = cfg_.user_rotation_steps;
        } else if (cfg_.include_user_rotations) {
            user_steps_actual = {1, -1, 2, -2};
        }
        if (!user_steps_actual.empty()) {
            // Compute freshest_chain_index here (mirrors the math below) so we
            // can size user keys to the user-scale segment depth instead of full-Q.
            const std::size_t first_idx_local = ctx_->get_first_index();
            const std::size_t num_c2s_local   = bk_.c2s.layers.size();
            const std::size_t num_s2c_local   = bk_.s2c.layers.size();
            constexpr std::size_t kEvalMod_local = 9;
            const std::size_t fresh_local =
                first_idx_local + num_c2s_local + kEvalMod_local + num_s2c_local;

            // Per-step target chain indices: empty -> all at fresh_local
            // (default). Non-empty -> must match user_rotation_steps length and
            // is honoured verbatim (caller is responsible for picking valid
            // chain indices). Note: this only applies when the override path
            // is active (user_rotation_steps non-empty); the legacy
            // include_user_rotations defaults always go to fresh_local.
            const bool use_per_step_targets =
                !cfg_.user_rotation_target_chain_indices.empty();
            if (use_per_step_targets) {
                if (cfg_.user_rotation_steps.empty()) {
                    throw std::invalid_argument(
                        "CKKSEngine: user_rotation_target_chain_indices set but "
                        "user_rotation_steps is empty");
                }
                if (cfg_.user_rotation_target_chain_indices.size()
                        != cfg_.user_rotation_steps.size()) {
                    throw std::invalid_argument(
                        "CKKSEngine: user_rotation_target_chain_indices size "
                        "must match user_rotation_steps size");
                }
            }

            std::vector<std::size_t> user_indices;
            std::vector<std::size_t> user_target_levels;
            auto find_idx = [&](std::uint32_t elt) -> std::size_t {
                for (std::size_t i = 0; i < galois_elts.size(); ++i) {
                    if (galois_elts[i] == elt) return i;
                }
                throw std::runtime_error(
                    "CKKSEngine: configured galois elt not in registered set");
            };
            // Conjugation: bootstrap fires it at C2S-layer depth (chain < freshest);
            // must be full-Q so it covers all relevant chain indices.
            user_indices.push_back(
                find_idx(static_cast<std::uint32_t>(2 * N - 1)));
            user_target_levels.push_back(0);  // full-Q
            // Each user rotation step: per-step target if provided, else fresh_local.
            for (std::size_t i = 0; i < user_steps_actual.size(); ++i) {
                const int step = user_steps_actual[i];
                user_indices.push_back(find_idx(
                    phantom::util::get_elt_from_step(step, N)));
                if (use_per_step_targets) {
                    user_target_levels.push_back(
                        cfg_.user_rotation_target_chain_indices[i]);
                } else {
                    user_target_levels.push_back(fresh_local);
                }
            }
            bk_.user_galois_keys =
                sk_->create_galois_keys_per_level(*ctx_, user_indices, user_target_levels);
        }

        // Chain-index mapping (num_scale_levels=14, num_special=6):
        //   first_idx = 1
        //   C2S: [1..3], ER: [4..12], S2C: [13..15]
        //   freshest_chain_index_ = 16 (top of S2C, start of user scale segment)
        //   pre_boot_index_ = freshest + (num_scale_levels - 1) = 29
        //     (two primes: q_scale_last + q_msg, what scale_up_for_bootstrap needs)
        //   bottom_chain_index_ = total_parm_size() - 1 = 30 (single q_msg)
        const std::size_t first_idx = ctx_->get_first_index();
        const std::size_t num_c2s   = bk_.c2s.layers.size();
        const std::size_t num_s2c   = bk_.s2c.layers.size();
        constexpr std::size_t kEvalMod = 9;  // K=28 R=3: 6 sine + 3 DA

        freshest_chain_index_ = first_idx + num_c2s + kEvalMod + num_s2c;
        bottom_chain_index_   = ctx_->total_parm_size() - 1;

        // pre_boot_index = bottom - 1: the level that has exactly 2 primes
        // (one scale prime + q_msg), as required by scale_up_for_bootstrap.
        const std::size_t pre_boot_index = bottom_chain_index_ - 1;

        // Sanity checks:
        //   freshest + num_scale_levels == bottom (user segment spans exactly
        //   num_scale_levels primes, with pre_boot being the last one).
        if (freshest_chain_index_ + static_cast<std::size_t>(cfg_.num_scale_levels)
                != bottom_chain_index_) {
            throw std::logic_error(
                "CKKSEngine: chain layout mismatch: freshest + num_scale_levels "
                "!= bottom. Check chain construction.");
        }
        //   max user_level = num_scale_levels - 1 -> chain pre_boot_index
        if (freshest_chain_index_ + static_cast<std::size_t>(cfg_.num_scale_levels - 1)
                != pre_boot_index) {
            throw std::logic_error(
                "CKKSEngine: pre_boot_index mismatch");
        }

        if (cfg_.build_two_scale_arrays) {
            build_scale_arrays();
        }
    }

    // ---- Phase 1: per-level CKKS scale arrays --------------------------
    // Mirrors the_lib's `make_ckks_scales` (src/ckks/scale.cpp:1420).
    //
    // Recurrence (from the_lib lines 1429-1500 simplified for our chain):
    //   ckks_scale[0]          = user_scale²       (squared at top)
    //   ckks_rescaled_scale[L] = ckks_scale[L] / q_drop[L]   (rescale)
    //   ckks_scale[L+1]        = ckks_rescaled_scale[L]²     (next squared)
    //
    // q_drop[L] is the back prime at chain_index L — the prime that gets
    // dropped on rescale_to_next from L to L+1.
    //
    // For the legacy chain the values rapidly underflow because user_scale²
    // (= 2^80) is too small to survive 28 rescales by 58/40/29-bit primes.
    // For the BootstrapTo17Levels chain, where user_scale matches the
    // chain's small-prime size, the rescaled scale stays near user_scale
    // across the user-scale segment as expected.
    void CKKSEngine::build_scale_arrays() {
        const std::size_t total = ctx_->total_parm_size();
        ckks_scale_.assign(total, 0.0L);
        ckks_rescaled_scale_.assign(total, 0.0L);

        const long double user_scale_ld =
            static_cast<long double>(cfg_.user_scale);
        long double cur_scale = user_scale_ld * user_scale_ld;

        for (std::size_t idx = 0; idx < total; ++idx) {
            ckks_scale_[idx] = cur_scale;

            const auto &cd = ctx_->get_context_data(idx);
            const auto &mods = cd.parms().coeff_modulus();
            if (mods.empty()) {
                ckks_rescaled_scale_[idx] = cur_scale;
                break;
            }
            const long double q_drop =
                static_cast<long double>(mods.back().value());

            const long double rescaled = cur_scale / q_drop;
            ckks_rescaled_scale_[idx] = rescaled;
            cur_scale = rescaled * rescaled;
        }
    }

    // ---- User-facing properties --------------------------------------------

    std::size_t CKKSEngine::slot_count() const noexcept {
        return enc_->slot_count();
    }

    int CKKSEngine::user_level(const PhantomCiphertext &ct) const {
        const std::size_t ci = ct.chain_index();
        // User-visible range: [freshest_chain_index_, bottom_chain_index_ - 1].
        // bottom_chain_index_ itself (single q_msg prime) is internal-only.
        const std::size_t pre_boot = bottom_chain_index_ - 1;
        if (ci < freshest_chain_index_ || ci > pre_boot) {
            throw std::invalid_argument(
                "CKKSEngine::user_level: ciphertext chain_index is outside "
                "the user-scale segment");
        }
        return static_cast<int>(ci - freshest_chain_index_);
    }

    // ---- Encoding / encryption ---------------------------------------------

    PhantomPlaintext CKKSEngine::encode(const std::vector<std::complex<double>> &v,
                                        int user_level) {
        // user_level 0 .. max_user_level() - 1  (NOT num_scale_levels).
        if (user_level < 0 || user_level >= cfg_.num_scale_levels) {
            throw std::invalid_argument("CKKSEngine::encode: user_level out of range");
        }
        const std::size_t chain_index = freshest_chain_index_ + static_cast<std::size_t>(user_level);
        auto v_cu = to_cu_complex(v);
        PhantomPlaintext pt;
        enc_->encode(*ctx_, v_cu, cfg_.user_scale, pt, chain_index);
        return pt;
    }

    PhantomCiphertext CKKSEngine::encrypt(const PhantomPlaintext &pt) {
        PhantomCiphertext ct;
        sk_->encrypt_symmetric(*ctx_, pt, ct);
        return ct;
    }

    std::vector<std::complex<double>>
    CKKSEngine::decrypt_decode(const PhantomCiphertext &ct) {
        PhantomPlaintext dec_pt;
        sk_->decrypt(*ctx_, ct, dec_pt);
        std::vector<cuDoubleComplex> decoded_cu;
        enc_->decode(*ctx_, dec_pt, decoded_cu);
        std::vector<std::complex<double>> out(decoded_cu.size());
        for (std::size_t i = 0; i < decoded_cu.size(); ++i) {
            out[i] = {decoded_cu[i].x, decoded_cu[i].y};
        }
        return out;
    }

    // ---- Arithmetic --------------------------------------------------------

    void CKKSEngine::add_inplace(PhantomCiphertext &dst, const PhantomCiphertext &src) {
        phantom::add_inplace(*ctx_, dst, src);
    }

    void CKKSEngine::sub_inplace(PhantomCiphertext &dst, const PhantomCiphertext &src) {
        phantom::sub_inplace(*ctx_, dst, src);
    }

    void CKKSEngine::mul_and_relin_inplace(PhantomCiphertext &dst,
                                           const PhantomCiphertext &src) {
        phantom::multiply_and_relin_inplace(*ctx_, dst, src, bk_.relin_key);
    }

    void CKKSEngine::mul_plain_inplace(PhantomCiphertext &dst,
                                       const PhantomPlaintext &pt) {
        phantom::multiply_plain_inplace(*ctx_, dst, pt);
    }

    // ---- Level management --------------------------------------------------

    void CKKSEngine::rescale_inplace(PhantomCiphertext &ct) {
        // Guard: don't rescale past pre_boot_index. The pre_boot level (bottom-1)
        // is the deepest user-visible level; rescaling from there would produce
        // a single-prime ct at bottom_chain_index_ which is not user-accessible.
        if (ct.chain_index() >= bottom_chain_index_ - 1) {
            throw std::logic_error(
                "CKKSEngine::rescale_inplace: ciphertext is already at max "
                "user_level (pre_boot_index); call bootstrap_inplace next.");
        }
        phantom::rescale_to_next_inplace(*ctx_, ct);
    }

    void CKKSEngine::rotate_inplace(PhantomCiphertext &ct, int step) {
        phantom::rotate_inplace(*ctx_, ct, step, bk_.user_galois_keys);
    }

    // ---- Bootstrap ---------------------------------------------------------

    void CKKSEngine::bootstrap_inplace(PhantomCiphertext &ct) {
        // Accepts ct at any user_level in [0, max_user_level()].
        // max_user_level() = num_scale_levels - 1.
        // Internally mod-switches to pre_boot_index (bottom - 1), which
        // bootstrap() / scale_up_for_bootstrap() requires (two primes:
        // q_scale_last + q_msg).
        // Returns ct at:
        //   - freshest_chain_index_         (user_level = 0)  [non-split]
        //   - freshest_chain_index_ + 1     (user_level = 1)  [split mode]
        //
        // Split-scale-down trades 1 user level for ~5 extra bits of
        // bootstrap precision (the_lib's BootstrapTo14Levels compact-mode
        // fix: separate scale_down_ratio division to a post-bootstrap step).
        if (cfg_.use_bootstrap_to_17_levels) {
            throw std::logic_error(
                "CKKSEngine::bootstrap_inplace: use_bootstrap_to_17_levels chain "
                "does not have a bootstrap algorithm yet. Phase 3+ of the "
                "the_lib BootstrapTo17Levels port (single-stage C2S, encode-"
                "then-rescale diagonals, rescale-first S2C, scale_down) wires "
                "this up. For now the new chain supports encode/encrypt/"
                "decrypt only.");
        }
        const int lvl = user_level(ct);  // validates range
        (void)lvl;

        const std::size_t pre_boot_index = bottom_chain_index_ - 1;
        if (ct.chain_index() != pre_boot_index) {
            phantom::mod_switch_to_inplace(*ctx_, ct, pre_boot_index);
        }

        // Snap scale to remove FP drift from 40-bit rescales.
        ct.set_scale(cfg_.user_scale);

        ct = phantom::bootstrap(*ctx_, *enc_, ct, bk_, cfg_.user_scale,
                                cfg_.split_scale_down);

        const std::size_t expected_chain =
            cfg_.split_scale_down ? freshest_chain_index_ + 1
                                  : freshest_chain_index_;
        if (ct.chain_index() != expected_chain) {
            throw std::logic_error(
                "CKKSEngine::bootstrap_inplace: post-bootstrap chain_index "
                "!= expected (freshest"
                + std::string(cfg_.split_scale_down ? "+1" : "") + ")");
        }
        ct.set_scale(cfg_.user_scale);
    }

}  // namespace phantom
