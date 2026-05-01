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
        // Verify size_Q % size_P == 0 for the hybrid key-switch decomposition.
        // size_Q = 1(msg) + num_scale_levels + 3(S2C) + 9(ER) + 3(C2S)
        // (K=28 R=3 EvalMod consumes 9 levels: 6 sine + 3 DA.)
        const int size_Q = 1 + cfg_.num_scale_levels + 3 + 9 + 3;
        if (size_Q % cfg_.num_special_primes != 0) {
            throw std::invalid_argument(
                "CKKSEngine: size_Q must be divisible by num_special_primes "
                "(required by the hybrid KSK decomposition).");
        }

        // Identical to bootstrap_test's run_bootstrap_round_trip chain.
        EncryptionParameters parms(scheme_type::ckks);
        const std::size_t N = std::size_t(1) << cfg_.log_n;
        parms.set_poly_modulus_degree(N);
        parms.set_special_modulus_size(static_cast<std::size_t>(cfg_.num_special_primes));

        std::vector<int> bits;
        bits.push_back(58);                               // msg (q_0, bottom)
        for (int i = 0; i < cfg_.num_scale_levels; ++i) {
            bits.push_back(40);                           // user-scale segment
        }
        for (int i = 0; i < 3; ++i) bits.push_back(58);   // S2C
        for (int i = 0; i < 9; ++i) bits.push_back(58);   // ER / EvalMod (K=28 R=3: 6 sine + 3 DA)
        for (int i = 0; i < 3; ++i) bits.push_back(29);   // C2S
        for (int i = 0; i < cfg_.num_special_primes; ++i) {
            bits.push_back(58);                           // special
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

        if (cfg_.include_user_rotations) {
            // Minimal rotation set for the test. Production code should pass a
            // full power-of-2 ladder via a config knob (not yet exposed).
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

        bk_ = create_bootstrap_key(*ctx_, *enc_, *sk_,
                                   static_cast<std::size_t>(cfg_.sparse_hw),
                                   /*eval_mod_levels=*/9,
                                   cfg_.user_scale);

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
        // q_scale_last + q_msg). Returns ct at freshest_chain_index_
        // (user_level = 0).
        const int lvl = user_level(ct);  // validates range
        (void)lvl;

        const std::size_t pre_boot_index = bottom_chain_index_ - 1;
        if (ct.chain_index() != pre_boot_index) {
            phantom::mod_switch_to_inplace(*ctx_, ct, pre_boot_index);
        }

        // Snap scale to remove FP drift from 40-bit rescales.
        ct.set_scale(cfg_.user_scale);

        ct = phantom::bootstrap(*ctx_, *enc_, ct, bk_, cfg_.user_scale);

        if (ct.chain_index() != freshest_chain_index_) {
            throw std::logic_error(
                "CKKSEngine::bootstrap_inplace: post-bootstrap chain_index "
                "!= freshest_chain_index_");
        }
        ct.set_scale(cfg_.user_scale);
    }

}  // namespace phantom
