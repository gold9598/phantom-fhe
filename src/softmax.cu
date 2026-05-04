#include "softmax.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "evaluate.cuh"
#include "ps.h"

namespace phantom {

    namespace {

        // Degree-8 Chebyshev fit of exp(y) on [-1, 1]. L_inf err 1.2e-8 (~f64
        // ULP — effectively the polynomial is exact). PS depth 4. Used with
        // NUM_SQUARINGS=2; folded 1/2^k supports input x ∈ [-2^k, 2^k] = [-4, 4].
        // Empirically this beats deg-15 + 3 squarings — fewer squarings
        // means less CKKS noise growth, which dominates over polynomial fit
        // error at the f64-ULP regime.
        const std::vector<double> EXP_CHEB_COEFFS_DEG4_R2 = {
                1.0000000000000002,    // y^0
                0.9999999011179665,    // y^1
                0.49999999014536933,   // y^2
                0.16666798420023443,   // y^3
                0.04166679798739991,   // y^4
                0.008328598903862764,  // y^5
                0.001388416857145537,  // y^6
                0.00020469833492755798,// y^7
                2.542872206845459e-05, // y^8
        };

        // ---- Helpers duplicated from ps.cu (intentional duplication; see spec). ----

        PhantomPlaintext encode_scalar_pt(
                const PhantomContext &ctx,
                PhantomCKKSEncoder &encoder,
                double c,
                std::size_t chain_index,
                double scale) {
            const std::size_t slots = encoder.slot_count();
            std::vector<double> values(slots, c);
            return encoder.encode<double>(ctx, values, scale, chain_index);
        }

        void mul_scalar_inplace(
                const PhantomContext &ctx,
                PhantomCKKSEncoder &encoder,
                PhantomCiphertext &ct,
                double c,
                double nominal) {
            const double q = static_cast<double>(
                    ctx.get_context_data(ct.chain_index()).parms().coeff_modulus().back().value());
            PhantomPlaintext pt = encode_scalar_pt(ctx, encoder, c, ct.chain_index(), q);
            multiply_plain_inplace(ctx, ct, pt);
            ct = rescale_to_next(ctx, ct);
            ct.set_scale(nominal);
        }

        bool is_power_of_two(std::size_t v) {
            return v != 0 && (v & (v - 1)) == 0;
        }

        // File-local: kept here (instead of as a public symbol) so the
        // Python port owns the public surface. finalize_softmax (KEEP) is
        // the only remaining caller.
        PhantomCiphertext sum_reduce_stride_local(
                const PhantomContext &ctx,
                const PhantomGaloisKey &galois_key,
                const PhantomCiphertext &ct,
                std::size_t num_tokens,
                std::size_t stride) {
            if (!is_power_of_two(num_tokens)) {
                throw std::invalid_argument("sum_reduce_stride: num_tokens must be a power of 2");
            }
            if (stride == 0) {
                throw std::invalid_argument("sum_reduce_stride: stride must be > 0");
            }

            PhantomCiphertext acc = ct;
            std::size_t step = stride;
            std::size_t reach = 1;
            while (reach < num_tokens) {
                PhantomCiphertext rotated = rotate(ctx, acc, static_cast<int>(step), galois_key);
                add_inplace(ctx, acc, rotated);
                step <<= 1;
                reach <<= 1;
            }
            return acc;
        }

    } // namespace

    PhantomCiphertext ps_exp_init(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomCiphertext &scores,
            std::size_t num_tokens,
            std::size_t num_squarings,
            double extra_scale) {
        const double t = static_cast<double>(num_tokens);
        const double scale_exp = std::pow(2.0, static_cast<double>(num_squarings));
        const double t_factor = std::pow(t, -1.0 / scale_exp);
        const double lead = extra_scale * t_factor;

        // Use the deg-4 Chebyshev fit by default to save one PS depth level
        // (m+l-2 = 3 instead of deg-5's 4). Combined with the folded scaling
        // below, ps_exp_init drops from 5 levels (1 mul_scalar + 4 PS) to
        // 3 levels — saves 2 levels per attention block.
        const std::vector<double> &base_coeffs = EXP_CHEB_COEFFS_DEG4_R2;

        // Fold y = x/2^k into the polynomial coefficients:
        //   Σ c_i · (x/2^k)^i = Σ (c_i / 2^(i·k)) · x^i
        // so we evaluate the polynomial directly on `scores` (no pre-scaling).
        // Also bake the `lead` scaling factor.
        //
        // Validity range: the original Chebyshev fit on [-2, 2] is accurate
        // for |y| ≤ 2, i.e. |x| ≤ 2 · 2^k. With num_squarings=2 (k=2):
        // |x| ≤ 8 (the un-scaled scores' magnitude bound).
        std::vector<double> coeffs(base_coeffs.size());
        double inv_scale_pow = 1.0; // 1/2^(i*k) accumulated as i increases
        const double inv_scale_exp = 1.0 / scale_exp;
        for (std::size_t i = 0; i < coeffs.size(); ++i) {
            coeffs[i] = lead * base_coeffs[i] * inv_scale_pow;
            inv_scale_pow *= inv_scale_exp;
        }

        // Evaluate the (folded) polynomial directly on the un-scaled scores.
        return eval_polynomial(ctx, encoder, relin_key, scores, coeffs);
    }

    void square_iterations_inplace(
            const PhantomContext &ctx,
            const PhantomRelinKey &relin_key,
            PhantomCiphertext &ct,
            std::size_t num_squarings) {
        const double nominal = ct.scale();
        for (std::size_t i = 0; i < num_squarings; ++i) {
            ct = multiply_and_relin(ctx, ct, ct, relin_key);
            ct = rescale_to_next(ctx, ct);
            ct.set_scale(nominal);
        }
    }

    void square_iterations_damped_inplace(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            PhantomCiphertext &ct,
            const std::vector<double> &damps) {
        const double nominal = ct.scale();
        for (std::size_t i = 0; i < damps.size(); ++i) {
            ct = multiply_and_relin(ctx, ct, ct, relin_key);
            ct = rescale_to_next(ctx, ct);
            ct.set_scale(nominal);
            if (std::abs(damps[i] - 1.0) > 1e-12) {
                mul_scalar_inplace(ctx, encoder, ct, damps[i], nominal);
            }
        }
    }

    PhantomCiphertext softmax_correct(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomCiphertext &e_ct,
            const PhantomCiphertext &a_ct,
            std::size_t iters) {
        const double nominal = e_ct.scale();
        PhantomCiphertext x = e_ct;
        PhantomCiphertext a = a_ct;

        for (std::size_t it = 0; it < iters; ++it) {
            // x*a and a*a at one level deeper.
            PhantomCiphertext xa = multiply_and_relin(ctx, x, a, relin_key);
            xa = rescale_to_next(ctx, xa);
            xa.set_scale(nominal);
            PhantomCiphertext aa = multiply_and_relin(ctx, a, a, relin_key);
            aa = rescale_to_next(ctx, aa);
            aa.set_scale(nominal);

            // 2x via x+x; 2a via a+a; mod-switch down to xa/aa's level.
            PhantomCiphertext two_x = x;
            add_inplace(ctx, two_x, x);
            mod_switch_to_inplace(ctx, two_x, xa.chain_index());
            PhantomCiphertext two_a = a;
            add_inplace(ctx, two_a, a);
            mod_switch_to_inplace(ctx, two_a, aa.chain_index());

            // 2x - x*a, 2a - a*a via level-free sub_inplace (was: scalar mul -1.0
            // + add, which cost 1 level per negation = 2 levels saved per iter).
            sub_inplace(ctx, two_x, xa, false);
            sub_inplace(ctx, two_a, aa, false);

            x = std::move(two_x);
            a = std::move(two_a);
        }
        return x;
    }

    PhantomCiphertext finalize_softmax(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &e_ct,
            std::size_t num_tokens,
            std::size_t stride,
            std::size_t iters) {
        PhantomCiphertext a = sum_reduce_stride_local(ctx, galois_key, e_ct, num_tokens, stride);
        return softmax_correct(ctx, encoder, relin_key, e_ct, a, iters);
    }

}
