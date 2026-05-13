#include "ps.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "evaluate.cuh"

namespace phantom {

    namespace {

        // Encode `c` replicated across all slots at scale = `scale`.
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

        // ct *= c via encode-at-q + multiply_plain + rescale.
        // Snaps ct.scale back to `nominal` after rescale (matched-prime invariant).
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

        // Add scalar `c` in <=100.0 pieces via add_plain (no chain change).
        // Encodes each piece at ct.scale() so phantom's scale-match check passes.
        // Threshold was 0.1 historically (for tiny eps adds in rmsnorm where
        // encoding a value < precision_floor would otherwise round to zero),
        // but the PS evaluator for high-degree polynomial fits passes chunk
        // constants on the order of 10^4 — splitting those into 10^5 pieces
        // dominated runtime (a deg=20 silu evaluation took 229s, ~750k splits).
        // CKKS encoding precision at SCALE=2^40 is ~10^-13 absolute, so
        // encoding a value of magnitude 100 in one piece has the same
        // precision as 1000 pieces of 0.1. The larger threshold is safe for
        // any value below ~10^10 (well below 60-bit prime).
        void add_scalar_split_inplace(
                const PhantomContext &ctx,
                PhantomCKKSEncoder &encoder,
                PhantomCiphertext &ct,
                double scalar) {
            if (scalar == 0.0) return;
            const double abs_v = std::fabs(scalar);
            std::size_t pieces = static_cast<std::size_t>(std::ceil(abs_v / 100.0));
            if (pieces < 1) pieces = 1;
            const double piece = scalar / static_cast<double>(pieces);
            for (std::size_t i = 0; i < pieces; ++i) {
                PhantomPlaintext pt = encode_scalar_pt(
                        ctx, encoder, piece, ct.chain_index(), ct.scale());
                add_plain_inplace(ctx, ct, pt);
            }
        }

        // Build c_k(x) = a_{km} + sum_{j=1..m-1} a_{km+j} * babies[j-1] at
        // `target_chain_index`. Each linear term: mod-switch baby to target-1,
        // scalar-mul (advances to target), then accumulate. Nominal scale Δ is
        // propagated so all results share the same scale for add_inplace.
        PhantomCiphertext build_c_chunk(
                const PhantomContext &ctx,
                PhantomCKKSEncoder &encoder,
                const std::vector<double> &coeffs,
                const std::vector<PhantomCiphertext> &babies,
                std::size_t k,
                std::size_t m,
                std::size_t target_chain_index,
                double nominal) {
            const std::size_t d = coeffs.size() - 1;
            const std::size_t base = k * m;

            std::vector<std::pair<std::size_t, double>> linear_terms;
            for (std::size_t j = 1; j < m; ++j) {
                const std::size_t idx = base + j;
                if (idx > d) break;
                if (coeffs[idx] != 0.0) linear_terms.emplace_back(j, coeffs[idx]);
            }
            const double const_term = (base <= d) ? coeffs[base] : 0.0;

            // target-1 is the chain index before the scalar mul advances by 1.
            const std::size_t prev_ci = (target_chain_index > 0)
                    ? (target_chain_index - 1)
                    : 0;

            bool have_acc = false;
            PhantomCiphertext acc;
            for (const auto &lt : linear_terms) {
                const std::size_t j = lt.first;
                const double a = lt.second;
                PhantomCiphertext t = babies[j - 1];
                mod_switch_to_inplace(ctx, t, prev_ci);
                mul_scalar_inplace(ctx, encoder, t, a, nominal);
                if (!have_acc) {
                    acc = std::move(t);
                    have_acc = true;
                } else {
                    add_inplace(ctx, acc, t);
                }
            }

            if (!have_acc) {
                // No linear terms: encode a zero ciphertext at target via ×0.
                PhantomCiphertext z = babies[0];
                mod_switch_to_inplace(ctx, z, prev_ci);
                mul_scalar_inplace(ctx, encoder, z, 0.0, nominal);
                acc = std::move(z);
            }

            add_scalar_split_inplace(ctx, encoder, acc, const_term);
            return acc;
        }

    } // namespace

    PhantomCiphertext eval_polynomial(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomCiphertext &ct,
            const std::vector<double> &coeffs) {
        if (coeffs.size() < 2) {
            throw std::invalid_argument("eval_polynomial: degree must be >= 1");
        }
        const std::size_t d = coeffs.size() - 1;
        const std::size_t m = static_cast<std::size_t>(
                std::ceil(std::sqrt(static_cast<double>(d + 1))));
        const std::size_t l = (d + 1 + m - 1) / m;

        const double nominal = ct.scale();
        const std::size_t input_chain = ct.chain_index();

        // ---- Stage 1: baby powers x^1 .. x^(m-1) ----
        std::vector<PhantomCiphertext> babies;
        babies.reserve(std::max<std::size_t>(m, 1));
        babies.push_back(ct);

        PhantomCiphertext x_run = ct;
        for (std::size_t i = 2; i < m; ++i) {
            PhantomCiphertext nxt = multiply_and_relin(ctx, babies[i - 2], x_run, relin_key);
            nxt = rescale_to_next(ctx, nxt);
            nxt.set_scale(nominal);
            babies.push_back(std::move(nxt));
            if (i < m - 1) {
                // Advance x_run one level so its chain_index matches babies[i-1]
                // for the next ct*ct multiply.
                mul_scalar_inplace(ctx, encoder, x_run, 1.0, nominal);
            }
        }

        const std::size_t baby_lvl = (m >= 2) ? (input_chain + m - 2) : input_chain;
        for (auto &b : babies) {
            mod_switch_to_inplace(ctx, b, baby_lvl);
        }
        mod_switch_to_inplace(ctx, x_run, baby_lvl);

        // ---- Stage 2: giant base x^m (only if l >= 2) ----
        bool have_xm = false;
        PhantomCiphertext x_m;
        if (l >= 2) {
            x_m = multiply_and_relin(ctx, babies.back(), x_run, relin_key);
            x_m = rescale_to_next(ctx, x_m);
            x_m.set_scale(nominal);
            have_xm = true;
        }

        // ---- Stage 3: Horner over giant chunks ----
        PhantomCiphertext result = build_c_chunk(
                ctx, encoder, coeffs, babies, l - 1, m, baby_lvl + 1, nominal);

        for (std::size_t kk = l - 1; kk-- > 0;) {
            const std::size_t r_lvl = result.chain_index();
            if (!have_xm) {
                throw std::logic_error("eval_polynomial: x_m absent with l>=2");
            }
            mod_switch_to_inplace(ctx, x_m, r_lvl);
            result = multiply_and_relin(ctx, result, x_m, relin_key);
            result = rescale_to_next(ctx, result);
            result.set_scale(nominal);

            PhantomCiphertext c_k = build_c_chunk(
                    ctx, encoder, coeffs, babies, kk, m, result.chain_index(), nominal);
            add_inplace(ctx, result, c_k);
        }

        return result;
    }

} // namespace phantom
