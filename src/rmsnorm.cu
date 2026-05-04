#include "rmsnorm.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "evaluate.cuh"
#include "ps.h"

namespace phantom {

    namespace {

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

        void add_scalar_split_inplace(
                const PhantomContext &ctx,
                PhantomCKKSEncoder &encoder,
                PhantomCiphertext &ct,
                double scalar) {
            if (scalar == 0.0) return;
            const double abs_v = std::fabs(scalar);
            std::size_t pieces = static_cast<std::size_t>(std::ceil(abs_v / 0.1));
            if (pieces < 1) pieces = 1;
            const double piece = scalar / static_cast<double>(pieces);
            for (std::size_t i = 0; i < pieces; ++i) {
                PhantomPlaintext pt = encode_scalar_pt(
                        ctx, encoder, piece, ct.chain_index(), ct.scale());
                add_plain_inplace(ctx, ct, pt);
            }
        }

        // pt*ct + rescale + scale snap (the lazuli ct_mul_pt_rescale analog).
        void pt_mul_rescale_inplace(
                const PhantomContext &ctx,
                PhantomCiphertext &ct,
                const PhantomPlaintext &pt,
                double nominal) {
            multiply_plain_inplace(ctx, ct, pt);
            ct = rescale_to_next(ctx, ct);
            ct.set_scale(nominal);
        }

        // ---- chebyshev_monomial_coeffs helpers ----

        double binomial_f64(std::size_t n, std::size_t k) {
            if (k > n) return 0.0;
            if (k > n - k) k = n - k;
            double result = 1.0;
            for (std::size_t i = 0; i < k; ++i) {
                result *= static_cast<double>(n - i);
                result /= static_cast<double>(i + 1);
            }
            return result;
        }

        bool is_power_of_two(std::size_t v) {
            return v != 0 && (v & (v - 1)) == 0;
        }

        // File-local: kept here (instead of as a public symbol) so the Python
        // port owns the public surface. invsqrt_direct_prescaled below is the
        // only remaining caller.
        std::vector<double> chebyshev_monomial_coeffs_local(
                double z_min, double z_max, double exponent, std::size_t degree) {
            if (z_min <= 0.0) {
                throw std::invalid_argument("chebyshev_monomial_coeffs: z_min must be positive");
            }
            if (z_max <= z_min) {
                throw std::invalid_argument("chebyshev_monomial_coeffs: z_max must exceed z_min");
            }
            if (degree < 1) {
                throw std::invalid_argument("chebyshev_monomial_coeffs: degree must be >= 1");
            }

            const std::size_t n = degree + 1;
            const double half_span = 0.5 * (z_max - z_min);
            const double midpoint = 0.5 * (z_min + z_max);

            // 1. Chebyshev nodes in t in [-1, 1] and corresponding z values.
            std::vector<double> nodes_t(n);
            std::vector<double> nodes_z(n);
            std::vector<double> values(n);
            const double pi = 3.14159265358979323846;
            for (std::size_t j = 0; j < n; ++j) {
                nodes_t[j] = std::cos(pi * (static_cast<double>(j) + 0.5) / static_cast<double>(n));
                nodes_z[j] = midpoint + half_span * nodes_t[j];
                values[j] = std::pow(nodes_z[j], exponent);
            }

            // 2. Chebyshev coefficients c_k = (2/N or 1/N for k=0) * sum_j f * T_k(t_j).
            std::vector<double> c_cheb(n, 0.0);
            for (std::size_t k = 0; k < n; ++k) {
                double sum = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    const double theta_j = pi * (static_cast<double>(j) + 0.5) / static_cast<double>(n);
                    sum += values[j] * std::cos(static_cast<double>(k) * theta_j);
                }
                c_cheb[k] = (k == 0) ? sum / static_cast<double>(n)
                                     : 2.0 * sum / static_cast<double>(n);
            }

            // 3. Convert c_cheb (Chebyshev-T basis in t) to monomial basis in t.
            std::vector<std::vector<double>> t_monomial_for_k;
            t_monomial_for_k.reserve(n);
            t_monomial_for_k.push_back({1.0});
            if (n >= 2) {
                t_monomial_for_k.push_back({0.0, 1.0});
            }
            for (std::size_t k = 2; k < n; ++k) {
                const auto &prev = t_monomial_for_k[k - 1];
                const auto &prev_prev = t_monomial_for_k[k - 2];
                std::vector<double> curr(k + 1, 0.0);
                for (std::size_t i = 0; i < prev.size(); ++i) {
                    curr[i + 1] += 2.0 * prev[i];
                }
                for (std::size_t i = 0; i < prev_prev.size(); ++i) {
                    curr[i] -= prev_prev[i];
                }
                t_monomial_for_k.push_back(std::move(curr));
            }
            std::vector<double> t_coeffs(n, 0.0);
            for (std::size_t k = 0; k < n; ++k) {
                for (std::size_t i = 0; i < t_monomial_for_k[k].size(); ++i) {
                    t_coeffs[i] += c_cheb[k] * t_monomial_for_k[k][i];
                }
            }

            // 4. Substitute t = a*z + b, expand via binomial.
            const double a = 2.0 / (z_max - z_min);
            const double b = -(z_min + z_max) / (z_max - z_min);
            std::vector<double> z_coeffs(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                if (t_coeffs[i] == 0.0) continue;
                for (std::size_t j = 0; j <= i; ++j) {
                    const double binom = binomial_f64(i, j);
                    const double term = t_coeffs[i] * binom *
                                        std::pow(a, static_cast<double>(j)) *
                                        std::pow(b, static_cast<double>(i - j));
                    z_coeffs[j] += term;
                }
            }
            return z_coeffs;
        }

        // sum_of_squares: square (ct*ct -> relin -> rescale -> set_scale), then
        // log2(d_model) rotations + adds. Result has Sum_j x_j^2 broadcast across
        // every slot within each d_model period.
        PhantomCiphertext sum_of_squares(
                const PhantomContext &ctx,
                const PhantomRelinKey &relin_key,
                const PhantomGaloisKey &galois_key,
                const PhantomCiphertext &x,
                std::size_t d_model,
                double nominal) {
            if (!is_power_of_two(d_model)) {
                throw std::invalid_argument("sum_of_squares: d_model must be a power of 2");
            }

            PhantomCiphertext acc = multiply_and_relin(ctx, x, x, relin_key);
            acc = rescale_to_next(ctx, acc);
            acc.set_scale(nominal);

            std::size_t stride = 1;
            while (stride < d_model) {
                PhantomCiphertext rotated = rotate(ctx, acc, static_cast<int>(stride), galois_key);
                add_inplace(ctx, acc, rotated);
                stride <<= 1;
            }
            return acc;
        }

        // (a/d) * sum_x^2 + (a*eps), folding the invsqrt_direct prescale alpha
        // into the same scalar mul that does the /d_model mean.
        void mean_squares_plus_eps_alpha(
                const PhantomContext &ctx,
                PhantomCKKSEncoder &encoder,
                PhantomCiphertext &sum_sq,
                std::size_t d_model,
                double epsilon,
                double alpha,
                double nominal) {
            if (d_model == 0) {
                throw std::invalid_argument("mean_squares_plus_eps_alpha: d_model must be > 0");
            }
            const double scaled_eps = alpha * epsilon;
            if (std::fabs(scaled_eps) > 0.5) {
                throw std::invalid_argument(
                        "mean_squares_plus_eps_alpha: |alpha*epsilon| must be <= 0.5");
            }
            const double scale = alpha / static_cast<double>(d_model);
            mul_scalar_inplace(ctx, encoder, sum_sq, scale, nominal);
            if (scaled_eps != 0.0) {
                add_scalar_split_inplace(ctx, encoder, sum_sq, scaled_eps);
            }
        }

        // 1/sqrt(alpha*z) via PS on prescaled input. Output is sqrt(zgeo) * 1/sqrt(z);
        // setup_rmsnorm_weights already folded 1/sqrt(zgeo) into g.
        PhantomCiphertext invsqrt_direct_prescaled(
                const PhantomContext &ctx,
                PhantomCKKSEncoder &encoder,
                const PhantomRelinKey &relin_key,
                const PhantomCiphertext &zp,
                double z_min,
                double z_max,
                std::size_t degree) {
            if (z_min <= 0.0) {
                throw std::invalid_argument("invsqrt_direct_prescaled: z_min must be positive");
            }
            if (z_max <= z_min) {
                throw std::invalid_argument("invsqrt_direct_prescaled: z_max must exceed z_min");
            }
            if (degree < 1) {
                throw std::invalid_argument("invsqrt_direct_prescaled: degree must be >= 1");
            }
            const double zgeo = std::sqrt(z_min * z_max);
            const double alpha = 1.0 / zgeo;
            const double zp_min = z_min * alpha;
            const double zp_max = z_max * alpha;
            std::vector<double> coeffs = chebyshev_monomial_coeffs_local(zp_min, zp_max, -0.5, degree);
            return eval_polynomial(ctx, encoder, relin_key, zp, coeffs);
        }

    } // namespace

    PhantomCiphertext rmsnorm_forward(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            const RmsNormWeights &weights,
            const RmsNormParams &params) {
        if (!is_power_of_two(params.d_model)) {
            throw std::invalid_argument("rmsnorm_forward: d_model must be a power of 2");
        }

        const double nominal = x.scale();

        // Stage 1: sum_of_squares.
        PhantomCiphertext sum_sq = sum_of_squares(
                ctx, relin_key, galois_key, x, params.d_model, nominal);

        // Stage 2: (alpha/d) * sum_x^2 + (alpha*eps).
        const double zgeo = std::sqrt(params.z_min * params.z_max);
        const double alpha = 1.0 / zgeo;
        mean_squares_plus_eps_alpha(
                ctx, encoder, sum_sq, params.d_model, params.epsilon, alpha, nominal);

        // Stage 3: 1/sqrt via Chebyshev PS on the prescaled input.
        PhantomCiphertext w = invsqrt_direct_prescaled(
                ctx, encoder, relin_key, sum_sq, params.z_min, params.z_max, params.poly_degree);

        // Stage 4: x * w.
        PhantomCiphertext x_bumped = x;
        mod_switch_to_inplace(ctx, x_bumped, w.chain_index());
        PhantomCiphertext xw = multiply_and_relin(ctx, x_bumped, w, relin_key);
        xw = rescale_to_next(ctx, xw);
        xw.set_scale(nominal);

        // Stage 5: pt*ct rescale by tiled g.
        const std::size_t g_chain = xw.chain_index();
        const double q = static_cast<double>(
                ctx.get_context_data(g_chain).parms().coeff_modulus().back().value());
        PhantomPlaintext g_pt = encoder.encode<double>(ctx, weights.g_tiled_real, q, g_chain);
        pt_mul_rescale_inplace(ctx, xw, g_pt, nominal);

        return xw;
    }

}
