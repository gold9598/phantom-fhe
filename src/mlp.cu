#include "mlp.h"

#include <cuComplex.h>

#include <cstddef>
#include <vector>

#include "evaluate.cuh"
#include "ps.h"

namespace {
    // Degree-8 Chebyshev fit of SiLU(x) = x*sigmoid(x) on [-2, 2].
    // L_inf fit err ~2.74e-5. Consumes 5 levels (PS depth m+l-1=5).
    // Inlined here from the deleted silu.cu — used only by mlp_forward
    // and mlp_forward_complex below.
    const std::vector<double> SILU_COEFFS_DEG5_R2 = {
            -4.596475110252296e-16,  // x^0
             0.4999999999999997,     // x^1
             0.24991346416127744,    // x^2
             4.3792130415770066e-16, // x^3
            -0.020536112891251988,   // x^4
            -2.1433472280957883e-16, // x^5
             0.0017936119840762485,  // x^6
             2.929755203871941e-17,  // x^7
            -9.492355514915103e-05,  // x^8
    };

    PhantomCiphertext silu_polynomial(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomCiphertext &ct) {
        return phantom::eval_polynomial(ctx, encoder, relin_key, ct, SILU_COEFFS_DEG5_R2);
    }
}

namespace phantom {

    PhantomCiphertext mlp_forward(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            const MlpWeights &w) {
        const double nominal = x.scale();

        // 1: gate = W_gate * x  (chain k+1)
        PhantomCiphertext gate = bsgs_matmul_preencoded(ctx, galois_key, x, w.w_gate);

        // 2: up = W_up * x  (chain k+1)
        PhantomCiphertext up = bsgs_matmul_preencoded(ctx, galois_key, x, w.w_up);

        // 3: gate_silu = silu(gate)  (consumes ps depth = m+l-2 = 4 chain levels)
        PhantomCiphertext gate_silu = silu_polynomial(ctx, encoder, relin_key, gate);

        // 4: drop up to gate_silu's level so the ct*ct multiply has matching levels.
        mod_switch_to_inplace(ctx, up, gate_silu.chain_index());

        // 5: h = gate_silu * up; relin; rescale; snap nominal scale.
        PhantomCiphertext h = multiply_and_relin(ctx, gate_silu, up, relin_key);
        h = rescale_to_next(ctx, h);
        h.set_scale(nominal);

        // 6: y = W_down * h
        PhantomCiphertext y = bsgs_matmul_preencoded(ctx, galois_key, h, w.w_down);
        return y;
    }

    // ---- Complex-folded MLP ----

    namespace {

        // Encode a constant pt = (re + i*im) at given chain index and scale.
        PhantomPlaintext encode_complex_constant_pt(
                const PhantomContext &ctx,
                PhantomCKKSEncoder &encoder,
                double re,
                double im,
                std::size_t chain_index,
                double scale) {
            const std::size_t slots = encoder.slot_count();
            std::vector<cuDoubleComplex> values(slots, make_cuDoubleComplex(re, im));
            PhantomPlaintext pt;
            encoder.encode<cuDoubleComplex>(ctx, values, scale, pt, chain_index);
            return pt;
        }

        // Multiply a ciphertext by a complex constant (re + i*im) in-place.
        // Costs +1 level (rescale after multiply_plain).
        void mul_complex_constant_inplace(
                const PhantomContext &ctx,
                PhantomCKKSEncoder &encoder,
                PhantomCiphertext &ct,
                double re,
                double im) {
            const double nominal = ct.scale();
            const double q = static_cast<double>(
                    ctx.get_context_data(ct.chain_index()).parms().coeff_modulus().back().value());
            PhantomPlaintext pt = encode_complex_constant_pt(
                    ctx, encoder, re, im, ct.chain_index(), q);
            multiply_plain_inplace(ctx, ct, pt);
            ct = rescale_to_next(ctx, ct);
            ct.set_scale(nominal);
        }

        // re(z) = (z + conj(z)) / 2. Costs +1 level (the *0.5 rescale).
        // The conjugation is `rotate(z, step=0, galois_key)` which uses the
        // galois element 2N-1 (no rescale; key-switch only).
        PhantomCiphertext extract_real_part_ct(
                const PhantomContext &ctx,
                PhantomCKKSEncoder &encoder,
                const PhantomGaloisKey &galois_key,
                const PhantomCiphertext &z) {
            PhantomCiphertext cj = rotate(ctx, z, 0, galois_key);
            PhantomCiphertext sum = z;
            add_inplace(ctx, sum, cj);
            mul_complex_constant_inplace(ctx, encoder, sum, 0.5, 0.0);
            return sum;
        }

        // (z - conj(z)) / 2 = i * im(z). Note: the result carries the value
        // `i * im(z)` at each slot, so element-wise multiplying two such cts
        // gives (i*a)(i*b) = -a*b as a real-only ct. Costs +1 level.
        PhantomCiphertext extract_imag_i_form_ct(
                const PhantomContext &ctx,
                PhantomCKKSEncoder &encoder,
                const PhantomGaloisKey &galois_key,
                const PhantomCiphertext &z) {
            PhantomCiphertext cj = rotate(ctx, z, 0, galois_key);
            PhantomCiphertext diff = z;
            sub_inplace(ctx, diff, cj);  // diff = z - cj
            mul_complex_constant_inplace(ctx, encoder, diff, 0.5, 0.0);
            return diff;
        }

    } // namespace

    PhantomCiphertext mlp_forward_complex(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            const MlpWeightsComplex &w) {
        const double nominal = x.scale();

        // Stage 1: complex-packed gate + up BSGS (each yields a complex ct
        // whose real = top half of W*x, imag = bottom half).
        // Share baby rotations of x between Wgate and Wup to save M-1 rotations
        // (w_gate.baby_steps == w_up.baby_steps by construction in setup).
        auto babies = compute_bsgs_babies(ctx, galois_key, x, w.w_gate.inner.baby_steps);
        PhantomCiphertext gate_cplx = bsgs_apply_giants_with_babies_complex(
                ctx, galois_key, babies, w.w_gate);
        PhantomCiphertext up_cplx = bsgs_apply_giants_with_babies_complex(
                ctx, galois_key, babies, w.w_up);

        // Stage 2: extract real (top half) and i-form imag (bottom half).
        PhantomCiphertext gate_top = extract_real_part_ct(ctx, encoder, galois_key, gate_cplx);
        PhantomCiphertext gate_bot = extract_imag_i_form_ct(ctx, encoder, galois_key, gate_cplx);
        PhantomCiphertext up_top = extract_real_part_ct(ctx, encoder, galois_key, up_cplx);
        PhantomCiphertext up_bot = extract_imag_i_form_ct(ctx, encoder, galois_key, up_cplx);

        // Stage 3: silu on top and bottom halves.
        // For the bottom half, the input is `i * (W_bot*x)`. Applying the
        // silu polynomial (with non-zero even-degree terms) at an imaginary
        // argument does NOT yield i * silu(real input). To get the right
        // semantic, twist by -i first to produce the real-valued (W_bot*x),
        // then silu it, then we'll twist up_bot back to real for the hmult.
        // -i * (i * im) = im (real-valued).
        mul_complex_constant_inplace(ctx, encoder, gate_bot, 0.0, -1.0);
        mul_complex_constant_inplace(ctx, encoder, up_bot, 0.0, -1.0);

        PhantomCiphertext gate_top_silu = silu_polynomial(ctx, encoder, relin_key, gate_top);
        PhantomCiphertext gate_bot_silu = silu_polynomial(ctx, encoder, relin_key, gate_bot);

        mod_switch_to_inplace(ctx, up_top, gate_top_silu.chain_index());
        mod_switch_to_inplace(ctx, up_bot, gate_bot_silu.chain_index());

        PhantomCiphertext h_top = multiply_and_relin(ctx, gate_top_silu, up_top, relin_key);
        h_top = rescale_to_next(ctx, h_top);
        h_top.set_scale(nominal);

        PhantomCiphertext h_bot = multiply_and_relin(ctx, gate_bot_silu, up_bot, relin_key);
        h_bot = rescale_to_next(ctx, h_bot);
        h_bot.set_scale(nominal);

        // Stage 5: re-pack into h_cplx = h_top + i * h_bot. Multiply h_bot
        // by (0 + i) so it occupies the imag slot; then h_top + that.
        mul_complex_constant_inplace(ctx, encoder, h_bot, 0.0, 1.0);
        // Align levels so add_inplace can run. Phantom only allows mod-switch
        // forward (smaller-modulus -> smaller-modulus), so move the deeper
        // ct (lower chain_index = more primes) up to match.
        if (h_top.chain_index() > h_bot.chain_index()) {
            mod_switch_to_inplace(ctx, h_bot, h_top.chain_index());
        } else if (h_bot.chain_index() > h_top.chain_index()) {
            mod_switch_to_inplace(ctx, h_top, h_bot.chain_index());
        }
        add_inplace(ctx, h_top, h_bot);

        // Stage 6: down BSGS (col-folded with conjugate so real part = full y).
        PhantomCiphertext y = bsgs_matmul_preencoded_complex(
                ctx, galois_key, h_top, w.w_down);
        return y;
    }

}
