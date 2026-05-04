#include "attention.h"

#include <stdexcept>

#include "evaluate.cuh"
#include "linear.h"

namespace phantom {

    std::vector<PhantomCiphertext> compute_qkt(
            const PhantomContext &ctx,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &q,
            const std::vector<PhantomCiphertext> &packed_k,
            std::size_t d_head) {
        const double nominal = q.scale();
        std::vector<PhantomCiphertext> scores;
        scores.reserve(packed_k.size());
        for (const auto &k_ct : packed_k) {
            PhantomCiphertext prod = multiply_and_relin(ctx, q, k_ct, relin_key);
            prod = rescale_to_next(ctx, prod);
            prod.set_scale(nominal);
            prod = inner_sum(ctx, galois_key, prod, d_head);
            scores.push_back(std::move(prod));
        }
        return scores;
    }

    PhantomCiphertext score_times_v(
            const PhantomContext &ctx,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const std::vector<PhantomCiphertext> &score_cts,
            const std::vector<PhantomCiphertext> &v_cts,
            const PhantomPlaintext &mask_pt,
            std::size_t d_head,
            std::size_t d_total,
            std::size_t positions_per_ct) {
        if (score_cts.empty()) {
            throw std::invalid_argument("score_times_v: score_cts is empty");
        }
        if (score_cts.size() != v_cts.size()) {
            throw std::invalid_argument("score_times_v: score_cts and v_cts size mismatch");
        }
        if (d_head == 0 || (d_head & (d_head - 1)) != 0) {
            throw std::invalid_argument("score_times_v: d_head must be a power of 2");
        }
        if (positions_per_ct == 0 || (positions_per_ct & (positions_per_ct - 1)) != 0) {
            throw std::invalid_argument("score_times_v: positions_per_ct must be a power of 2");
        }

        const double nominal = score_cts[0].scale();
        const std::size_t max_accumulate = positions_per_ct * d_total;

        PhantomCiphertext total_acc;
        bool have_total = false;

        for (std::size_t idx = 0; idx < score_cts.size(); ++idx) {
            // 1. Mask: pt × ct (mask is at score's chain_index already).
            PhantomCiphertext masked = multiply_plain(ctx, score_cts[idx], mask_pt);
            masked = rescale_to_next(ctx, masked);
            masked.set_scale(nominal);

            // 2. Broadcast within d_head blocks via negative-stride rotations.
            std::size_t bstride = d_head / 2;
            while (bstride >= 1) {
                PhantomCiphertext rot = rotate(ctx, masked, -static_cast<int>(bstride), galois_key);
                add_inplace(ctx, masked, rot);
                if (bstride == 1) break;
                bstride >>= 1;
            }

            // 3. ct × ct: score_broadcast × V. Drop V to masked's level first.
            PhantomCiphertext v_at_level = mod_switch_to(ctx, v_cts[idx], masked.chain_index());
            PhantomCiphertext prod = multiply_and_relin(ctx, masked, v_at_level, relin_key);
            prod = rescale_to_next(ctx, prod);
            prod.set_scale(nominal);

            // 4. Accumulate across packed positions: tree-sum by d_total strides.
            std::size_t astride = d_total;
            while (astride < max_accumulate) {
                PhantomCiphertext rot = rotate(ctx, prod, static_cast<int>(astride), galois_key);
                add_inplace(ctx, prod, rot);
                astride <<= 1;
            }

            // 5. Cross-chunk add.
            if (!have_total) {
                total_acc = std::move(prod);
                have_total = true;
            } else {
                add_inplace(ctx, total_acc, prod);
            }
        }

        return total_acc;
    }

}
