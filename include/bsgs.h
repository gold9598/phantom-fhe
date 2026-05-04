#pragma once

#include <cstddef>
#include <vector>

#include "ciphertext.h"
#include "ckks.h"
#include "context.cuh"
#include "secretkey.h"
#include "single_chain_plaintext.h"

namespace phantom {

    struct BsgsDiagonals {
        // Length d_pad, indexed [g*M + b], plaintext pre-rotated by g*baby_steps
        // (giant rotation baked in at encode time).
        std::vector<SingleChainPlaintext> diagonals;
        std::size_t d_pad;
        std::size_t baby_steps;
        std::size_t giant_steps;
    };

    // Pre-encode `matrix` (row-major, num_rows x num_cols) into BSGS diagonals
    // as SingleChainPlaintexts. d_pad must be a power of 2 with
    // num_slots % d_pad == 0 and d_pad >= max(num_rows, num_cols).
    // baby_steps * giant_steps must equal d_pad.
    BsgsDiagonals pre_encode_bsgs_diagonals(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const std::vector<double> &matrix,
            std::size_t num_rows,
            std::size_t num_cols,
            std::size_t d_pad,
            std::size_t baby_steps,
            double scale);

    // Galois steps the matmul's rotate calls need: powers of 2 in
    // [1, baby_steps) (for chained baby rotations) plus baby_steps itself
    // (giant Horner stride).
    std::vector<int> bsgs_required_steps(std::size_t baby_steps);

    // Compute y = W * x via BSGS. Input x in replicated-block layout
    // (period = d_pad). Output also in replicated-block layout, period
    // d_pad. Output values y[0..num_rows) live in slots [0..num_rows)
    // within each period; slots [num_rows..d_pad) are zero (pad).
    PhantomCiphertext bsgs_matmul_preencoded(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            const BsgsDiagonals &diags);

    // Compute the M=baby_steps rotations of x: babies[0]=x, babies[b]=rotate(x, b).
    // Reusable across multiple BSGS matmuls that share the same input.
    std::vector<PhantomCiphertext> compute_bsgs_babies(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            std::size_t baby_steps);

    // Apply BSGS giants given precomputed babies. Equivalent to
    // `bsgs_matmul_preencoded(ctx, galois_key, x, diags)` if `babies` was
    // produced from the same x with `baby_steps == diags.baby_steps`.
    PhantomCiphertext bsgs_apply_giants_with_babies(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const std::vector<PhantomCiphertext> &babies,
            const BsgsDiagonals &diags);

    // ---- Complex-folded BSGS ----
    //
    // Pack two real rows into one complex row to halve d_pad. Each diagonal
    // is encoded as a complex slot vector (real + imag both used). The
    // ct/pt multiply pattern is identical to the real BSGS — the only
    // difference is the plaintext now has nonzero imag. After the matmul
    // the result is a complex-packed ciphertext whose real and imag halves
    // must be extracted via conjugation.
    struct ComplexBsgsDiagonals {
        BsgsDiagonals inner;  // structurally identical; matmul reuses real path.
    };

    enum class ComplexFoldMode {
        // Row fold: pack rows i and i+num_rows/2 of (num_rows x num_cols) into
        // one complex row. Used by W_gate / W_up. After matmul:
        //   re(result)[i] = (M @ x)[i]
        //   im(result)[i] = (M @ x)[i + num_rows/2]
        // The folded matrix has shape (ceil(num_rows/2) x num_cols).
        Rows,
        // Column fold (with conjugate on the imag half): pack columns j and
        // j+num_cols/2 into a complex column with sign flip on imag.
        // pt[i][j] = M[i][j] - i * M[i][j + num_cols/2]
        // Used by W_down on a complex-packed input h = h_top + i * h_bot:
        //   real(pt . h)[i] = M_left[i] @ h_top + M_right[i] @ h_bot = (M @ h_full)[i]
        // The folded matrix has shape (num_rows x ceil(num_cols/2)).
        ColsConj,
    };

    // Encode a real (num_rows x num_cols) matrix as complex-folded BSGS
    // diagonals. The folded matrix has half the rows (Rows mode) or half the
    // cols (ColsConj mode), so d_pad must satisfy:
    //   Rows:     d_pad >= max(ceil(num_rows/2), num_cols)
    //   ColsConj: d_pad >= max(num_rows, ceil(num_cols/2))
    ComplexBsgsDiagonals pre_encode_bsgs_diagonals_complex(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const std::vector<double> &matrix,
            std::size_t num_rows,
            std::size_t num_cols,
            std::size_t d_pad,
            std::size_t baby_steps,
            double scale,
            ComplexFoldMode mode);

    // Same as bsgs_matmul_preencoded but for complex-folded diagonals.
    // Structurally identical: forwards to bsgs_matmul_preencoded.
    PhantomCiphertext bsgs_matmul_preencoded_complex(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            const ComplexBsgsDiagonals &diags);

    // Apply BSGS giants for complex-folded diagonals given precomputed babies.
    // Forwards to bsgs_apply_giants_with_babies(ctx, galois_key, babies, diags.inner).
    PhantomCiphertext bsgs_apply_giants_with_babies_complex(
            const PhantomContext &ctx,
            const PhantomGaloisKey &galois_key,
            const std::vector<PhantomCiphertext> &babies,
            const ComplexBsgsDiagonals &diags);

}
