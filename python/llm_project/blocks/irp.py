"""Cachemir Interleaved Replicated Packing (IRP) for ct-pt vector-matrix multiply.

Implements Section 4.1 + 4.2 of arXiv:2602.11470 (Cachemir).  Cuts plaintext count
from d (vanilla diagonal/BSGS) to d^2 / N for square ct-pt matvec.  For LLaMA-3-8B
with N = 32768, d = 4096 this is 4096 -> 512 plaintexts per Wq/Wk/Wv/Wo (8x cut).

Key notation: t = N / d,  K = d / t = d^2 / N.
  1) Preprocess: log(t) rotate-and-adds replicate A across t sub-positions.
  2) BSGS multiply-accumulate with M babies x G giants (M*G = K).
  3) Reduce: log(t) rotate-and-adds collapse sub-positions.
  4) Optional mask (Section 4.2): zero out the t-1 junk slots per d-block,
     or let the caller fuse the mask into a downstream pt-ct multiply.
"""

from typing import List, Optional

import numpy as np
import pyPhantom as phantom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _check_dims(N: int, d: int) -> tuple:
    if not _is_pow2(N) or not _is_pow2(d):
        raise ValueError(f"irp: N and d must be powers of 2, got N={N}, d={d}")
    if N % d != 0:
        raise ValueError(f"irp: N must be divisible by d, got N={N}, d={d}")
    if d * d % N != 0:
        raise ValueError(f"irp: d*d must be divisible by N (i.e. d >= N**0.5), got d={d}, N={N}")
    t = N // d
    K = d // t  # = d*d / N
    return t, K


def _build_irp_slots(matrix: np.ndarray, N: int, d: int, t: int, K: int,
                     M: int, G: int, dtype=np.float64) -> List[np.ndarray]:
    """Build the K = M*G slot arrays for IRP diagonal encoding (shared by
    GPU-plaintext and host-storage variants).

    Returns a list of N-element arrays ordered as [(j=0,g=0), ..., (j=M-1,g=G-1)].
    """
    i_idx = np.arange(d, dtype=np.int64)[:, None]    # (d, 1)
    r_idx = np.arange(t, dtype=np.int64)[None, :]    # (1, t)
    slot_arrays = []
    for j in range(M):
        for g in range(G):
            row = (i_idx + j + r_idx * K) % d        # (d, t)
            col = (i_idx - g * M) % d                 # (d, 1) -> broadcast
            block = matrix[row, np.broadcast_to(col, (d, t))]
            slots = np.zeros(N, dtype=dtype)
            slots.reshape(d, t)[:] = block if dtype == np.float64 else block.astype(dtype)
            slot_arrays.append(slots)
    return slot_arrays


# ---------------------------------------------------------------------------
# Plaintext encoding
# ---------------------------------------------------------------------------

def encode_irp_diagonals(
    ctx,
    encoder,
    matrix: np.ndarray,
    N: int,
    d: int,
    scale: float,
    chain_index: int,
    baby_steps: int = 1,
) -> List:
    """Encode the d x d weight matrix into K = d^2 / N IRP plaintexts.

    Plaintexts are ordered as [(j=0,g=0), ..., (j=M-1,g=G-1)] where M = baby_steps
    and G = K / M.  Pass the same `baby_steps` to `irp_matvec`.
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (d, d):
        raise ValueError(f"encode_irp_diagonals: matrix must be ({d}, {d}), got {matrix.shape}")
    t, K = _check_dims(N, d)
    M = baby_steps
    if not _is_pow2(M):
        raise ValueError(f"encode_irp_diagonals: baby_steps must be power of 2, got {M}")
    if K % M != 0:
        raise ValueError(f"encode_irp_diagonals: baby_steps={M} must divide K={K}")
    G = K // M

    slot_arrays = _build_irp_slots(matrix, N, d, t, K, M, G, dtype=np.float64)
    # Pass numpy arrays directly — the C++ binding has a numpy fast path that
    # avoids the GIL-held per-element tolist() marshalling.
    return [encoder.encode_double_vector(ctx, s, scale, chain_index)
            for s in slot_arrays]


def encode_irp_diagonals_host(
    ctx,
    encoder,
    matrix: np.ndarray,
    N: int,
    d: int,
    scale: float,
    baby_steps: int = 1,
) -> List:
    """Like `encode_irp_diagonals`, but stores K plaintexts as
    `SingleChainPlaintext`s on pinned host memory (expanded JIT in
    `irp_matvec_host`).
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (d, d):
        raise ValueError(f"encode_irp_diagonals_host: matrix must be ({d}, {d}), got {matrix.shape}")
    t, K = _check_dims(N, d)
    M = baby_steps
    if not _is_pow2(M):
        raise ValueError(f"encode_irp_diagonals_host: baby_steps must be power of 2, got {M}")
    if K % M != 0:
        raise ValueError(f"encode_irp_diagonals_host: baby_steps={M} must divide K={K}")
    G = K // M

    slot_arrays = _build_irp_slots(matrix, N, d, t, K, M, G, dtype=complex)
    # Pass numpy complex arrays directly — avoids the GIL-held tolist()
    # marshalling that previously dominated the Wq IRP pre-encode phase
    # (~50 ms per call × 256 SCPs × 32 layers = ~6 min per worker).
    return [phantom.encode_single_chain_plaintext(ctx, encoder, s, scale)
            for s in slot_arrays]


def encode_irp_diagonals_rect_host(
    ctx,
    encoder,
    matrix: np.ndarray,
    N: int,
    d_in: int,
    d_out: int,
    scale: float,
    baby_steps: int = 1,
) -> List:
    """Host-storage analog of `encode_irp_diagonals_rect`.

    Decomposes the rectangular weight into `alpha` square sub-blocks and
    encodes each via `encode_irp_diagonals_host`.
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (d_in, d_out):
        raise ValueError(f"encode_irp_diagonals_rect_host: matrix must be "
                         f"({d_in}, {d_out}), got {matrix.shape}")
    if d_in == d_out:
        raise ValueError("encode_irp_diagonals_rect_host: d_in == d_out -- use "
                         "encode_irp_diagonals_host (square) instead")
    d = min(d_in, d_out)
    alpha = max(d_in, d_out) // d
    _check_rect_dims(N, d, alpha)

    plaintexts = []
    for q in range(alpha):
        W_q = (matrix[:, q * d:(q + 1) * d] if d_in < d_out
               else matrix[q * d:(q + 1) * d, :])
        plaintexts.extend(encode_irp_diagonals_host(
            ctx, encoder, W_q, N=N, d=d, scale=scale, baby_steps=baby_steps,
        ))
    return plaintexts


def irp_matvec_host(
    ctx,
    encoder,
    gk,
    ct_in,
    plaintexts: List,
    N: int,
    d: int,
    baby_steps: int = 1,
    mask_pt=None,
    output_scale: float = None,
) -> "phantom.ciphertext":
    """Like `irp_matvec` but consumes host-stored `SingleChainPlaintext`s
    (expanded JIT to the running chain).  If `output_scale` is set, an extra
    rescale + set_scale snaps the output for a downstream ct*ct multiply.
    """
    t, K = _check_dims(N, d)
    M = baby_steps
    if not _is_pow2(M) or K % M != 0:
        raise ValueError(f"irp_matvec_host: invalid baby_steps={M} for K={K}")
    if len(plaintexts) != K:
        raise ValueError(f"irp_matvec_host: expected {K} plaintexts, got {len(plaintexts)}")
    G = K // M
    log_t = int(round(np.log2(t)))

    c_pp = ct_in
    for s in range(log_t):
        rot_amt = (1 << s) * (d - 1)
        rot_ct = phantom.rotate(ctx, c_pp, int(rot_amt), gk)
        c_pp = phantom.add(ctx, c_pp, rot_ct)

    babies = [c_pp]
    for j in range(1, M):
        babies.append(phantom.rotate(ctx, c_pp, int(j * t), gk))

    babies_chain = babies[0].chain_index()

    giant_partials: List[Optional[object]] = []
    for g in range(G):
        # giant_partials[g] = Sum_j plaintexts[j*G + g] . babies[j], computed in
        # one fused MAC kernel (numerically equivalent to the per-term
        # multiply_plain + add loop). expand_single_chain_to_full yields
        # full-RNS NTT-form plaintexts for every SCP dtype; fused_mac_accumulate
        # gathers them with no re-expand / no extra NTT.
        pts_g = [
            phantom.expand_single_chain_to_full(ctx, plaintexts[j * G + g], babies_chain)
            for j in range(M)
        ]
        giant_partials.append(phantom.fused_mac_accumulate(ctx, babies, pts_g))
        del pts_g

    acc = giant_partials[0]
    for g in range(1, G):
        rot_amt = g * M * t
        rotated = phantom.rotate(ctx, giant_partials[g], int(rot_amt), gk)
        acc = phantom.add(ctx, acc, rotated)

    for s in range(log_t):
        rot_amt = 1 << s
        rotated = phantom.rotate(ctx, acc, int(rot_amt), gk)
        acc = phantom.add(ctx, acc, rotated)

    if mask_pt is not None:
        nominal = acc.scale()
        acc = phantom.multiply_plain(ctx, acc, mask_pt)
        acc = phantom.rescale_to_next(ctx, acc)
        acc.set_scale(nominal)
    if output_scale is not None:
        acc = phantom.rescale_to_next(ctx, acc)
        acc.set_scale(float(output_scale))
    return acc


def irp_matvec_rect_host(
    ctx,
    encoder,
    gk,
    ct_in,
    plaintexts: List,
    N: int,
    d_in: int,
    d_out: int,
    baby_steps: int = 1,
    sub_mask_pt=None,
    input_mask_pt=None,
    output_scale: float = None,
):
    """Host-storage analog of `irp_matvec_rect`."""
    if d_in == d_out:
        return irp_matvec_host(ctx, encoder, gk, ct_in, plaintexts,
                                 N=N, d=d_in, baby_steps=baby_steps,
                                 mask_pt=sub_mask_pt, output_scale=output_scale)
    if sub_mask_pt is None:
        raise ValueError("irp_matvec_rect_host: sub_mask_pt required")

    if d_in < d_out:
        d = d_in
        alpha = d_out // d
        t, t_prime, K_sq, K_total = _check_rect_dims(N, d, alpha)
        if len(plaintexts) != K_total:
            raise ValueError(f"irp_matvec_rect_host (wide): expected {K_total} plaintexts, got {len(plaintexts)}")
        out = None
        for q in range(alpha):
            sub_pts = plaintexts[q * K_sq:(q + 1) * K_sq]
            sub_q = irp_matvec_host(ctx, encoder, gk, ct_in, sub_pts,
                                       N=N, d=d, baby_steps=baby_steps,
                                       mask_pt=sub_mask_pt,
                                       output_scale=output_scale)
            if q > 0:
                rot_amt = (N - q * t_prime) % N
                sub_q = phantom.rotate(ctx, sub_q, int(rot_amt), gk)
            if out is None:
                out = sub_q
            else:
                out = phantom.add(ctx, out, sub_q)
        return out

    # Tall path
    d = d_out
    alpha = d_in // d
    t, t_prime, K_sq, K_total = _check_rect_dims(N, d, alpha)
    if len(plaintexts) != K_total:
        raise ValueError(f"irp_matvec_rect_host (tall): expected {K_total} plaintexts, got {len(plaintexts)}")
    if input_mask_pt is None:
        raise ValueError("irp_matvec_rect_host (tall): input_mask_pt required")
    out = None
    for q in range(alpha):
        if q == 0:
            x_q_aligned = ct_in
        else:
            x_q_aligned = phantom.rotate(ctx, ct_in, int(q * t_prime), gk)
        nominal = x_q_aligned.scale()
        x_q_only = phantom.multiply_plain(ctx, x_q_aligned, input_mask_pt)
        x_q_only = phantom.rescale_to_next(ctx, x_q_only)
        x_q_only.set_scale(nominal)
        sub_pts = plaintexts[q * K_sq:(q + 1) * K_sq]
        sub_q = irp_matvec_host(ctx, encoder, gk, x_q_only, sub_pts,
                                   N=N, d=d, baby_steps=baby_steps,
                                   mask_pt=sub_mask_pt,
                                   output_scale=output_scale)
        if out is None:
            out = sub_q
        else:
            out = phantom.add(ctx, out, sub_q)
    return out


def encode_irp_mask(
    ctx,
    encoder,
    N: int,
    d: int,
    scale: float,
    chain_index: int,
):
    """Plaintext mask: 1 at slot i*t for i in [0, d), 0 elsewhere.

    Multiply the IRP output by this to zero out junk slots, or fold the mask
    into a downstream pt-ct multiply to save a level (Section 4.2).
    """
    t, _ = _check_dims(N, d)
    slots = np.zeros(N, dtype=np.float64)
    slots[::t][:d] = 1.0
    return encoder.encode_double_vector(ctx, slots, scale, chain_index)


# ---------------------------------------------------------------------------
# Required galois steps
# ---------------------------------------------------------------------------

def irp_required_steps(N: int, d: int, baby_steps: int = 1) -> List[int]:
    """All galois rotation steps used by `irp_matvec`.

    Steps:
      - Preprocess:  2^s * (d - 1)  for s = 0 .. log(t) - 1     (log(t) steps)
      - BSGS babies: j * t          for j = 1 .. M - 1          (M - 1 steps)
      - BSGS giants: g * M * t      for g = 1 .. G - 1          (G - 1 steps)
      - Reduce:      2^s            for s = 0 .. log(t) - 1     (log(t) steps)
    Total ~ 2*log(t) + (M - 1) + (G - 1).  Optimal M = G = sqrt(K) gives
    2*log(N/d) + 2*sqrt(d^2/N) - 2 distinct rotations.
    """
    t, K = _check_dims(N, d)
    M = baby_steps
    if not _is_pow2(M) or K % M != 0:
        raise ValueError(f"irp_required_steps: invalid baby_steps={M} for K={K}")
    G = K // M

    steps = set()
    log_t = int(round(np.log2(t)))
    # Preprocess
    for s in range(log_t):
        steps.add((1 << s) * (d - 1))
    # Babies
    for j in range(1, M):
        steps.add(j * t)
    # Giants
    for g in range(1, G):
        steps.add(g * M * t)
    # Reduce
    for s in range(log_t):
        steps.add(1 << s)
    return sorted(steps)


# ---------------------------------------------------------------------------
# Matvec
# ---------------------------------------------------------------------------

def irp_matvec(
    ctx,
    encoder,
    gk,
    ct_in,
    plaintexts: List,
    N: int,
    d: int,
    baby_steps: int = 1,
    mask_pt=None,
) -> "phantom.ciphertext":
    """Run the IRP ct-pt VMM (preprocess -> BSGS babies/giants -> reduce).

    ct_in must be in interleaved layout (slot[i*t] = A[i]).  plaintexts must
    come from `encode_irp_diagonals` with matching baby_steps.  If mask_pt is
    provided, applies mask + rescale; otherwise the caller fuses the mask
    downstream (Section 4.2).  Returns ciphertext in interleaved layout.
    """
    t, K = _check_dims(N, d)
    M = baby_steps
    if not _is_pow2(M) or K % M != 0:
        raise ValueError(f"irp_matvec: invalid baby_steps={M} for K={K}")
    if len(plaintexts) != K:
        raise ValueError(f"irp_matvec: expected {K} plaintexts, got {len(plaintexts)}")
    G = K // M
    log_t = int(round(np.log2(t)))

    # ---- Step 1: preprocess (replicate A into the interleaved layout) ----
    c_pp = ct_in
    for s in range(log_t):
        rot_amt = (1 << s) * (d - 1)
        rot_ct = phantom.rotate(ctx, c_pp, int(rot_amt), gk)
        c_pp = phantom.add(ctx, c_pp, rot_ct)

    # ---- Step 2: BSGS babies (M - 1 rotations) ----
    babies = [c_pp]
    for j in range(1, M):
        babies.append(phantom.rotate(ctx, c_pp, int(j * t), gk))

    # ---- Step 2/3: multiply-accumulate per giant + giant aggregation ----
    giant_partials: List[Optional[object]] = [None] * G
    for j in range(M):
        for g in range(G):
            pt = plaintexts[j * G + g]
            prod = phantom.multiply_plain(ctx, babies[j], pt)
            if giant_partials[g] is None:
                giant_partials[g] = prod
            else:
                giant_partials[g] = phantom.add(ctx, giant_partials[g], prod)

    # acc = sum_g rot(giant_partials[g], g * M * t)
    acc = giant_partials[0]
    for g in range(1, G):
        rot_amt = g * M * t
        rotated = phantom.rotate(ctx, giant_partials[g], int(rot_amt), gk)
        acc = phantom.add(ctx, acc, rotated)

    # ---- Step 4: reduce in ciphertext (log(t) rotations) ----
    for s in range(log_t):
        rot_amt = 1 << s
        rotated = phantom.rotate(ctx, acc, int(rot_amt), gk)
        acc = phantom.add(ctx, acc, rotated)

    # Optional mask + rescale (Section 4.2; when fused with downstream layer, skip this).
    if mask_pt is not None:
        nominal = acc.scale()
        acc = phantom.multiply_plain(ctx, acc, mask_pt)
        acc = phantom.rescale_to_next(ctx, acc)
        acc.set_scale(nominal)
    return acc


# ---------------------------------------------------------------------------
# Convenience: encrypt input in IRP layout
# ---------------------------------------------------------------------------

def encrypt_irp_input(
    ctx,
    encoder,
    sk,
    a: np.ndarray,
    N: int,
    d: int,
    scale: float,
    chain_index: int,
):
    """Encrypt vector A into the interleaved-with-zeros layout expected by `irp_matvec`.

    slot[i*t] = a[i] for i in [0, d); all other slots zero.
    """
    t, _ = _check_dims(N, d)
    if not isinstance(a, np.ndarray):
        a = np.asarray(a, dtype=np.float64)
    if a.shape != (d,):
        raise ValueError(f"encrypt_irp_input: a must have shape ({d},), got {a.shape}")
    slots = np.zeros(N, dtype=np.float64)
    slots[::t][:d] = a  # only first d strided positions
    pt = encoder.encode_double_vector(ctx, slots, scale, chain_index)
    return sk.encrypt_symmetric(ctx, pt)


def decode_irp_output(decoded_slots: np.ndarray, N: int, d: int) -> np.ndarray:
    """Extract the d valid output values from a decoded IRP result vector."""
    t, _ = _check_dims(N, d)
    return np.asarray(decoded_slots, dtype=np.float64)[::t][:d]


# ===========================================================================
# Rectangular IRP (Cachemir Appendix C: non-square matrices)
# ===========================================================================
#
# Wide  (d_in < d_out): alpha = d_out/d_in stacked square IRPs, outputs
#   combined via mask + rotation into permuted stride-t' layout.
# Tall  (d_in > d_out): alpha = d_in/d_out sub-IRPs, each consuming a
#   rotate+mask-extracted slice of the permuted input.
# Plaintext count: alpha * d^2 / N.  Wide output permutation is intentional
# (the down projection consumes it directly).

def _check_rect_dims(N: int, d: int, alpha: int) -> tuple:
    """Validate (N, d, alpha) and return (t, t', K_sq, K_total).

    t = N/d (stride for length-d ciphertext)
    t' = N/(alpha*d) (stride for length-alpha*d ciphertext)
    K_sq = d^2/N (per-sub-IRP plaintext count)
    K_total = alpha * K_sq = alpha * d^2 / N (total plaintexts)
    """
    if not _is_pow2(N) or not _is_pow2(d) or not _is_pow2(alpha):
        raise ValueError(f"irp_rect: N, d, alpha must be powers of 2; "
                         f"got N={N}, d={d}, alpha={alpha}")
    if N % (alpha * d) != 0:
        raise ValueError(f"irp_rect: N must be divisible by alpha*d; "
                         f"N={N}, alpha*d={alpha*d}")
    if d * d % N != 0:
        raise ValueError(f"irp_rect: d^2 must be divisible by N (i.e. d >= sqrt(N)); "
                         f"d={d}, N={N}")
    t = N // d
    t_prime = N // (alpha * d)
    K_sq = d * d // N
    K_total = alpha * K_sq
    return t, t_prime, K_sq, K_total


def encode_irp_diagonals_rect(
    ctx,
    encoder,
    matrix: np.ndarray,
    N: int,
    d_in: int,
    d_out: int,
    scale: float,
    chain_index: int,
    baby_steps: int = 1,
) -> List:
    """Encode a (d_in, d_out) weight matrix into alpha * d^2/N IRP plaintexts.

    matrix.shape must be (d_in, d_out).  Returns alpha * K_sq plaintexts,
    ordered as [W_0 sub-plaintexts, W_1, ...].  For tall matrices the sub-IRP
    plaintexts are encoded at chain_index + 1 (one level past the input mask).
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (d_in, d_out):
        raise ValueError(f"encode_irp_diagonals_rect: matrix must be "
                         f"({d_in}, {d_out}), got {matrix.shape}")

    if d_in == d_out:
        raise ValueError("encode_irp_diagonals_rect: d_in == d_out -- use "
                         "encode_irp_diagonals (square) instead")

    if d_in < d_out:
        # Wide: matrix shape (d, alpha*d), d = d_in, alpha = d_out/d_in
        d = d_in
        alpha = d_out // d_in
        if alpha * d != d_out:
            raise ValueError(f"encode_irp_diagonals_rect: d_out={d_out} must be a "
                             f"multiple of d_in={d_in}")
        _check_rect_dims(N, d, alpha)
        # Decompose W into alpha column blocks W_q in R^(d x d).
        # Wide plaintexts encoded at the input ct's chain_index (no input mask
        # needed in the wide path).
        plaintexts = []
        for q in range(alpha):
            W_q = matrix[:, q * d:(q + 1) * d]
            sub_pts = encode_irp_diagonals(
                ctx, encoder, W_q, N=N, d=d, scale=scale,
                chain_index=chain_index, baby_steps=baby_steps,
            )
            plaintexts.extend(sub_pts)
        return plaintexts
    else:
        # Tall: matrix shape (alpha*d, d), d = d_out, alpha = d_in/d_out
        d = d_out
        alpha = d_in // d_out
        if alpha * d != d_in:
            raise ValueError(f"encode_irp_diagonals_rect: d_in={d_in} must be a "
                             f"multiple of d_out={d_out}")
        _check_rect_dims(N, d, alpha)
        # Tall plaintexts must be at chain_index + 1 because the tall path
        # masks the input ciphertext (consuming one level via multiply_plain
        # + rescale_to_next) before invoking the square IRP machinery.  In
        # Phantom, rescale_to_next increases chain_index by 1, so the post-
        # mask ciphertext lives one chain level deeper.
        sub_chain = chain_index + 1
        plaintexts = []
        for q in range(alpha):
            W_q = matrix[q * d:(q + 1) * d, :]
            sub_pts = encode_irp_diagonals(
                ctx, encoder, W_q, N=N, d=d, scale=scale,
                chain_index=sub_chain, baby_steps=baby_steps,
            )
            plaintexts.extend(sub_pts)
        return plaintexts


def encode_irp_mask_rect(
    ctx,
    encoder,
    N: int,
    d_in: int,
    d_out: int,
    scale: float,
    chain_index: int,
):
    """Mask plaintext for the rectangular IRP intermediate (per-sub-IRP).

    Each sub-IRP's output is at stride t = N/d (where d is the *smaller*
    dimension), so the per-sub-IRP mask is identical to the square mask.
    """
    if d_in == d_out:
        raise ValueError("encode_irp_mask_rect: use encode_irp_mask for square")
    d = min(d_in, d_out)
    return encode_irp_mask(ctx, encoder, N=N, d=d, scale=scale, chain_index=chain_index)


def irp_required_steps_rect(
    N: int,
    d_in: int,
    d_out: int,
    baby_steps: int = 1,
) -> List[int]:
    """All galois rotation steps used by `irp_matvec_rect` for a (d_in, d_out) layer.

    Includes:
      - Square sub-IRP steps (preprocess, babies, giants, reduce) at d = min(d_in, d_out).
      - For wide: combine rotations N - q*t' for q ∈ [1, alpha) (i.e., right-rotate by q*t').
      - For tall: input alignment rotations q*t' for q ∈ [1, alpha).
    """
    if d_in == d_out:
        return irp_required_steps(N, d_in, baby_steps=baby_steps)
    d = min(d_in, d_out)
    alpha = max(d_in, d_out) // d
    _check_rect_dims(N, d, alpha)
    t_prime = N // (alpha * d)

    steps = set(irp_required_steps(N, d, baby_steps=baby_steps))
    if d_in < d_out:
        # Wide: combine via right-rotate by q*t' = left-rotate by N - q*t'
        for q in range(1, alpha):
            steps.add((N - q * t_prime) % N)
    else:
        # Tall: input alignment via left-rotate by q*t'
        for q in range(1, alpha):
            steps.add(q * t_prime)
    return sorted(s for s in steps if s != 0)


def encrypt_irp_input_rect(
    ctx,
    encoder,
    sk,
    a: np.ndarray,
    N: int,
    d_in: int,
    d_out: int,
    scale: float,
    chain_index: int,
):
    """Encrypt the input vector for a rectangular IRP layer.

    For wide (d_in < d_out): standard interleaved layout slot[i*t] = a[i] for
    i in [0, d_in), where t = N/d_in.

    For tall (d_in > d_out): permuted layout
    slot[i*t + q*t'] = a[i + q*d] for i in [0, d), q in [0, alpha),
    where d = d_out, alpha = d_in/d, t = N/d, t' = N/(alpha*d).  This is the
    layout produced by the wide IRP's combine step (so an MLP up-then-down
    chain composes naturally).
    """
    if d_in == d_out:
        return encrypt_irp_input(ctx, encoder, sk, a, N=N, d=d_in,
                                 scale=scale, chain_index=chain_index)
    if not isinstance(a, np.ndarray):
        a = np.asarray(a, dtype=np.float64)
    if a.shape != (d_in,):
        raise ValueError(f"encrypt_irp_input_rect: a must have shape ({d_in},), "
                         f"got {a.shape}")
    if d_in < d_out:
        # Wide input: same as square at d = d_in
        return encrypt_irp_input(ctx, encoder, sk, a, N=N, d=d_in,
                                 scale=scale, chain_index=chain_index)
    # Tall input: permuted stride-t' layout
    d = d_out
    alpha = d_in // d
    _check_rect_dims(N, d, alpha)
    t = N // d
    t_prime = N // (alpha * d)
    slots = np.zeros(N, dtype=np.float64)
    for q in range(alpha):
        for i in range(d):
            slots[i * t + q * t_prime] = a[i + q * d]
    pt = encoder.encode_double_vector(ctx, slots, scale, chain_index)
    return sk.encrypt_symmetric(ctx, pt)


def decode_irp_output_rect(
    decoded_slots: np.ndarray,
    N: int,
    d_in: int,
    d_out: int,
) -> np.ndarray:
    """Extract the d_out valid output values from a rectangular-IRP result.

    For wide: output is at stride t' = N/d_out in *permuted* order.  We extract
    d_out slots and permute back to natural y[0], y[1], ..., y[d_out-1] order.

    For tall: output at stride t = N/d_out (same as square IRP for d=d_out).
    """
    arr = np.asarray(decoded_slots, dtype=np.float64)
    if d_in == d_out:
        return decode_irp_output(arr, N=N, d=d_in)
    if d_in < d_out:
        # Wide: permuted stride-t' output
        d = d_in
        alpha = d_out // d
        t_prime = N // d_out
        permuted = arr[::t_prime][:d_out]
        # Slot c_perm = c'*alpha + q holds y[c' + q*d].  Invert:
        natural = np.zeros(d_out, dtype=np.float64)
        for c_perm in range(d_out):
            c_prime = c_perm // alpha
            q = c_perm % alpha
            a = c_prime + q * d
            natural[a] = permuted[c_perm]
        return natural
    # Tall: stride t = N/d_out
    return decode_irp_output(arr, N=N, d=d_out)


def irp_matvec_rect(
    ctx,
    encoder,
    gk,
    ct_in,
    plaintexts: List,
    N: int,
    d_in: int,
    d_out: int,
    baby_steps: int = 1,
    sub_mask_pt=None,
    input_mask_pt=None,
):
    """Run the rectangular IRP ct-pt VMM.

    sub_mask_pt: per-sub-IRP output mask (from encode_irp_mask_rect).
    input_mask_pt: TALL ONLY -- mask to extract x_q from permuted input.
    Wide output is in permuted stride-t' layout; tall output is stride-t.
    """
    if d_in == d_out:
        return irp_matvec(ctx, encoder, gk, ct_in, plaintexts,
                          N=N, d=d_in, baby_steps=baby_steps,
                          mask_pt=sub_mask_pt)

    if sub_mask_pt is None:
        raise ValueError("irp_matvec_rect: sub_mask_pt required (use "
                         "encode_irp_mask_rect to create it)")

    if d_in < d_out:
        # Wide path
        d = d_in
        alpha = d_out // d
        t, t_prime, K_sq, K_total = _check_rect_dims(N, d, alpha)
        if len(plaintexts) != K_total:
            raise ValueError(f"irp_matvec_rect (wide): expected {K_total} plaintexts, "
                             f"got {len(plaintexts)}")
        out = None
        for q in range(alpha):
            sub_pts = plaintexts[q * K_sq:(q + 1) * K_sq]
            sub_q = irp_matvec(ctx, encoder, gk, ct_in, sub_pts,
                               N=N, d=d, baby_steps=baby_steps,
                               mask_pt=sub_mask_pt)
            if q > 0:
                # Right-rotate by q*t' (= left-rotate by (N - q*t') mod N) so
                # sub_q's slot c'*t lands at slot c'*t + q*t' in combined ciphertext.
                rot_amt = (N - q * t_prime) % N
                sub_q = phantom.rotate(ctx, sub_q, int(rot_amt), gk)
            if out is None:
                out = sub_q
            else:
                out = phantom.add(ctx, out, sub_q)
        return out

    # Tall path
    d = d_out
    alpha = d_in // d
    t, t_prime, K_sq, K_total = _check_rect_dims(N, d, alpha)
    if len(plaintexts) != K_total:
        raise ValueError(f"irp_matvec_rect (tall): expected {K_total} plaintexts, "
                         f"got {len(plaintexts)}")
    if input_mask_pt is None:
        raise ValueError("irp_matvec_rect (tall): input_mask_pt required for "
                         "the tall path (encode at the input ciphertext's chain_index)")
    out = None
    for q in range(alpha):
        # Extract x_q: rotate left by q*t' to bring slot a'*t + q*t' to slot a'*t,
        # then mask to keep only stride-t valid slots (zeros out other q's data).
        if q == 0:
            x_q_aligned = ct_in
        else:
            x_q_aligned = phantom.rotate(ctx, ct_in, int(q * t_prime), gk)
        # Mask using input_mask_pt (at the input ct's chain_index).
        nominal = x_q_aligned.scale()
        x_q_only = phantom.multiply_plain(ctx, x_q_aligned, input_mask_pt)
        x_q_only = phantom.rescale_to_next(ctx, x_q_only)
        x_q_only.set_scale(nominal)

        sub_pts = plaintexts[q * K_sq:(q + 1) * K_sq]
        sub_q = irp_matvec(ctx, encoder, gk, x_q_only, sub_pts,
                           N=N, d=d, baby_steps=baby_steps,
                           mask_pt=sub_mask_pt)
        if out is None:
            out = sub_q
        else:
            out = phantom.add(ctx, out, sub_q)
    return out


def encode_irp_diagonals_rect_pair_host(
    ctx,
    encoder,
    matrix_real: np.ndarray,
    matrix_imag: np.ndarray,
    N: int,
    d_in: int,
    d_out: int,
    scale: float,
    baby_steps: int = 1,
) -> List:
    """Pack two real rect matrices into one complex IRP plaintext set.

    `matrix_real` becomes the real part, `matrix_imag` the imag part of
    each slot. A subsequent irp_matvec_host call multiplies a real-only
    ciphertext by these complex plaintexts and accumulates a complex-
    valued result: re(result) = matrix_real @ x, im(result) = matrix_imag @ x.

    Halves the number of plaintexts vs encoding the two matrices
    separately (since both share the same diagonal indexing). Halves
    the matvec multiplication count for the pair.

    Both matrices must have the same (d_in, d_out) shape; alpha sub-blocks
    are folded by the rect encoder identically for both.
    """
    if matrix_real.shape != matrix_imag.shape:
        raise ValueError(f"encode_irp_diagonals_rect_pair_host: shape mismatch "
                         f"{matrix_real.shape} vs {matrix_imag.shape}")
    if matrix_real.shape != (d_in, d_out):
        raise ValueError(f"encode_irp_diagonals_rect_pair_host: matrices must be "
                         f"({d_in}, {d_out}), got {matrix_real.shape}")
    if d_in == d_out:
        raise ValueError("encode_irp_diagonals_rect_pair_host: d_in == d_out -- use "
                         "square pair variant instead")
    d = min(d_in, d_out)
    alpha = max(d_in, d_out) // d
    _check_rect_dims(N, d, alpha)
    t, K = _check_dims(N, d)
    M = baby_steps
    if not _is_pow2(M):
        raise ValueError(f"encode_irp_diagonals_rect_pair_host: baby_steps must be power of 2, got {M}")
    if K % M != 0:
        raise ValueError(f"encode_irp_diagonals_rect_pair_host: baby_steps={M} must divide K={K}")
    G = K // M

    plaintexts = []
    for q in range(alpha):
        slice_axis = slice(None, None)
        if d_in < d_out:
            W_q_real = matrix_real[:, q * d:(q + 1) * d]
            W_q_imag = matrix_imag[:, q * d:(q + 1) * d]
        else:
            W_q_real = matrix_real[q * d:(q + 1) * d, :]
            W_q_imag = matrix_imag[q * d:(q + 1) * d, :]
        slots_real = _build_irp_slots(W_q_real, N, d, t, K, M, G, dtype=complex)
        slots_imag = _build_irp_slots(W_q_imag, N, d, t, K, M, G, dtype=complex)
        for sr, si in zip(slots_real, slots_imag):
            # Combine: real part from sr (already real-only), imag from si.real.
            combined = sr.real + 1j * si.real
            # Pass numpy complex array directly — fast path via binding's
            # numpy overload (no Python iteration of 65K complex objects).
            plaintexts.append(phantom.encode_single_chain_plaintext(
                ctx, encoder, combined, scale))
    return plaintexts


def extract_real_imag_pair(ctx, encoder, galois_key, ct_complex,
                            slot_count, user_scale):
    """Split a complex ct (= ct_re + i·ct_im) into ct_re, ct_im via
    conjugation. Uses Phantom's auto-generated step=0 galois key.
    Costs 1 chain level (the *0.5 / -0.5i multiply)."""
    ct_conj = phantom.rotate(ctx, ct_complex, 0, galois_key)
    half_scpt = phantom.encode_single_chain_plaintext(
        ctx, encoder, [0.5 + 0j] * slot_count, user_scale)
    neg_half_i_scpt = phantom.encode_single_chain_plaintext(
        ctx, encoder, [-0.5j] * slot_count, user_scale)
    half_pt = phantom.expand_single_chain_to_full(
        ctx, half_scpt, ct_complex.chain_index())
    neg_half_i_pt = phantom.expand_single_chain_to_full(
        ctx, neg_half_i_scpt, ct_complex.chain_index())
    s = phantom.add(ctx, ct_complex, ct_conj)
    ct_re = phantom.multiply_plain(ctx, s, half_pt)
    ct_re = phantom.rescale_to_next(ctx, ct_re)
    ct_re.set_scale(user_scale)
    d = phantom.sub(ctx, ct_complex, ct_conj)
    ct_im = phantom.multiply_plain(ctx, d, neg_half_i_pt)
    ct_im = phantom.rescale_to_next(ctx, ct_im)
    ct_im.set_scale(user_scale)
    return ct_re, ct_im
