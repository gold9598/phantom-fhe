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


# Quantization scale for IRP WEIGHT SingleChainPlaintexts. The signed-centered
# IRP diagonal coefficients are stored as int16 at this scale; the full message
# scale (engine user_scale, 2^40) is restored at expand time by the per-tower
# scale_2 = round(scale / coeff_scale) multiply in the C++ kernel.
#
# Scale choice: int16 holds +/-32767. The coeffs must use enough of that range
# that integer ROUNDING is negligible. Empirically (random N(0,0.02) weights,
# the worst case): at 2^16 coeffs peak ~22 -> rounding loses ~7%/coeff -> output
# rel-RMS ~5.6e-2 (catastrophic). At 2^24 coeffs peak ~5600 -> rel-RMS ~2e-4
# (well below the ~4.2e-3 baseline) with ~6x int16 headroom against per-weight
# outliers. 2^24 is the lossless sweet spot that still fits int16 (4x smaller
# cache than int64). Larger scales (>=2^28) overflow int16 and fall back to
# int64 storage in the encoder (lossless but no size win).
#
# Only the IRP weight encoders below pass this; every other SCP (rmsnorm gammas,
# merge-bootstrap constants, masks, the complex-bridge half/neg-half constants)
# encodes at full scale, where coeff_scale == scale => scale_2 == 1 (unchanged).
IRP_COEFF_SCALE = 2.0 ** 40   # quant-64bit control: coeff_scale==user_scale
# => scale_2==1 (no in-kernel rescale); coeffs ~2^40 overflow int16/int32 so the
# encoder stores them as int64 — i.e. the IRP pipeline UNQUANTIZED. This is the
# precision-sweep anchor (same IRP/fold/bridgeless path as quant-8/16/32, only
# the SCP dtype differs), NOT the old dense `baseline` branch.


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
    return [phantom.encode_single_chain_plaintext(
                ctx, encoder, s, scale, IRP_COEFF_SCALE)
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


# ===========================================================================
# Complex-folded SQUARE IRP (Phase-1 optimization)
# ===========================================================================
#
# The square IRP plaintexts waste the imaginary half of every complex SCP
# (real weight cast to complex with imag = 0).  For a square d x d matvec
# y = x @ M with REAL encrypted x, fold the OUTPUT columns into the imaginary
# part:
#
#     M_fold[k, c] = M[k, c] + 1j * M[k, c + d/2]    for c in [0, d/2)
#
# M_fold is a TALL rect (d_in = d > d_out = d/2).  Because x is real and the
# IRP rotations + multiply-accumulate + reduce are all C-linear, the folded
# matvec produces a complex result whose stride-t' slots carry
#
#     real(y_fold[c]) = (x @ M)[c]
#     imag(y_fold[c]) = (x @ M)[c + d/2]      for c in [0, d/2).
#
# This rides the EXISTING tall-rect machinery: the d x (d/2) tall matrix
# decomposes into alpha = 2 square sub-IRPs of dim d/2, so the total SCP
# count is alpha * (d/2)^2 / N = d^2 / (2N) = K_square / 2.  HALVED disk
# cache, encode time, and ct*pt multiplies.
#
# Recovery (length-d y) costs ONE conjugation (the step-0 galois key) plus
# one chain level (the 0.5 / -0.5i multiply), exactly like merge_bootstrap.

def encode_irp_diagonals_folded_host(
    ctx,
    encoder,
    matrix: np.ndarray,
    N: int,
    d: int,
    scale: float,
    baby_steps: int = 1,
) -> List:
    """Encode a square (d x d) REAL weight into K/2 complex IRP SCPs.

    Folds the output columns: M_fold[:, c] = M[:, c] + 1j*M[:, c + d/2] for
    c in [0, d/2).  M_fold is a (d, d/2) tall rect encoded via the existing
    tall-rect host machinery (alpha = 2 square sub-IRPs of dim d/2).

    Returns alpha * (d/2)^2 / N = d^2/(2N) SCPs (HALF of the square count).
    Consume with `irp_matvec_folded_host` + `extract_real_imag_pair`.
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (d, d):
        raise ValueError(f"encode_irp_diagonals_folded_host: matrix must be "
                         f"({d}, {d}), got {matrix.shape}")
    if d % 2 != 0:
        raise ValueError(f"encode_irp_diagonals_folded_host: d must be even, got {d}")
    d_out = d // 2
    # Square sub-IRP at dim d_out must satisfy d_out^2 % N == 0.
    if (d_out * d_out) % N != 0:
        raise ValueError(f"encode_irp_diagonals_folded_host: (d/2)^2={d_out*d_out} "
                         f"must be divisible by N={N} (fold needs K_sub = (d/2)^2/N >= 1)")
    M_fold = matrix[:, :d_out] + 1j * matrix[:, d_out:]
    return encode_irp_diagonals_rect_host(
        ctx, encoder, M_fold, N=N, d_in=d, d_out=d_out,
        scale=scale, baby_steps=baby_steps,
    )


def irp_matvec_folded_host(
    ctx,
    encoder,
    gk,
    ct_in,
    plaintexts: List,
    N: int,
    d: int,
    baby_steps: int = 1,
    sub_mask_pt=None,
    input_mask_pt=None,
) -> "phantom.ciphertext":
    """Run the complex-folded square IRP matvec (tall-rect under the hood).

    `ct_in` must be encrypted in the TALL-rect permuted layout for
    (d_in=d, d_out=d/2) -- use `encrypt_irp_input_rect(..., d_in=d, d_out=d/2)`.
    `plaintexts` come from `encode_irp_diagonals_folded_host`.  Returns a
    COMPLEX ciphertext in stride-t' (t' = N/(d/2*... )) layout: slot c*t holds
    (x@M)[c] + 1j*(x@M)[c+d/2] for c in [0, d/2).  Split with
    `extract_real_imag_pair` -> ct_re = (x@M)[:d/2], ct_im = (x@M)[d/2:].

    sub_mask_pt / input_mask_pt: tall-rect masks (encode at dim d/2).
    """
    if d % 2 != 0:
        raise ValueError(f"irp_matvec_folded_host: d must be even, got {d}")
    d_out = d // 2
    return irp_matvec_rect_host(
        ctx, encoder, gk, ct_in, plaintexts,
        N=N, d_in=d, d_out=d_out, baby_steps=baby_steps,
        sub_mask_pt=sub_mask_pt, input_mask_pt=input_mask_pt,
    )


# ===========================================================================
# Complex-folded RECTANGULAR IRP (Phase-1b optimization)
# ===========================================================================
#
# Generalizes the square output-fold to RECT weights (the MLP gate/up/down,
# K = alpha * d^2/N each).  For a (d_in, d_out) matvec y = x @ M with REAL
# encrypted x, fold the OUTPUT columns into the imaginary part:
#
#     M_fold[k, c] = M[k, c] + 1j * M[k, c + d_out/2]    for c in [0, d_out/2)
#
# M_fold is (d_in, d_out/2).  Halving d_out preserves the orientation:
#   wide  (d_in < d_out)  stays wide   (d_in < d_out/2 once d_out > 2*d_in)
#   tall  (d_in > d_out)  stays tall and gets *more* tall (alpha doubles)
# The fold rides the EXISTING rect machinery: the rect rotations (preprocess,
# babies, giants, reduce, combine/align) and the multiply-accumulate are all
# C-linear, and x is real, so each output slot carries
#   real(y_fold[c]) = (x @ M)[c]   ,   imag(y_fold[c]) = (x @ M)[c + d_out/2].
# SCP count = alpha_fold * (d_min_fold)^2 / N = (full rect K) / 2 -- HALVED.
#
# Recovery: split real/imag (extract_real_imag_pair, +1 conjugation + 1 level),
# then read each half with the rect output layout (wide: permuted stride-t';
# tall: stride-t) and concatenate -> length-d_out y.

def encode_irp_diagonals_rect_folded_host(
    ctx,
    encoder,
    matrix: np.ndarray,
    N: int,
    d_in: int,
    d_out: int,
    scale: float,
    baby_steps: int = 1,
) -> List:
    """Encode a (d_in, d_out) REAL rect weight into HALF the rect SCPs.

    Folds the output columns: M_fold[:, c] = M[:, c] + 1j*M[:, c + d_out/2]
    for c in [0, d_out/2).  M_fold is (d_in, d_out/2), encoded via the existing
    rect host machinery (orientation preserved).  Returns
    (full rect K)/2 = alpha_fold * min(d_in, d_out/2)^2 / N complex SCPs.

    Consume with `irp_matvec_rect_folded_host` + `extract_real_imag_pair`.
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (d_in, d_out):
        raise ValueError(f"encode_irp_diagonals_rect_folded_host: matrix must be "
                         f"({d_in}, {d_out}), got {matrix.shape}")
    if d_out % 2 != 0:
        raise ValueError(f"encode_irp_diagonals_rect_folded_host: d_out must be "
                         f"even, got {d_out}")
    d_out_fold = d_out // 2
    if d_in == d_out_fold:
        raise ValueError("encode_irp_diagonals_rect_folded_host: folded d_out == "
                         "d_in (square) -- use encode_irp_diagonals_folded_host")
    d_fold = min(d_in, d_out_fold)
    if (d_fold * d_fold) % N != 0:
        raise ValueError(f"encode_irp_diagonals_rect_folded_host: folded sub-IRP "
                         f"dim^2={d_fold * d_fold} must be divisible by N={N}")
    M_fold = matrix[:, :d_out_fold] + 1j * matrix[:, d_out_fold:]
    return encode_irp_diagonals_rect_host(
        ctx, encoder, M_fold, N=N, d_in=d_in, d_out=d_out_fold,
        scale=scale, baby_steps=baby_steps,
    )


def irp_matvec_rect_folded_host(
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
) -> "phantom.ciphertext":
    """Run the complex-folded RECT IRP matvec (rect machinery under the hood).

    `ct_in` must be encrypted in the rect input layout for (d_in, d_out/2)
    -- use `encrypt_irp_input_rect(..., d_in=d_in, d_out=d_out//2)`.
    `plaintexts` come from `encode_irp_diagonals_rect_folded_host`.  Returns a
    COMPLEX ciphertext in the folded rect output layout (wide: permuted
    stride-t' for d_out/2; tall: stride-t = N/(d_out/2)).  Split with
    `extract_real_imag_pair` then read each half with the folded rect layout:
    real -> (x@M)[:d_out/2], imag -> (x@M)[d_out/2:].

    sub_mask_pt / input_mask_pt: rect masks encoded at the folded dims.
    """
    if d_out % 2 != 0:
        raise ValueError(f"irp_matvec_rect_folded_host: d_out must be even, "
                         f"got {d_out}")
    d_out_fold = d_out // 2
    return irp_matvec_rect_host(
        ctx, encoder, gk, ct_in, plaintexts,
        N=N, d_in=d_in, d_out=d_out_fold, baby_steps=baby_steps,
        sub_mask_pt=sub_mask_pt, input_mask_pt=input_mask_pt,
    )


# ===========================================================================
# In-FHE interleave-recombine for the complex output-fold (Phase-1.5)
# ===========================================================================
#
# `extract_real_imag_pair` on a folded RECT matvec output gives two cts that
# BOTH live in the same folded-rect output layout (wide: permuted stride-t' for
# d_out/2; tall: stride-t = N/(d_out/2)).  ct_re holds y[:d_out/2], ct_im holds
# y[d_out/2:], each at valid stride `t_fold` = N / d_out_fold.
#
# Instead of decrypting and concatenating in numpy, recombine the two halves
# IN FHE by interleaving them onto a finer stride `t_fold/2`:
#
#     recombined = ct_re + rotate(ct_im, right by t_fold/2)
#
# After this, valid data sits at stride `t_fold/2`; at the c-th valid position
# (slot c * t_fold/2) the value alternates re/im:
#     slot (2j)   * (t_fold/2)  = y[:d_out/2] half, j-th valid output of ct_re
#     slot (2j+1) * (t_fold/2)  = y[d_out/2:] half, j-th valid output of ct_im
# i.e. the d_out outputs appear INTERLEAVED in the folded-rect halves' order.
# Cost: exactly 1 galois rotation + 1 add, 0 chain levels (no multiply/rescale).
#
# The interleave permutation is INVARIANT under elementwise ops (silu, mult):
# silu(interleaved) == interleaved(silu).  A downstream matvec ABSORBS the
# permutation for free by reordering its weight ROWS at encode time to match
# the interleaved input order (see `interleave_output_order`).

def interleave_recombine(ctx, gk, ct_re, ct_im, N: int, d_out_fold: int):
    """Recombine the two folded-rect halves onto stride t_fold/2 IN FHE.

    ct_re, ct_im come from `extract_real_imag_pair` on a folded matvec whose
    folded output dim is `d_out_fold` (= d_out/2).  Each half has valid data
    at stride t_fold = N / d_out_fold.  Returns one ciphertext whose valid data
    sits at stride t_fold/2, alternating ct_re / ct_im per the folded layout.

    Cost: 1 rotate + 1 add, 0 chain levels.  The result keeps ct_re's scale and
    chain index (ct_im must share them; they do, both from extract_real_imag_pair).
    """
    if (N % d_out_fold) != 0:
        raise ValueError(f"interleave_recombine: N={N} not divisible by "
                         f"d_out_fold={d_out_fold}")
    t_fold = N // d_out_fold
    if t_fold % 2 != 0:
        raise ValueError(f"interleave_recombine: t_fold={t_fold} must be even "
                         f"(need a half-stride slot to interleave into)")
    half = t_fold // 2
    # Right-rotate ct_im by `half` = left-rotate by (N - half): moves the value
    # at slot c*t_fold to slot c*t_fold + half (the gap between ct_re slots).
    ct_im_shift = phantom.rotate(ctx, ct_im, int((N - half) % N), gk)
    return phantom.add(ctx, ct_re, ct_im_shift)


def _interleave_recombine_slot_to_y(N: int, d_in: int, d_out: int) -> dict:
    """Map {physical slot -> natural y-index} that `interleave_recombine`
    produces for a FOLDED matvec of a (d_in, d_out) weight.

    ct_re carries y[:d_out/2], ct_im carries y[d_out/2:], each in the FOLDED
    rect output layout at stride t_fold = N/(d_out/2).  interleave_recombine
    shifts ct_im by +t_fold/2 so:
        slot c*t_fold            -> ct_re's c-th valid output (y[:d_out/2])
        slot c*t_fold + t_fold/2 -> ct_im's c-th valid output (y[d_out/2:])
    The c-th valid output's natural local index is given by the folded rect
    output layout (wide: permuted stride-t'; tall/square: natural stride-t).
    """
    if d_out % 2 != 0:
        raise ValueError(f"interleave slot map: d_out must be even, got {d_out}")
    d_out_fold = d_out // 2
    t_fold = N // d_out_fold
    half = t_fold // 2
    if d_in < d_out_fold:
        # Wide folded: half[c_perm] holds natural local index c'+q*d.
        d = d_in
        alpha = d_out_fold // d
        half_local = np.zeros(d_out_fold, dtype=np.int64)
        for c_perm in range(d_out_fold):
            half_local[c_perm] = (c_perm // alpha) + (c_perm % alpha) * d
    else:
        # Tall / square folded: half in natural stride-t order.
        half_local = np.arange(d_out_fold, dtype=np.int64)
    slot_to_y = {}
    for c in range(d_out_fold):
        slot_to_y[c * t_fold] = int(half_local[c])               # re -> y[:d/2]
        slot_to_y[c * t_fold + half] = int(half_local[c]) + d_out_fold  # im
    return slot_to_y


def _rect_consume_slot_to_input(N: int, d_in: int, d_out: int) -> dict:
    """Map {physical slot -> natural input index a[k]} that a downstream rect
    matvec READS, mirroring `encrypt_irp_input_rect`.

    Wide / square: slot[i*t] = a[i], t = N/d_in.
    Tall: slot[i*t + q*t'] = a[i + q*d], d = d_out, alpha = d_in/d_out,
          t = N/d, t' = N/(alpha*d) = N/d_in.
    """
    slot_to_input = {}
    if d_in <= d_out:
        t = N // d_in
        for i in range(d_in):
            slot_to_input[i * t] = i
    else:
        d = d_out
        alpha = d_in // d
        t = N // d
        t_prime = N // (alpha * d)
        for q in range(alpha):
            for i in range(d):
                slot_to_input[i * t + q * t_prime] = i + q * d
    return slot_to_input


def interleave_output_order(N: int, d_in: int, d_out: int,
                            down_d_in: int = None,
                            down_d_out: int = None) -> "np.ndarray":
    """Row permutation a downstream rect matvec uses to ABSORB the interleave.

    For a FOLDED matvec of a (d_in, d_out) weight, `interleave_recombine`
    yields a ciphertext whose slots carry the d_out natural outputs in a
    layout described by `_interleave_recombine_slot_to_y`.  A downstream
    UNFOLDED rect matvec (down_d_in, down_d_out) reads that ciphertext as its
    input per `_rect_consume_slot_to_input`.

    Returns `order` (length down_d_in == d_out): the downstream matvec's input
    index k carries natural y[order[k]].  Reorder the downstream weight ROWS to
    `W_perm = W[order, :]` (numpy-time, free) so that
        interleaved_input @ W_perm == natural_y @ W,
    outputting NATURAL order with NO numpy un-permute.

    `down_d_in`/`down_d_out` default to (d_out, ...) for a square-ish consumer;
    pass the real downstream dims when the consumer is rect (e.g. Wdown tall).
    """
    if down_d_in is None:
        down_d_in = d_out
    if down_d_in != d_out:
        raise ValueError(f"interleave_output_order: downstream input dim "
                         f"{down_d_in} must equal folded matvec output {d_out}")
    if down_d_out is None:
        down_d_out = d_out  # square placeholder
    slot_to_y = _interleave_recombine_slot_to_y(N, d_in, d_out)
    slot_to_input = _rect_consume_slot_to_input(N, down_d_in, down_d_out)
    order = np.full(down_d_in, -1, dtype=np.int64)
    for slot, k in slot_to_input.items():
        if slot in slot_to_y:
            order[k] = slot_to_y[slot]
    if (order < 0).any():
        missing = int((order < 0).sum())
        raise ValueError(f"interleave_output_order: {missing} downstream input "
                         f"slots have no interleaved data (layout mismatch)")
    if len(set(order.tolist())) != down_d_in:
        raise ValueError("interleave_output_order: result is not a permutation")
    return order
