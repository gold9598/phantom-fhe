"""FHE-side dense token-major K/Q packing — Stage 2 of the dense-layout rewrite.

ADDITIVE. Does NOT modify the IRP path, kv_layout.py, irp.py, or any .cu.

This module produces ciphertext slot vectors whose geometry is BYTE-IDENTICAL
to the verified numpy oracle `blocks.kv_layout_dense` (Stage 1, commit 744e61f,
independently verified). The oracle is the trusted source-faithful spec; every
packer here mirrors its slot formula exactly so the FHE QK^T can be validated
against `kv_layout_dense.dense_qkt` on the same Q/K.

Layout (token-MAJOR, j-innermost; D = d_total = n_heads*d_head, H = d_head):

  q_slot[tok_local*D + h*H + j] = Q[h, j]   (query pre-broadcast over P frames)
  k_slot[tok_local*D + h*H + j] = K[tok_abs, h, j]   (GQA-expanded, tail 0)

Shard params (oracle-exact, kv_layout_dense.positions_per_ct / pack_kv_dense_shards):
  P        = min(next_pow2(real_nt), num_slots // D)
  n_shards = ceil(real_nt / P)
  slot len = P * D   (== num_slots when P == num_slots//D, the LLaMA case)

The query pre-broadcast across token frames is FREE: phantom.bsgs_matmul_preencoded
takes / returns the "replicated-block" layout (period = d_pad). With d_pad == D,
the BSGS Wq output slot[k*D + r] = Q_flat[r] for every period k in [0, P) — which
is exactly q_slot[tok_local*D + h*H + j] = Q[h,j]  (== oracle pack_q_dense).
"""

import math

import numpy as np
import pyPhantom as phantom

from blocks import kv_layout_dense as _oracle


def positions_per_ct(real_nt: int, num_slots: int, d_total: int) -> int:
    """P = min(next_pow2(real_nt), num_slots // d_total). Oracle-exact."""
    nt_pad = _oracle.next_pow2(real_nt)
    return _oracle.positions_per_ct(nt_pad, num_slots, d_total)


def n_shards_for(real_nt: int, P: int) -> int:
    """n_shards = ceil(real_nt / P). Oracle-exact (kv_layout_dense:283)."""
    return math.ceil(real_nt / P)


# ---------------------------------------------------------------------------
# Wq -> BSGS diagonals (the attention_forward / bsgs_test pattern; d_pad == D)
# ---------------------------------------------------------------------------

def bsgs_wq_diags_dense(ctx, encoder, Wq_baked: np.ndarray,
                        d_total: int, baby_steps: int, scale: float):
    """Pre-encode Wq_baked (R_P already applied) into BSGS diagonals.

    Wq_baked: (D_TOTAL, D_MODEL) == (num_rows, num_cols) == (d_total, d_total)
    for LLaMA (d_model == d_total == 4096). phantom.bsgs_matmul_preencoded
    then computes q = Wq_baked @ x with x in replicated-block (period d_pad)
    layout; output q is replicated-block period d_pad with q[0..num_rows) in
    the first num_rows slots of every period (== dense pre-broadcast Q).
    """
    num_rows, num_cols = Wq_baked.shape
    if d_total < max(num_rows, num_cols):
        raise ValueError(
            f"bsgs_wq_diags_dense: d_pad ({d_total}) must be >= "
            f"max(num_rows={num_rows}, num_cols={num_cols})")
    matrix_flat = np.ascontiguousarray(Wq_baked, dtype=np.float64).ravel().tolist()
    return phantom.pre_encode_bsgs_diagonals(
        ctx, encoder, matrix_flat, num_rows, num_cols,
        d_total, baby_steps, scale)


def encrypt_x_replicated_block(ctx, encoder, sk, xn_query: np.ndarray,
                               d_total: int, num_slots: int,
                               scale: float, chain_index: int):
    """Encrypt the rmsnormed query vector in replicated-block layout.

    slot[k*d_total + j] = xn_query[j]  for j in [0, len(xn_query)),
    every period k in [0, num_slots // d_total); pad slots zero.

    This is the EXACT input layout phantom.bsgs_matmul_preencoded consumes
    (see blocks/bsgs_test.py:50-57). bsgs_wq_diags_dense @ this -> dense Q.
    """
    x = np.asarray(xn_query, dtype=np.float64).ravel()
    if x.shape[0] > d_total:
        raise ValueError(
            f"encrypt_x_replicated_block: xn len {x.shape[0]} > d_total {d_total}")
    periods = num_slots // d_total
    slots = np.zeros(num_slots, dtype=np.float64)
    for k in range(periods):
        slots[k * d_total:k * d_total + x.shape[0]] = x
    pt = encoder.encode_double_vector(ctx, slots, scale, chain_index)
    return sk.encrypt_symmetric(ctx, pt)


# ---------------------------------------------------------------------------
# K -> dense token-major shards (mirrors oracle pack_kv_dense_shards exactly)
# ---------------------------------------------------------------------------

def pack_k_dense_shards_slots(K_full_h: np.ndarray, real_nt: int, P: int,
                              n_heads: int, d_head: int,
                              num_slots: int) -> list:
    """Build the per-shard K slot vectors (numpy) for the dense token-major
    layout. Each shard has length num_slots (== P*D for the LLaMA case).

      k_slot[tok_local*D + h*H + j] = K_full_h[shard*P + tok_local, h, j]

    K_full_h is GQA-EXPANDED [real_nt, n_heads, d_head] (np.repeat already
    applied by the caller, matching kv_layout_dense / encrypt_layer_inputs).
    Tail tok >= real_nt (last partial shard) stays exact 0.0.

    The slot formula is byte-identical to kv_layout_dense.pack_kv_shard with
    n_heads == n_kv_heads_in (i.e. K already expanded), padded out to
    num_slots instead of P*D (extra slots, if any, are zero and unused).
    """
    K = np.asarray(K_full_h, dtype=np.float64)
    if K.ndim != 3 or K.shape[0] != real_nt:
        raise ValueError(
            f"pack_k_dense_shards_slots: expected [real_nt={real_nt}, "
            f"n_heads, d_head], got {K.shape}")
    D = n_heads * d_head
    n_shards = n_shards_for(real_nt, P)
    shards = []
    for b in range(n_shards):
        slots = np.zeros(num_slots, dtype=np.float64)
        for tok_local in range(P):
            tok_abs = b * P + tok_local
            if tok_abs >= real_nt:
                break
            frame = tok_local * D
            for h in range(n_heads):
                hb = frame + h * d_head
                slots[hb:hb + d_head] = K[tok_abs, h, :]
        shards.append(slots)
    return shards


def encrypt_k_dense_shards(ctx, encoder, sk, K_full_h: np.ndarray,
                           real_nt: int, P: int, n_heads: int, d_head: int,
                           num_slots: int, scale: float, chain_index: int):
    """Encrypt the dense token-major K shards -> list of ciphertexts."""
    slot_shards = pack_k_dense_shards_slots(
        K_full_h, real_nt, P, n_heads, d_head, num_slots)
    return [
        sk.encrypt_symmetric(ctx, encoder.encode_double_vector(
            ctx, ks, scale, chain_index))
        for ks in slot_shards
    ]


# ---------------------------------------------------------------------------
# Fused scale*mask + per-head C subtraction plaintexts (token-major base slots)
# ---------------------------------------------------------------------------

def dense_scale_mask_slots(num_slots: int, d_head: int, d_total: int,
                           shard_tok_start: int, P: int, real_nt: int,
                           scale_value: float) -> np.ndarray:
    """Mask*scale slot vector for one shard's scores (token-major base slots).

    After compute_qkt's inner_sum(d_head) the score for (tok_local, h) lives
    at the d_head-aligned base slot tok_local*D + h*H (mid-block slots hold
    partial-junk). Keep `scale_value` (== 1/sqrt(d_head)) ONLY at base slots
    of real tokens (shard_tok_start + tok_local < real_nt); zero everything
    else. This both rescales and applies the pad-token-zero + head mask in
    one fused multiply_plain (mirrors the IRP path's stage-A fuse).
    """
    n_heads = d_total // d_head
    slots = np.zeros(num_slots, dtype=np.float64)
    for tok_local in range(P):
        if shard_tok_start + tok_local >= real_nt:
            break
        frame = tok_local * d_total
        for h in range(n_heads):
            idx = frame + h * d_head
            if idx < num_slots:
                slots[idx] = scale_value
    return slots


def dense_per_head_sub_slots(num_slots: int, d_head: int, d_total: int,
                             shard_tok_start: int, P: int, real_nt: int,
                             c_per_head) -> np.ndarray:
    """Per-head centering slot vector for one shard (token-major base slots).

    Place c_per_head[h] at base slot tok_local*D + h*H for real tokens only;
    zero elsewhere so sub_plain leaves pad/junk slots untouched-as-zero.
    c_per_head: (n_heads,) — the SAME per-head softmax shift the IRP path's
    qkt_irp_per_head_sub_plaintext applies (computed over real keys).
    """
    n_heads = d_total // d_head
    c = np.asarray(c_per_head, dtype=np.float64).ravel()
    if c.shape[0] != n_heads:
        raise ValueError(
            f"dense_per_head_sub_slots: c_per_head len {c.shape[0]} "
            f"!= n_heads {n_heads}")
    slots = np.zeros(num_slots, dtype=np.float64)
    for tok_local in range(P):
        if shard_tok_start + tok_local >= real_nt:
            break
        frame = tok_local * d_total
        for h in range(n_heads):
            idx = frame + h * d_head
            if idx < num_slots:
                slots[idx] = c[h]
    return slots


# ---------------------------------------------------------------------------
# Stage 3: strict 0/1 base-slot mask (the poly(0) trap fix)
# ---------------------------------------------------------------------------

def dense_real_base_mask_slots(num_slots: int, d_head: int, d_total: int,
                               shard_tok_start: int, P: int,
                               real_nt: int,
                               keep_value: float = 1.0) -> np.ndarray:
    """Strict mask: `keep_value` (default 1.0) ONLY at base slots
    tok_local*D + h*H of REAL tokens (shard_tok_start + tok_local < real_nt),
    0.0 everywhere else.

    `keep_value` defaults to 1.0 (a pure 0/1 strict mask). Pass the IRP
    path's `softmax_safety_scale` to additionally fold the post-exp safety
    rescale into this same multiply_plain (exactly like the IRP path's
    stage-C qkt_irp_mask_scale_plaintext, which masks AND scales by
    safety_scale in one fused op). Softmax weights are scale-invariant
    ((s·e)/Σ(s·e) == e/Σe), so this leaves the final weights unchanged while
    keeping the Goldschmidt denominator a = s·Σe inside the convergence
    window (0,2) for peaky heads whose un-scaled per-head Σexp exceeds 2.

    THE poly(0) trap fix. After compute_qkt + scale*mask, padded-token base
    slots (tok >= real_nt) and ALL mid-block (j>0) slots are exact-0. But
    ps_exp_init evaluates a polynomial whose value at 0 is poly(0) != 0
    (~0.449). So after ps_exp_init + damped squarings EVERY zero slot holds a
    nonzero junk value. If the per-head finalize sum-reduce
    (sum_reduce_stride over the P token frames at stride d_total) runs over
    those slots it adds (nt_pad - real_nt) bogus poly(0) terms per head and
    pollutes every per-head denominator (the recurring softmax-layout trap,
    the single biggest historical bug in this codebase).

    Applying this strict-0/1 mask (multiply_plain) AFTER ps_exp_init + the
    damped squarings — exactly mirroring the IRP path's stage-C
    qkt_irp_mask_scale_plaintext re-mask — hard-zeros every padded / junk
    slot so ONLY the real_nt legitimate exp values enter the per-head sum.
    Geometry is byte-identical to dense_scale_mask_slots with scale_value=1.0
    (same base-slot, same real-token guard).
    """
    n_heads = d_total // d_head
    slots = np.zeros(num_slots, dtype=np.float64)
    for tok_local in range(P):
        if shard_tok_start + tok_local >= real_nt:
            break
        frame = tok_local * d_total
        for h in range(n_heads):
            idx = frame + h * d_head
            if idx < num_slots:
                slots[idx] = keep_value
    return slots


# ---------------------------------------------------------------------------
# Readout: dense scores ct list -> scores[real_nt, n_heads]
# ---------------------------------------------------------------------------

def unpack_dense_scores(score_slot_shards: list, real_nt: int, P: int,
                        n_heads: int, d_head: int) -> np.ndarray:
    """Read scores[real_nt, n_heads] from decrypted per-shard score slot
    vectors at the token-major base slots tok_local*D + h*H.

    Mirrors kv_layout_dense.unpack_scores_shard but across all shards and
    AFTER the scale*mask fuse (so base slots already hold the final scaled,
    pad-masked, NOT-yet-centered score; centering is applied separately by
    the caller via sub_plain before decrypt if comparing post-sub)."""
    D = n_heads * d_head
    scores = np.zeros((real_nt, n_heads), dtype=np.float64)
    for b, sv in enumerate(score_slot_shards):
        sv = np.asarray(sv, dtype=np.float64)
        for tok_local in range(P):
            tok_abs = b * P + tok_local
            if tok_abs >= real_nt:
                break
            for h in range(n_heads):
                scores[tok_abs, h] = sv[tok_local * D + h * d_head]
    return scores
