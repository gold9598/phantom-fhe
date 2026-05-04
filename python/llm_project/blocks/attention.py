"""
Attention orchestration ported from src/attention.cu to Python.

C++ primitives (ct x ct + lazy relin/rescale): phantom.compute_qkt, phantom.score_times_v.
Everything else here is pure orchestration over those CUDA primitives.

encode_scale convention: callers pass encode_scale (default = ct.scale()) as
the plaintext encode scale.  For BITS-uniform chains every middle prime is
~2^40 = SCALE, so set_scale(nominal) snaps the residue back exactly.
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

# Support both import styles:
#  - top-level (`from attention import ...` after adding blocks/ to sys.path,
#    used by the per-block regression tests)
#  - package-qualified (`from blocks.attention import ...`, used by headlines)
try:
    from blocks.linear import inner_sum_required_steps, replicate_required_steps
    from blocks.softmax import softmax_damping_schedule, softmax_required_steps
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from linear import inner_sum_required_steps, replicate_required_steps
    from softmax import softmax_damping_schedule, softmax_required_steps


# ---------------------------------------------------------------------------
# Shape / step helpers
# ---------------------------------------------------------------------------

def _is_pow2(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def qkt_required_steps(d_head: int):
    """Galois steps for QK^T inner-sum: powers of 2 in [1, d_head)."""
    return inner_sum_required_steps(d_head)


def score_v_required_steps(d_head: int, d_total: int, positions_per_ct: int):
    """Steps for score_times_v: in-block broadcast (negative) + cross-position
    accumulation (positive)."""
    if not _is_pow2(d_head):
        raise ValueError("score_v_required_steps: d_head must be a power of 2")
    if not _is_pow2(positions_per_ct):
        raise ValueError("score_v_required_steps: positions_per_ct must be a power of 2")
    steps = []
    # Broadcast within d_head blocks: negative strides d_head/2, d_head/4, ..., 1.
    bstride = d_head // 2
    while bstride >= 1:
        steps.append(-int(bstride))
        if bstride == 1:
            break
        bstride >>= 1
    # Accumulate across packed positions: d_total, 2*d_total, ..., (positions_per_ct/2)*d_total.
    max_accumulate = positions_per_ct * d_total
    astride = d_total
    while astride < max_accumulate:
        steps.append(int(astride))
        astride <<= 1
    return steps


def broadcast_required_steps(block_size: int):
    """Steps for broadcast_within_blocks: -block_size/2, ..., -2, -1."""
    if not _is_pow2(block_size):
        raise ValueError("broadcast_required_steps: block_size must be a power of 2")
    steps = []
    bstride = block_size // 2
    while bstride >= 1:
        steps.append(-int(bstride))
        if bstride == 1:
            break
        bstride >>= 1
    return steps


def sdpa_required_steps(d_head: int, d_total: int, num_tokens: int, slot_count: int):
    """Combined Galois steps for full SDPA: QK^T | softmax | score*V."""
    steps = []
    steps.extend(qkt_required_steps(d_head))
    # Softmax sum_reduce uses cyclic-wrap count = slot_count/d_total.
    steps.extend(softmax_required_steps(slot_count // d_total, d_total))
    steps.extend(score_v_required_steps(d_head, d_total, num_tokens))
    steps = sorted(set(int(s) for s in steps))
    return steps


def attention_forward_required_steps(
    baby_steps: int,
    d_head: int,
    d_total: int,
    num_tokens: int,
    slot_count: int,
):
    """Union of BSGS | SDPA | post-SDPA replicate(period=d_total) steps."""
    steps = []
    steps.extend(phantom.bsgs_required_steps(baby_steps))
    steps.extend(sdpa_required_steps(d_head, d_total, num_tokens, slot_count))
    # Post-SDPA replicate (period d_total). C++ conservatively covers up to
    # N/2 at logN=16; we mirror that default but accept slot_count if larger.
    rep_num_slots = max(slot_count, 1 << 15)
    steps.extend(replicate_required_steps(d_total, rep_num_slots))
    steps = sorted(set(int(s) for s in steps))
    return steps


# ---------------------------------------------------------------------------
# Internal helpers (not part of the public API)
# ---------------------------------------------------------------------------

def _head_stride_mask(num_slots, d_head, d_total, num_tokens, value=1.0):
    """Build a numpy slot vector with `value` at each head-stride position.

    Sets slot[t * d_total + h * d_head] = value for t in [0, num_tokens),
    h in [0, n_heads); all other slots are zero.
    """
    n_heads = d_total // d_head
    slots = np.zeros(num_slots, dtype=np.float64)
    for t in range(num_tokens):
        for h in range(n_heads):
            idx = t * d_total + h * d_head
            if idx < num_slots:
                slots[idx] = value
    return slots


def _encode_mul_rescale_snap(ctx, encoder, ct, slots, encode_scale, nominal=None):
    """Encode a slot vector, multiply-plain into ct, rescale, and snap scale.

    This is the recurring idiom: encode_double_vector -> multiply_plain ->
    rescale_to_next -> set_scale(nominal).  Returns the result ciphertext.
    """
    if nominal is None:
        nominal = ct.scale()
    pt = encoder.encode_double_vector(
        ctx, slots.tolist(), encode_scale, ct.chain_index(),
    )
    result = phantom.multiply_plain(ctx, ct, pt)
    result = phantom.rescale_to_next(ctx, result)
    result.set_scale(nominal)
    return result


# ---------------------------------------------------------------------------
# Plaintext mask builders
# ---------------------------------------------------------------------------

def score_mask_plaintext(
    ctx, encoder, d_head: int, d_total: int, positions_per_ct: int,
    chain_index: int, scale: float,
):
    """Plaintext with 1.0 at each head-stride position, 0 elsewhere."""
    if d_head == 0 or d_total == 0:
        raise ValueError("score_mask_plaintext: dimensions must be non-zero")
    if d_total % d_head != 0:
        raise ValueError("score_mask_plaintext: d_total must be a multiple of d_head")
    num_slots = encoder.slot_count()
    slots = _head_stride_mask(num_slots, d_head, d_total, positions_per_ct)
    return encoder.encode_double_vector(ctx, slots.tolist(), scale, chain_index)


def mask_scale_plaintext(
    ctx, encoder, d_head: int, d_total: int, num_tokens: int,
    scale_value: float, chain_index: int, encode_scale: float,
):
    """Plaintext with scale_value at each head-stride position, 0 elsewhere."""
    if d_head == 0 or d_total == 0:
        raise ValueError("mask_scale_plaintext: dimensions must be non-zero")
    if d_total % d_head != 0:
        raise ValueError("mask_scale_plaintext: d_total must be a multiple of d_head")
    num_slots = encoder.slot_count()
    slots = _head_stride_mask(num_slots, d_head, d_total, num_tokens, scale_value)
    return encoder.encode_double_vector(ctx, slots.tolist(), encode_scale, chain_index)


# ---------------------------------------------------------------------------
# Broadcast within blocks
# ---------------------------------------------------------------------------

def broadcast_within_blocks(ctx, galois_key, ct, block_size: int):
    """Broadcast each block's slot-0 to all positions via negative-stride rotate+add."""
    if not _is_pow2(block_size):
        raise ValueError("broadcast_within_blocks: block_size must be a power of 2")
    if block_size == 1:
        return ct
    result = ct
    bstride = block_size // 2
    while bstride >= 1:
        rot = phantom.rotate(ctx, result, -int(bstride), galois_key)
        result = phantom.add(ctx, result, rot)
        if bstride == 1:
            break
        bstride >>= 1
    return result


# ---------------------------------------------------------------------------
# Scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    ctx, encoder, relin_key, galois_key,
    q_ct, k_ct, v_ct,
    d_head: int, n_heads: int, num_tokens: int,
    *,
    encode_scale: float = None,
):
    """QK^T -> mask*scale -> softmax -> score*V.  Defaults: NUM_SQUARINGS=0,
    EXTRA_SCALE=0.5, ITERS=2 (matching the C++ constants).
    encode_scale defaults to q_ct.scale().
    """
    if not _is_pow2(d_head):
        raise ValueError("scaled_dot_product_attention: d_head must be a power of 2")
    if n_heads == 0:
        raise ValueError("scaled_dot_product_attention: n_heads must be non-zero")
    if not _is_pow2(num_tokens):
        raise ValueError("scaled_dot_product_attention: num_tokens must be a power of 2")
    d_total = n_heads * d_head
    slot_count = encoder.slot_count()
    nominal = q_ct.scale()
    if encode_scale is None:
        encode_scale = nominal

    # Phase 1: QK^T -> scores at slot[t*d_total + h*d_head].
    scores = phantom.compute_qkt(ctx, relin_key, galois_key, q_ct, [k_ct], d_head)[0]

    # Phase 2: mask + scale by 1/sqrt(d_head).
    inv_sqrt_d = 1.0 / math.sqrt(float(d_head))
    ms_slots = _head_stride_mask(slot_count, d_head, d_total, num_tokens, inv_sqrt_d)
    scores = _encode_mul_rescale_snap(ctx, encoder, scores, ms_slots, encode_scale, nominal)

    # Phase 3: softmax over t-axis with stride d_total.
    NUM_SQUARINGS = 0
    EXTRA_SCALE = 0.5
    ITERS = 2
    e_ct = phantom.ps_exp_init(
        ctx, encoder, relin_key, scores, num_tokens, NUM_SQUARINGS, EXTRA_SCALE,
    )
    phantom.square_iterations_inplace(ctx, relin_key, e_ct, NUM_SQUARINGS)

    # Strip ps_exp_init's leading constant combined with the slot mask.
    s_factor_inv = (1.0 / EXTRA_SCALE) ** (2.0 ** float(NUM_SQUARINGS))
    s_slots = _head_stride_mask(slot_count, d_head, d_total, num_tokens, s_factor_inv)
    e_ct = _encode_mul_rescale_snap(ctx, encoder, e_ct, s_slots, encode_scale)

    reduce_count = slot_count // d_total
    weights = phantom.finalize_softmax(
        ctx, encoder, relin_key, galois_key, e_ct, reduce_count, d_total, ITERS,
    )

    # Phase 4: score × V via the C++ primitive.
    weights_ci = weights.chain_index()
    sv_mask = score_mask_plaintext(
        ctx, encoder, d_head, d_total, num_tokens, weights_ci, encode_scale,
    )
    return phantom.score_times_v(
        ctx, relin_key, galois_key,
        [weights], [v_ct], sv_mask,
        d_head, d_total, num_tokens,
    )


# ---------------------------------------------------------------------------
# Attention forward (Wq + SDPA + Wo)
# ---------------------------------------------------------------------------

def attention_forward(
    ctx, encoder, relin_key, galois_key,
    x_ct, w_q, w_o,
    packed_k, packed_v,
    d_head: int, n_heads: int, num_tokens: int,
    *,
    encode_scale: float = None,
):
    """BSGS Wq -> SDPA -> mask+replicate -> BSGS Wo.

    w_q, w_o: pre-encoded BSGS diagonals with d_pad == n_heads * d_head.
    packed_k, packed_v: single-element lists (single-chunk K/V only).
    """
    if not packed_k or not packed_v:
        raise ValueError("attention_forward: packed_k/packed_v must be non-empty")
    if len(packed_k) != 1 or len(packed_v) != 1:
        raise ValueError("attention_forward: only single-chunk K/V supported in this slice")
    d_total = n_heads * d_head
    if w_q.d_pad != d_total:
        raise ValueError("attention_forward: w_q.d_pad must equal d_total (== d_model)")
    if w_o.d_pad != d_total:
        raise ValueError("attention_forward: w_o.d_pad must equal d_total (== d_model)")

    nominal = x_ct.scale()
    if encode_scale is None:
        encode_scale = nominal

    # 1. Q projection: q = W_q * x.
    q_ct = phantom.bsgs_matmul_preencoded(ctx, galois_key, x_ct, w_q)

    # 2. SDPA: drop K, V to q's level.
    k_at_level = phantom.mod_switch_to(ctx, packed_k[0], q_ct.chain_index())
    v_at_level = phantom.mod_switch_to(ctx, packed_v[0], q_ct.chain_index())
    attn = scaled_dot_product_attention(
        ctx, encoder, relin_key, galois_key,
        q_ct, k_at_level, v_at_level,
        d_head, n_heads, num_tokens,
        encode_scale=encode_scale,
    )

    # 3. Mask block-0 then replicate across all d_total-wide periods (1 level).
    num_slots = encoder.slot_count()
    block0 = np.zeros(num_slots, dtype=np.float64)
    block0[:d_total] = 1.0
    attn = _encode_mul_rescale_snap(ctx, encoder, attn, block0, encode_scale, nominal)
    attn = phantom.replicate(ctx, galois_key, attn, d_total, num_slots)

    # 4. Output projection: out = W_o * attn.
    return phantom.bsgs_matmul_preencoded(ctx, galois_key, attn, w_o)


# ---------------------------------------------------------------------------
# LLaMA-style attention forward (BSGS Wq + calibrated softmax + score×V + Wo).
#
# Differs from attention_forward / scaled_dot_product_attention above in:
#   - subtracts a per-head calibration constant C[h] before exp (FHE max-shift)
#   - uses NUM_SQUARINGS > 0 with damped squarings (deeper softmax range)
#   - applies the slot-mask BEFORE finalize_softmax (the deg-4 poly does NOT
#     evaluate to zero at zero, so non-meaningful slots must be zeroed before
#     sum_reduce_stride pollutes the in-block sum)
#   - optionally interleaves bootstrap calls between sub-stages A/B and B/C
#
# This is the path used by both llama3_simulation (no bootstrap_fn) and
# llama3 (bootstrap_fn=lambda ct: boot_centered(...)).
# ---------------------------------------------------------------------------

def attention_forward_llama(
    ctx, encoder, sk, relin_key, galois_key,
    x_ct, w_q, w_o,
    k_ct, v_ct,
    c_per_head,
    *,
    d_head: int, n_heads: int, num_tokens: int,
    num_squarings: int, extra_scale: float, target_mag: float, iters: int,
    encode_scale: float,
    bootstrap_fn=None,
    stage_times=None,
):
    """LLaMA attention: Wq -> QK^T -> mask*scale -> sub(C) -> damped softmax ->
    score*V -> mask+replicate -> Wo.

    c_per_head: per-head score upper-bound for the sub(C) centering step.
    bootstrap_fn: optional ct->ct callback invoked between stages A/B and B/C.
    stage_times: if provided, accumulates per-stage wall-time (ms).
    """
    if not _is_pow2(d_head):
        raise ValueError("attention_forward_llama: d_head must be a power of 2")
    if not _is_pow2(num_tokens):
        raise ValueError("attention_forward_llama: num_tokens must be a power of 2")
    d_total = n_heads * d_head
    num_slots = encoder.slot_count()

    def _t():
        return time.perf_counter()

    def _rec(name, t0):
        if stage_times is None:
            return
        stage_times.setdefault(name, 0.0)
        stage_times[name] += (time.perf_counter() - t0) * 1000.0

    # ---- Stage A: Wq + QK^T + mask*scale + sub(C). ----
    t0 = _t()
    q_ct = phantom.bsgs_matmul_preencoded(ctx, galois_key, x_ct, w_q)
    phantom.mod_switch_to_inplace(ctx, k_ct, q_ct.chain_index())
    scores_ct = phantom.compute_qkt(ctx, relin_key, galois_key, q_ct, [k_ct], d_head)[0]
    nominal = scores_ct.scale()
    inv_sqrt_d = 1.0 / math.sqrt(float(d_head))
    ms_slots = _head_stride_mask(num_slots, d_head, d_total, num_tokens, inv_sqrt_d)
    scores_ct = _encode_mul_rescale_snap(
        ctx, encoder, scores_ct, ms_slots, encode_scale, nominal)
    # Build per-head sub(C) mask: each head gets its own c_per_head[h] value.
    sub_slots = np.zeros(num_slots, dtype=np.float64)
    for t in range(num_tokens):
        for h in range(n_heads):
            sub_slots[t * d_total + h * d_head] = c_per_head[h]
    sub_pt = encoder.encode_double_vector(
        ctx, sub_slots.tolist(), scores_ct.scale(), scores_ct.chain_index())
    scores_ct = phantom.sub_plain(ctx, scores_ct, sub_pt)
    _rec("attn_A_wq_qkt_mask_sub", t0)

    if bootstrap_fn is not None:
        t0 = _t()
        scores_ct = bootstrap_fn(scores_ct)
        _rec("bootstrap", t0)

    # ---- Stage B: ps_exp_init + damped squarings. ----
    t0 = _t()
    damps = softmax_damping_schedule(num_squarings, num_tokens, extra_scale, target_mag)
    e_ct = phantom.ps_exp_init(
        ctx, encoder, relin_key, scores_ct,
        num_tokens, num_squarings, extra_scale)
    phantom.square_iterations_damped_inplace(ctx, encoder, relin_key, e_ct, damps)
    _rec("attn_B_ps_exp_sq", t0)

    if bootstrap_fn is not None:
        t0 = _t()
        e_ct = bootstrap_fn(e_ct)
        _rec("bootstrap", t0)

    # ---- Stage C: mask + finalize_softmax + score*V + mask*replicate + Wo. ----
    t0 = _t()
    # Zero non-meaningful slots before sum_reduce_stride (poly(0) != 0).
    mask_slots = _head_stride_mask(num_slots, d_head, d_total, num_tokens)
    e_ct = _encode_mul_rescale_snap(ctx, encoder, e_ct, mask_slots, encode_scale)

    weights_ct = phantom.finalize_softmax(
        ctx, encoder, relin_key, galois_key, e_ct,
        num_slots // d_total, d_total, iters)

    phantom.mod_switch_to_inplace(ctx, v_ct, weights_ct.chain_index())
    sv_mask = score_mask_plaintext(
        ctx, encoder, d_head, d_total, num_tokens,
        weights_ct.chain_index(), encode_scale)
    attn_h = phantom.score_times_v(
        ctx, relin_key, galois_key, [weights_ct], [v_ct],
        sv_mask, d_head, d_total, num_tokens)

    b0 = np.zeros(num_slots, dtype=np.float64)
    b0[:d_total] = 1.0
    attn_h = _encode_mul_rescale_snap(ctx, encoder, attn_h, b0, encode_scale)
    attn_h = phantom.replicate(ctx, galois_key, attn_h, d_total, num_slots)
    attn_out_ct = phantom.bsgs_matmul_preencoded(ctx, galois_key, attn_h, w_o)
    _rec("attn_C_softmax_sv_wo", t0)
    return attn_out_ct


# ---------------------------------------------------------------------------
# Plaintext reference
# ---------------------------------------------------------------------------

def reference_attention_forward(
    x, w_q, w_o, packed_k, packed_v,
    d_model: int, d_head: int, n_heads: int, num_tokens: int,
):
    """Plaintext attention reference (Wq -> QK^T -> softmax -> V -> Wo)."""
    d_total = n_heads * d_head
    x_arr = np.asarray(x, dtype=np.float64)
    w_q_arr = np.asarray(w_q, dtype=np.float64)
    w_o_arr = np.asarray(w_o, dtype=np.float64)
    pk_arr = np.asarray(packed_k, dtype=np.float64)
    pv_arr = np.asarray(packed_v, dtype=np.float64)

    if x_arr.size != d_model:
        raise ValueError("reference_attention_forward: x size != d_model")
    if w_q_arr.size != d_total * d_model:
        raise ValueError("reference_attention_forward: w_q size != d_total * d_model")
    if w_o_arr.size != d_model * d_total:
        raise ValueError("reference_attention_forward: w_o size != d_model * d_total")
    if pk_arr.size != num_tokens * d_total:
        raise ValueError("reference_attention_forward: packed_k size != num_tokens * d_total")
    if pv_arr.size != num_tokens * d_total:
        raise ValueError("reference_attention_forward: packed_v size != num_tokens * d_total")

    w_q_mat = w_q_arr.reshape(d_total, d_model)
    w_o_mat = w_o_arr.reshape(d_model, d_total)
    pk_mat = pk_arr.reshape(num_tokens, d_total)
    pv_mat = pv_arr.reshape(num_tokens, d_total)

    # q = W_q · x  (length d_total)
    q = w_q_mat @ x_arr

    # scores[t][h] = (Q[h] · K[t][h]) / sqrt(d_head)
    inv_sqrt_d = 1.0 / math.sqrt(float(d_head))
    q_per_head = q.reshape(n_heads, d_head)
    k_per_head = pk_mat.reshape(num_tokens, n_heads, d_head)
    v_per_head = pv_mat.reshape(num_tokens, n_heads, d_head)
    # einsum: (h,i),(t,h,i)->(t,h)
    scores = np.einsum("hi,thi->th", q_per_head, k_per_head) * inv_sqrt_d

    # weights[t][h] = softmax_t(scores[:,h]); numerically stable.
    s_max = scores.max(axis=0, keepdims=True)
    ex = np.exp(scores - s_max)
    weights = ex / ex.sum(axis=0, keepdims=True)

    # attn_per_head[h,i] = sum_t weights[t,h] * V[t,h,i]
    attn_ph = np.einsum("th,thi->hi", weights, v_per_head)
    attn = attn_ph.reshape(d_total)

    # out = W_o @ attn  (length d_model)
    out = w_o_mat @ attn
    return out.tolist()
