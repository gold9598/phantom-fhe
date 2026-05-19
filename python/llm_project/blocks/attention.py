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


# DIAGNOSTIC ONLY (opt-in via PROBE_DECRYPT_STAGES=1). Dumps a decrypted
# slot vector to disk so an offline harness can compute rel-RMS vs plain-math
# for the softmax internals (denominator `a`, broadcast sum). When the flag is
# unset _probe_dump_stage is a no-op: byte-identical to the original.
_PROBE_DECRYPT_STAGES = os.environ.get("PROBE_DECRYPT_STAGES") == "1"
_PROBE_DUMP_DIR = os.environ.get("PROBE_DUMP_DIR", "/tmp/probe_stage_dump")
_PROBE_DUMP_LAYER = [None]  # set by llama3_mrpc per verbose layer


def _probe_dump_stage(tag, v):
    if not (_PROBE_DECRYPT_STAGES and _PROBE_DUMP_LAYER[0] is not None):
        return
    os.makedirs(_PROBE_DUMP_DIR, exist_ok=True)
    safe = (tag.replace("/", "_").replace(" ", "_")
            .replace("[", "").replace("]", "").replace("(", "").replace(")", ""))
    np.save(f"{_PROBE_DUMP_DIR}/L{_PROBE_DUMP_LAYER[0]}__smx_{safe}.npy",
            np.asarray(v, dtype=np.float64))


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


# ---------------------------------------------------------------------------
# Cachemir interleaved-replicated SDPA (Section 5.1)
# ---------------------------------------------------------------------------
#
# Layout convention (used by compute_qkt_irp / score_times_v_irp /
# finalize_softmax_irp_t):
#
#   Q ciphertext (post-Wq IRP, stride-t = stride-(N/d_total)):
#     q_slot[(h*d_head + j) * t] = Q[h, j]                 for h<n_heads, j<d_head
#     all other slots = 0
#
#   K cache ciphertext (interleaved across t tokens within one ct, all
#   n_heads expanded post-GQA):
#     k_slot[(h*d_head + j) * t + tok] = K_full[tok, h, j] for tok<num_tokens
#     k_slot[..., tok>=num_tokens] = 0
#
#   V cache ciphertext: same layout as K cache.
#
#   Attention map ciphertext (after compute_qkt_irp + scale + sub(C)):
#     scores_slot[h * d_head * t + tok] = m[tok, h]        for h<n_heads, tok<num_tokens
#     all other slots = 0
#
#   Pre-softmax interleaved ct (after exp+squarings+pre-mask, before
#   finalize_softmax_irp_t): same layout as attention map but with e[tok,h]
#   in the same valid slots, replicated to slots tok in [0, t) by a single
#   `-num_tokens` rotate-add so the cyclic sum_reduce broadcasts the full
#   per-head sum to every valid token slot.
#
#   Score×V output (post-mask): stride-t at d=d_total, directly consumable
#   by Wo IRP (no relayout needed):
#     attn_slot[(h*d_head + j) * t] = Σ_tok weights[tok,h] * V[tok,h,j]
#
# All step builders below assume these conventions.


def qkt_irp_required_steps(d_head: int, d_total: int, t: int):
    """Galois steps for compute_qkt_irp.

    - Q preprocess (PURE replicate Q across t-slots within each t-stride
      block, i.e. q[i*t + r] = Q[i] for all r in [0, t)):  -2^s for s in
      [0, log2(t)).  This differs from the IRP preprocess used by Wq IRP,
      which intermixes diagonals via step (d_total-1)*2^s.
    - Reduce over j-axis at stride t * 2^s for s in [0, log2(d_head))
    """
    if not _is_pow2(d_head) or not _is_pow2(d_total) or not _is_pow2(t):
        raise ValueError("qkt_irp_required_steps: d_head, d_total, t must be powers of 2")
    log_t = int(round(math.log2(t)))
    log_d_head = int(round(math.log2(d_head)))
    steps = set()
    for s in range(log_t):
        steps.add(-int(1 << s))
    for s in range(log_d_head):
        steps.add(int(t * (1 << s)))
    return sorted(steps)


def softmax_irp_t_required_steps(num_tokens: int):
    """Galois steps for finalize_softmax_irp_t.

    - sum_reduce over t-axis at stride 1, count num_tokens: {1, 2, ..., num_tokens/2}
    - cyclic-replica fill via -num_tokens (one rotation; data lives in 0..num_tokens-1
      and is replicated to num_tokens..2*num_tokens-1 to make the count=num_tokens
      cyclic sum broadcast the full sum to every valid token slot).
    """
    if not _is_pow2(num_tokens) or num_tokens < 2:
        raise ValueError("softmax_irp_t_required_steps: num_tokens must be a power of 2 >= 2")
    steps = set()
    steps.add(-int(num_tokens))
    stride = 1
    while stride < num_tokens:
        steps.add(int(stride))
        stride <<= 1
    return sorted(steps)


def score_v_irp_required_steps(d_head: int, num_tokens: int, t: int):
    """Galois steps for score_times_v_irp.

    - Broadcast weights over j-axis: -t * 2^s for s in [0, log2(d_head))
    - Cross-token sum over tok-axis: 2^s for s in [0, log2(num_tokens))
    """
    if not _is_pow2(d_head) or not _is_pow2(num_tokens) or not _is_pow2(t):
        raise ValueError("score_v_irp_required_steps: d_head, num_tokens, t must be powers of 2")
    log_d_head = int(round(math.log2(d_head)))
    log_num_tokens = int(round(math.log2(num_tokens)))
    steps = set()
    for s in range(log_d_head):
        steps.add(-int(t * (1 << s)))
    for s in range(log_num_tokens):
        steps.add(int(1 << s))
    return sorted(steps)


def sdpa_irp_required_steps(d_head: int, d_total: int, num_tokens: int, t: int):
    """Combined Galois steps for full IRP-native SDPA: QK^T | softmax | score*V."""
    steps = set()
    steps.update(qkt_irp_required_steps(d_head, d_total, t))
    steps.update(softmax_irp_t_required_steps(num_tokens))
    steps.update(score_v_irp_required_steps(d_head, num_tokens, t))
    return sorted(steps)


# ---------------------------------------------------------------------------
# IRP-native plaintext mask builders (Cachemir Section 5.1)
# ---------------------------------------------------------------------------

def _qkt_irp_head_mask_slots(num_slots, d_head, d_total, t, num_tokens, value=1.0):
    """Slot vector with `value` at slot[h*d_head*t + tok] for h<n_heads, tok<num_tokens."""
    n_heads = d_total // d_head
    slots = np.zeros(num_slots, dtype=np.float64)
    head_block = d_head * t
    for h in range(n_heads):
        base = h * head_block
        for tok in range(num_tokens):
            idx = base + tok
            if idx < num_slots:
                slots[idx] = value
    return slots


def qkt_irp_mask_scale_plaintext(
    ctx, encoder, d_head: int, d_total: int, num_tokens: int, t: int,
    scale_value: float, chain_index: int, encode_scale: float,
):
    """Mask×scale plaintext for IRP attention map: keep `scale_value` at
    slot[h*d_head*t + tok] for h<n_heads, tok<num_tokens; zero elsewhere."""
    num_slots = encoder.slot_count()
    slots = _qkt_irp_head_mask_slots(num_slots, d_head, d_total, t, num_tokens, scale_value)
    return encoder.encode_double_vector(ctx, slots, encode_scale, chain_index)


def qkt_irp_per_head_sub_plaintext(
    ctx, encoder, d_head: int, d_total: int, num_tokens: int, t: int,
    c_per_head, chain_index: int, encode_scale: float,
):
    """Per-head subtraction plaintext (centering): place c_per_head[h] at
    slot[h*d_head*t + tok] for h<n_heads, tok<num_tokens."""
    n_heads = d_total // d_head
    if len(c_per_head) != n_heads:
        raise ValueError("qkt_irp_per_head_sub_plaintext: c_per_head length != n_heads")
    num_slots = encoder.slot_count()
    slots = np.zeros(num_slots, dtype=np.float64)
    head_block = d_head * t
    for h in range(n_heads):
        base = h * head_block
        for tok in range(num_tokens):
            idx = base + tok
            if idx < num_slots:
                slots[idx] = c_per_head[h]
    return encoder.encode_double_vector(ctx, slots, encode_scale, chain_index)


def score_v_irp_output_mask_plaintext(
    ctx, encoder, d_head: int, d_total: int, t: int,
    chain_index: int, encode_scale: float,
):
    """Output mask for score_times_v_irp: keep slot[(h*d_head + j)*t] = 1,
    zero elsewhere. The mask covers the full d_total = n_heads*d_head dims at
    stride t."""
    if d_total % d_head != 0:
        raise ValueError("score_v_irp_output_mask_plaintext: d_total must be a multiple of d_head")
    num_slots = encoder.slot_count()
    slots = np.zeros(num_slots, dtype=np.float64)
    # Stride-t at every i in [0, d_total).
    for i in range(d_total):
        idx = i * t
        if idx < num_slots:
            slots[idx] = 1.0
    return encoder.encode_double_vector(ctx, slots, encode_scale, chain_index)


# ---------------------------------------------------------------------------
# IRP-native compute_qkt
# ---------------------------------------------------------------------------

def compute_qkt_irp(
    ctx, encoder, relin_key, galois_key,
    q_ct, k_ct,
    d_head: int, d_total: int, t: int,
):
    """QK^T over interleaved-packed Q (stride-t) and K cache (interleaved
    across t tokens within one ct).

    Returns: ciphertext with attention map at slot[h*d_head*t + tok] = m[tok, h];
    other slots within each head's d_head*t block hold partial-junk that the
    caller must mask out (typically fused with the mask*scale step).

    Algorithm (Cachemir §5.1, single K-cache ct case):
      1. Preprocess Q: log2(t) rotate-adds with step -2^s replicate Q purely
         across all t-slots within each t-stride block (q[i*t + r] = Q[i] for
         all r). Note: this is PURE replication, NOT the diagonals-interleave
         preprocess used inside Wq IRP (which uses step (d-1)*2^s).
      2. ct·ct multiply Q_pp × K_cache.
      3. Reduce over j-axis: log2(d_head) rotate-adds with step t*2^s.

    Caller is responsible for: (a) ensuring q_ct and k_ct are at compatible
    chain levels, (b) applying a head-stride mask + 1/sqrt(d_head) scale
    after this function (mask is fused with rescale to save a level).
    """
    if not _is_pow2(d_head):
        raise ValueError("compute_qkt_irp: d_head must be a power of 2")
    if not _is_pow2(d_total):
        raise ValueError("compute_qkt_irp: d_total must be a power of 2")
    if not _is_pow2(t):
        raise ValueError("compute_qkt_irp: t must be a power of 2")
    log_t = int(round(math.log2(t)))
    log_d_head = int(round(math.log2(d_head)))

    # 1. Preprocess Q: pure replicate across t-slots via -2^s rotations.
    q_pp = q_ct
    for s in range(log_t):
        rot_amt = -(1 << s)
        rot_ct = phantom.rotate(ctx, q_pp, int(rot_amt), galois_key)
        q_pp = phantom.add(ctx, q_pp, rot_ct)

    # 2. ct·ct multiply.
    nominal = q_pp.scale()
    prod = phantom.multiply_and_relin(ctx, q_pp, k_ct, relin_key)
    prod = phantom.rescale_to_next(ctx, prod)
    prod.set_scale(nominal)

    # 3. Reduce over j-axis.
    acc = prod
    for s in range(log_d_head):
        rot_amt = t * (1 << s)
        rot_ct = phantom.rotate(ctx, acc, int(rot_amt), galois_key)
        acc = phantom.add(ctx, acc, rot_ct)

    return acc


def pack_score_blocks_tree(ctx, galois_key, score_blocks, t,
                            encoder=None, d_head=None, d_total=None,
                            encode_scale=None):
    """Tree-rotation pack: n_blocks score ciphertexts -> 1 packed ct in
    log2(n_blocks) rounds, using only n_blocks distinct rotation steps
    {-t, -2t, -4t, ..., -(n_blocks/2)*t}.

    Input layout (per block k):
      slot[h*d_head*t + tok_local] = m[k, h, tok_local] for tok_local<t,
      other slots zero (assumed masked).

    Output layout (packed):
      slot[h*d_head*t + (k*t + tok_local)] = m[k, h, tok_local]
      Equivalently: slot[h*d_head*t + tok_global] = m[h, tok_global]
      for tok_global<n_blocks*t.

    Required rotation step keys: {-t * 2^s for s in range(log2(n_blocks))}
    (negative). For n_blocks=64, t=8: {-8, -16, -32, -64, -128, -256}.

    If `encoder`, `d_head`, `d_total`, `encode_scale` are all provided,
    apply a strict 0/1 mask after each merge round that keeps only the
    slots populated so far (cumulative populated range expands as
    `[h*d_head*t + 0..(2*stride*t)-1]` per head) and zeroes everything
    else. This clips per-round leakage (junk * encoding_noise from
    unpopulated input slots) before it can accumulate over log2(n_blocks)
    rounds — needed when the score blocks have non-zero junk at their
    unpopulated slots, which is the case when this function is called
    on the QKT output (post stage-A premask only). Each mask consumes
    1 level; log2(n_blocks) = 6 levels for n_blocks=64.

    If those args are None, the function falls back to the cheap pack
    with no intermediate masks — correct only when input score_blocks
    have clean zeros at their unpopulated slots (e.g. post-stage-C).
    """
    n_blocks = len(score_blocks)
    if n_blocks == 0:
        raise ValueError("pack_score_blocks_tree: empty input")
    if n_blocks == 1:
        return score_blocks[0]
    use_mask = (encoder is not None and d_head is not None
                and d_total is not None and encode_scale is not None)
    cts = list(score_blocks)
    stride = 1
    while stride < n_blocks:
        new_cts = []
        for i in range(0, len(cts), 2):
            if i + 1 < len(cts):
                rot = phantom.rotate(ctx, cts[i + 1], -stride * t, galois_key)
                merged = phantom.add(ctx, cts[i], rot)
                if use_mask:
                    populated_len = 2 * stride * t
                    nominal = merged.scale()
                    mask_pt = encoder.encode_double_vector(
                        ctx,
                        _qkt_irp_head_mask_slots(
                            encoder.slot_count(), d_head, d_total, t,
                            populated_len, 1.0),
                        encode_scale, merged.chain_index())
                    merged = phantom.multiply_plain(ctx, merged, mask_pt)
                    merged = phantom.rescale_to_next(ctx, merged)
                    merged.set_scale(nominal)
                new_cts.append(merged)
            else:
                new_cts.append(cts[i])
        cts = new_cts
        stride *= 2
    return cts[0]


def packed_softmax_finalize(
    ctx, encoder, relin_key, galois_key,
    packed_e, head_first_slot_mask_pt,
    n_heads: int, d_head: int, t: int, n_blocks: int, iters: int,
    user_scale: float,
    sk=None, verbose=False,
):
    """Softmax aggregation on a packed score ciphertext (replaces
    multi_ct_softmax_finalize for the packed path).

    Packed input layout:
      slot[h*d_head*t + tok_global] = e[h, tok_global] for h<n_heads,
        tok_global<n_blocks*t. Slots beyond are zero.

    Algorithm (single-ct version of multi_ct_softmax_finalize):
      1. (No cross-block sum: already packed in one ct.)
      2. Within-head sum_reduce stride=1, count=n_blocks*t. Slot
         [h*d_head*t + 0] ends up holding per-head global sum.
      3. Mask + broadcast: spread per-head sum to slots
         [h*d_head*t + 0..n_blocks*t-1] for the per-token goldschmidt.
      4. Goldschmidt softmax_correct.

    Returns: single packed weights ct with same layout, weights[h, tok] /
    per_head_sum.

    Required rotation step keys (positive for sum_reduce, negative for
    broadcast): {±1, ±2, ±4, ..., ±(n_blocks*t/2)}.
    """
    total_t = n_blocks * t
    if not _is_pow2(total_t):
        raise ValueError(f"packed_softmax_finalize: total_t={total_t} must be pow2")

    def _p(tag, ct):
        if not (verbose and sk is not None): return
        v = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                     dtype=np.float64)
        head_block = d_head * t
        slot0 = v[0]
        head1_slot0 = v[head_block]
        print(f"      [packed-smx-probe] {tag:25s} slot0={slot0:.4e} "
              f"h1_slot0={head1_slot0:.4e} max|.|={np.abs(v).max():.4e}")
        _probe_dump_stage(f"packed_{tag}", v)

    _p("packed_e (input)", packed_e)
    a = packed_e
    step = 1
    reach = 1
    while reach < total_t:
        rot = phantom.rotate(ctx, a, int(step), galois_key)
        a = phantom.add(ctx, a, rot)
        step <<= 1
        reach <<= 1
    _p("a (post-reduce)", a)

    a_masked = phantom.multiply_plain(ctx, a, head_first_slot_mask_pt)
    a_masked = phantom.rescale_to_next(ctx, a_masked)
    a_masked.set_scale(user_scale)
    _p("a_masked", a_masked)
    a_bc = a_masked
    s = 1
    while s < total_t:
        rot = phantom.rotate(ctx, a_bc, -int(s), galois_key)
        a_bc = phantom.add(ctx, a_bc, rot)
        s <<= 1
    _p("a_bc (broadcast)", a_bc)

    a_chain = a_bc.chain_index()
    if packed_e.chain_index() != a_chain:
        packed_e = phantom.mod_switch_to(ctx, packed_e, a_chain)
    wb = phantom.softmax_correct(ctx, encoder, relin_key, packed_e, a_bc, iters)
    _p("wb (post-softmax_correct)", wb)
    return wb


def unpack_packed_weights(
    ctx, encoder, galois_key, packed_weights,
    n_blocks: int, d_head: int, d_total: int, t: int,
    chain_index: int, encode_scale: float,
):
    """Inverse of pack_score_blocks_tree. Extracts n_blocks per-block
    weights ciphertexts from a single packed weights ct, so the existing
    per-block score_v_irp_multi can consume them unchanged.

    For each block k:
      1. rotate packed by +k*t (slot k*t -> slot 0 within each head)
      2. multiply by a per-head 'first-t-slots' mask so only slots
         [h*d_head*t + 0..t-1] remain non-zero.
      3. rescale + set_scale.

    BSGS decomposition (baby_count=8, giant_count=8 covers k in [0, 64)):
      Factor k = baby + giant * baby_count, baby in [0, 8), giant in [0, 8).
      Required positive rotation step keys:
        baby:  {t * b for b in 1..baby_count-1}  (7 keys)
        giant: {t * baby_count * g for g in 1..giant_count-1}  (7 keys)
      Total 14 keys vs. 63 for the naive per-block path.

    Returns: list of n_blocks weights ciphertexts in per-block layout.
    """
    n_heads = d_total // d_head
    num_slots = encoder.slot_count()
    mask_slots = np.zeros(num_slots, dtype=np.float64)
    head_block = d_head * t
    for h in range(n_heads):
        base = h * head_block
        for tok_local in range(t):
            if base + tok_local < num_slots:
                mask_slots[base + tok_local] = 1.0
    mask_pt = encoder.encode_double_vector(
        ctx, mask_slots, encode_scale, chain_index)

    # BSGS factoring k = baby + giant * baby_count covering k in [0, n_blocks).
    baby_count = 8
    giant_count = (n_blocks + baby_count - 1) // baby_count

    # Pre-rotate 'packed_weights' by +t*baby for baby in [0, baby_count).
    # pre_baby[0] is the original ct (no rotation needed).
    pre_baby = [packed_weights]
    for b in range(1, baby_count):
        pre_baby.append(phantom.rotate(ctx, packed_weights, b * t, galois_key))

    blocks = []
    for k in range(n_blocks):
        baby = k % baby_count
        giant = k // baby_count
        base_ct = pre_baby[baby]
        if giant == 0:
            rotated = base_ct
        else:
            rotated = phantom.rotate(
                ctx, base_ct, giant * baby_count * t, galois_key)
        nominal = rotated.scale()
        masked = phantom.multiply_plain(ctx, rotated, mask_pt)
        masked = phantom.rescale_to_next(ctx, masked)
        masked.set_scale(nominal)
        blocks.append(masked)
    return blocks


def multi_ct_softmax_finalize(
    ctx, encoder, relin_key, galois_key,
    e_blocks, head_first_slot_mask_pt,
    n_heads: int, d_head: int, t: int, iters: int,
    user_scale: float,
    sk=None, verbose=False,
):
    """Cross-block softmax aggregation for multi-ct e (Stage 3b-c).

    Each e_block_k holds scaled exp values for tokens in block k:
        slot[h*d_head*t + tok_local] = e[h, k*t + tok_local]   for tok_local in [0, t)
    Slots beyond block_sizes[k] in the last (partial) block are zero (caller
    must mask them).

    Computes weights_block_k = e_block_k / global_sum_per_head, where
        global_sum_per_head[h] = Σ_k Σ_tok_local e[h, k*t + tok_local]
                              = Σ_t (over all token positions) e[h, t]

    Algorithm:
      1. Cross-block sum: e_sum = Σ_k e_block_k.
      2. Within-block sum_reduce stride=1 count=t. The stock sum_reduce only
         puts the correct per-head sum at slot[h*d_head*t + 0]; slots
         tok_local in [1, t) hold sliding-window partial sums and are wrong
         for our non-cyclic layout (the multi-block case has all t slots
         populated with distinct values per block, so the existing single-ct
         cyclic-replica trick doesn't apply).
      3. Mask + broadcast: a_broadcast = (a * head_first_slot_mask) then
         doubling rotate-adds to populate slots [h*d_head*t + 0..t) with the
         per-head sum. Costs 1 level for the mask multiply + rescale.
      4. Per-block Goldschmidt: weights_block_k = phantom.softmax_correct(
         e_block_k, a_broadcast, iters) -> e_block_k / a_broadcast.

    Args:
      head_first_slot_mask_pt: precomputed plaintext with 1.0 at slot
        [h*d_head*t + 0] for h in [0, n_heads) and 0.0 elsewhere, encoded at
        the chain matching `a` after step 2.
      user_scale: engine.user_scale() — used to snap a's scale after the
        mask multiply + rescale.

    Required Galois rotation steps:
      step 2: {1, 2, ..., t/2} (positive)
      step 3: {-1, -2, ..., -t/2} (negative, for the broadcast doubling)

    Returns: list of n_blocks weights ciphertexts.
    """
    n_blocks = len(e_blocks)
    if n_blocks == 0:
        raise ValueError("multi_ct_softmax_finalize: e_blocks is empty")
    if not _is_pow2(t):
        raise ValueError("multi_ct_softmax_finalize: t must be a power of 2")

    # 1. Cross-block sum.
    def _p(tag, ct):
        if not (verbose and sk is not None): return
        v = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                     dtype=np.float64)
        # show value at slot[h*d_head*t + 0] for h=0 (representative per-head sum)
        slot0 = v[0]
        print(f"      [softmax-probe] {tag:25s} slot0={slot0:.4e} max|.|={np.abs(v).max():.4e}")
        _probe_dump_stage(f"multi_{tag}", v)

    e_sum = e_blocks[0]
    for k in range(1, n_blocks):
        e_sum = phantom.add(ctx, e_sum, e_blocks[k])
    _p("e_sum (cross-block)", e_sum)

    # 2. Within-block sum_reduce: stride=1, count=t. After log2(t) iterations,
    # acc[h*d_head*t + 0] holds the global per-head sum; slots 1..t-1 hold
    # sliding-window partial sums (wrong for our layout).
    a = e_sum
    step = 1
    reach = 1
    while reach < t:
        rot = phantom.rotate(ctx, a, int(step), galois_key)
        a = phantom.add(ctx, a, rot)
        step <<= 1
        reach <<= 1
    _p("a (post-reduce)", a)

    # 3. Mask + broadcast. Mask zeroes everything except slot[h*d_head*t + 0]
    # for each h, leaving the correct per-head sum at slot 0 only. Then
    # doubling rotate-adds spread that value to slots [0..t).
    a_masked = phantom.multiply_plain(ctx, a, head_first_slot_mask_pt)
    a_masked = phantom.rescale_to_next(ctx, a_masked)
    a_masked.set_scale(user_scale)
    _p("a_masked", a_masked)
    a_bc = a_masked
    s = 1
    while s < t:
        rot = phantom.rotate(ctx, a_bc, -int(s), galois_key)
        a_bc = phantom.add(ctx, a_bc, rot)
        s <<= 1
    _p("a_bc (broadcast)", a_bc)

    # DIAGNOSTIC ONLY (PROBE_DENSE_SMX=1): decrypt the IRP path's per-head
    # denominator a_bc at the per-head first slots (h*d_head*t + 0) so it can
    # be compared directly against the dense path's a_total magnitude. This is
    # the `a` IRP's softmax_correct consumes; its distribution vs the (0,2)/
    # 1.5 window is the inter-pipeline conditioning reference. Decrypt only;
    # zero ciphertext effect. Not entered unless the env flag is set and a
    # decrypt key was threaded in (the dense gate passes sk when verbose; we
    # also accept it via the env flag for this targeted probe).
    if os.environ.get("PROBE_DENSE_SMX") == "1" and sk is not None:
        import numpy as _np
        _abv = _np.array(
            encoder.decode_double_vector(ctx, sk.decrypt(ctx, a_bc)),
            dtype=_np.float64)
        _hb = d_head * t
        _ah = _abv[_np.array([_h * _hb + 0 for _h in range(n_heads)])]
        print(f"      [PROBE-SMX] IRP a_bc (decrypted, per-head first "
              f"slots, nH={n_heads}): min={_ah.min():.6f} "
              f"max={_ah.max():.6f} mean={_ah.mean():.6f} "
              f">1.9:{int((_ah>1.9).sum())} >1.99:{int((_ah>1.99).sum())} "
              f">=2.0:{int((_ah>=2.0).sum())} <0.05:{int((_ah<0.05).sum())}")

    # 4. Per-block Goldschmidt. The e_blocks are still at the pre-mask-rescale
    # chain; mod_switch them down to a_bc's chain so softmax_correct accepts
    # both at the same level.
    a_chain = a_bc.chain_index()
    weights_blocks = []
    for e_block in e_blocks:
        if e_block.chain_index() != a_chain:
            e_block = phantom.mod_switch_to(ctx, e_block, a_chain)
        wb = phantom.softmax_correct(ctx, encoder, relin_key, e_block, a_bc, iters)
        weights_blocks.append(wb)
    return weights_blocks


def compute_qkt_irp_multi(
    ctx, encoder, relin_key, galois_key,
    q_ct, k_blocks_cts,
    d_head: int, d_total: int, t: int,
    num_tokens: int = None,
):
    """QK^T over a multi-ciphertext K cache (Stage 3b-b).

    Each k_block holds up to t = T_MODEL = 8 token positions packed in the
    same in-block layout that compute_qkt_irp consumes. Block k holds tokens
    [k*t, k*t + t). The final block may be partial (block_size < t); slots
    beyond block_size are zero in the K-block ct, and the resulting score
    block has zeros at those positions too — those are masked downstream.

    Args:
        q_ct: query ciphertext (single, post-Wq-IRP, stride-t Q layout).
        k_blocks_cts: list of n_blocks K-block ciphertexts.
        d_head, d_total, t: same as compute_qkt_irp.
        num_tokens: total number of active tokens across all blocks. Used to
            compute per-block sizes (returned alongside the score blocks).
            If None, defaults to len(k_blocks_cts) * t (assume all full).

    Returns:
        score_blocks: list of n_blocks ciphertexts, each with the same per-
            block layout that compute_qkt_irp's single-ct output has —
            slot[h*d_head*t + tok_local] = m[tok_local, h] for tok_local in
            [0, t). The mid-head junk slots [d_head*tok_local + 1 .. ] hold
            partial-junk that the caller must mask out.
        block_sizes: list of n_blocks ints, the number of valid tokens in
            each block. block_sizes[k] = min(t, num_tokens - k*t).

    Q is preprocessed (replicated across t-slots) ONCE outside the per-block
    loop — saves log2(t) rotations per block at the cost of one extra ct
    holding the preprocessed Q.

    Caller is responsible for:
      * mod_switch'ing each k_block to q_ct's chain (the existing pipeline
        does this via mod_switch_to_inplace before compute_qkt_irp);
      * applying mask*scale + per-head sub on each score block (block-aware
        mask: only block_sizes[k] tokens are meaningful in block k);
      * cross-block softmax aggregation in the multi-ct softmax pipeline.
    """
    n_blocks = len(k_blocks_cts)
    if num_tokens is None:
        num_tokens = n_blocks * t
    block_sizes = [min(t, num_tokens - k * t) for k in range(n_blocks)]

    if not _is_pow2(d_head):
        raise ValueError("compute_qkt_irp_multi: d_head must be a power of 2")
    if not _is_pow2(d_total):
        raise ValueError("compute_qkt_irp_multi: d_total must be a power of 2")
    if not _is_pow2(t):
        raise ValueError("compute_qkt_irp_multi: t must be a power of 2")
    log_t = int(round(math.log2(t)))
    log_d_head = int(round(math.log2(d_head)))

    # Preprocess Q ONCE across the per-block loop: pure replicate across t-slots.
    q_pp = q_ct
    for s in range(log_t):
        rot_amt = -(1 << s)
        rot_ct = phantom.rotate(ctx, q_pp, int(rot_amt), galois_key)
        q_pp = phantom.add(ctx, q_pp, rot_ct)

    score_blocks = []
    for k_ct in k_blocks_cts:
        # ct·ct multiply Q_pp × K_block.
        nominal = q_pp.scale()
        prod = phantom.multiply_and_relin(ctx, q_pp, k_ct, relin_key)
        prod = phantom.rescale_to_next(ctx, prod)
        prod.set_scale(nominal)
        # Reduce over j-axis (the d_head dimension).
        acc = prod
        for s in range(log_d_head):
            rot_amt = t * (1 << s)
            rot_ct = phantom.rotate(ctx, acc, int(rot_amt), galois_key)
            acc = phantom.add(ctx, acc, rot_ct)
        score_blocks.append(acc)

    return score_blocks, block_sizes


# ---------------------------------------------------------------------------
# IRP-native finalize_softmax (replaces phantom.finalize_softmax for stride-t)
# ---------------------------------------------------------------------------

def finalize_softmax_irp_t(
    ctx, encoder, relin_key, galois_key,
    e_ct, num_tokens: int, iters: int,
):
    """Cyclic-broadcast softmax for the IRP layout.

    Input layout (post-mask): valid e[tok, h] at slot[h*head_stride + tok]
    for tok in [0, num_tokens), where head_stride = d_head*t. Slots
    [num_tokens, t) within each head's first-t-slots are zero.

    Trick: rotate-add by -num_tokens to copy [e0, e1, e2, e3, 0, 0, 0, 0] into
    [e0, e1, e2, e3, e0, e1, e2, e3]; sum_reduce_stride(stride=1, count=num_tokens)
    then broadcasts the full per-head sum to every valid token slot via the
    cyclic shift property. Apply softmax_correct (Goldschmidt) with the
    broadcast sum.

    Cost: 1 extra negative rotation (step -num_tokens), log2(num_tokens)
    standard sum-reduce rotations, then `iters` Goldschmidt iterations
    (2 levels each).
    """
    if not _is_pow2(num_tokens) or num_tokens < 2:
        raise ValueError("finalize_softmax_irp_t: num_tokens must be power of 2 >= 2")

    # Cyclic-replica fill: [e0,...,e_{n-1}, 0,...,0] -> [e0,...,e_{n-1}, e0,...,e_{n-1}].
    e_replica = phantom.rotate(ctx, e_ct, -int(num_tokens), galois_key)
    e_cyclic = phantom.add(ctx, e_ct, e_replica)

    # Sum-reduce stride=1, count=num_tokens. Cyclic block of size num_tokens:
    # every slot in [0, 2*num_tokens) holds the full per-head sum.
    a = e_cyclic
    step = 1
    reach = 1
    while reach < num_tokens:
        rot = phantom.rotate(ctx, a, int(step), galois_key)
        a = phantom.add(ctx, a, rot)
        step <<= 1
        reach <<= 1

    # Goldschmidt 1/a iterations via softmax_correct (consumes 2 levels per iter).
    return phantom.softmax_correct(ctx, encoder, relin_key, e_cyclic, a, iters)


# ---------------------------------------------------------------------------
# IRP-native score_times_v
# ---------------------------------------------------------------------------

def score_times_v_irp(
    ctx, encoder, relin_key, galois_key,
    weights_ct, v_ct,
    d_head: int, d_total: int, num_tokens: int, t: int,
    output_mask_pt,
):
    """Compute Σ_tok weights[tok, h] * V[tok, h, j] in IRP layout.

    Inputs:
      weights_ct: post-softmax-cyclic ct with weights[tok, h] at slot
        [h*d_head*t + tok] AND its cyclic replica at slot [h*d_head*t + (tok+num_tokens)]
        (this is the natural output of finalize_softmax_irp_t).
      v_ct: V cache in same interleaved-tokens layout as K cache.
      output_mask_pt: stride-t mask (1 at slot i*t for i in [0, d_total), else 0)
        encoded at the chain level resulting after the ct·ct + reduce steps.

    Algorithm (dual of compute_qkt_irp):
      1. Broadcast weights over j-axis: log2(d_head) rotate-adds with step -t*2^s.
      2. ct·ct multiply weights_b × V_cache.
      3. Reduce over tok-axis: log2(num_tokens) rotate-adds with step 2^s.
      4. Mask at stride t to keep one copy per (h, j); rescale + snap scale.

    Returns: stride-t ciphertext consumable by Wo IRP (slot[(h*d_head+j)*t] = attn[h,j]).
    """
    if not _is_pow2(d_head):
        raise ValueError("score_times_v_irp: d_head must be a power of 2")
    if not _is_pow2(d_total):
        raise ValueError("score_times_v_irp: d_total must be a power of 2")
    if not _is_pow2(num_tokens) or num_tokens < 2:
        raise ValueError("score_times_v_irp: num_tokens must be power of 2 >= 2")
    if not _is_pow2(t):
        raise ValueError("score_times_v_irp: t must be a power of 2")
    log_d_head = int(round(math.log2(d_head)))
    log_num_tokens = int(round(math.log2(num_tokens)))

    # 1. Broadcast weights over j-axis (negative-stride add tree).
    wb = weights_ct
    for s in range(log_d_head):
        rot_amt = -int(t * (1 << s))
        rot_ct = phantom.rotate(ctx, wb, int(rot_amt), galois_key)
        wb = phantom.add(ctx, wb, rot_ct)

    # 2. ct·ct: wb × V.
    nominal = wb.scale()
    prod = phantom.multiply_and_relin(ctx, wb, v_ct, relin_key)
    prod = phantom.rescale_to_next(ctx, prod)
    prod.set_scale(nominal)

    # 3. Reduce over tok-axis.
    acc = prod
    for s in range(log_num_tokens):
        rot_amt = 1 << s
        rot_ct = phantom.rotate(ctx, acc, int(rot_amt), galois_key)
        acc = phantom.add(ctx, acc, rot_ct)

    # 4. Mask at stride t: keep tok=0 slot per (h, j).
    nominal = acc.scale()
    out = phantom.multiply_plain(ctx, acc, output_mask_pt)
    out = phantom.rescale_to_next(ctx, out)
    out.set_scale(nominal)
    return out


def score_times_v_irp_multi(
    ctx, encoder, relin_key, galois_key,
    weights_blocks, v_blocks_cts,
    d_head: int, d_total: int, t: int,
    output_mask_pt,
    num_tokens_per_block: int = None,
):
    """Compute Σ_t weights[t, h] * V[t, h, j] over a multi-ciphertext
    weights/V cache (Stage 3b-d).

    Each weights_block_k and v_block_k holds tokens [k*t, (k+1)*t). Calls
    score_times_v_irp on each pair to produce a per-block partial attn
    output, then sums across blocks for the final attn_irp ct.

    For partial blocks (last block with block_size < t), the weights_block
    is already masked (slots beyond block_size are zero from the per-block
    softmax pipeline), so the contribution from those slots is zero — we
    pass num_tokens_per_block=t for all blocks. The reduce step in
    score_times_v_irp sums all t slots; the masked-out positions contribute
    zero.

    Required Galois steps: same as score_times_v_irp for a single block —
    {-t, -2t, ..., -t*d_head/2} (broadcast over j-axis) and {1, 2, ..., t/2}
    (reduce over tok-axis).
    """
    if num_tokens_per_block is None:
        num_tokens_per_block = t
    n_blocks = len(weights_blocks)
    if len(v_blocks_cts) != n_blocks:
        raise ValueError("score_times_v_irp_multi: weights and v block counts mismatch")
    if n_blocks == 0:
        raise ValueError("score_times_v_irp_multi: blocks list is empty")

    partials = []
    for w_block, v_block in zip(weights_blocks, v_blocks_cts):
        partial = score_times_v_irp(
            ctx, encoder, relin_key, galois_key,
            w_block, v_block,
            d_head, d_total, num_tokens_per_block, t, output_mask_pt)
        partials.append(partial)

    # Cross-block sum: free in level cost (just adds).
    attn = partials[0]
    for k in range(1, n_blocks):
        attn = phantom.add(ctx, attn, partials[k])
    return attn


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
        ctx, slots, encode_scale, ct.chain_index(),
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
    return encoder.encode_double_vector(ctx, slots, scale, chain_index)


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
    return encoder.encode_double_vector(ctx, slots, encode_scale, chain_index)


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
        ctx, sub_slots, scores_ct.scale(), scores_ct.chain_index())
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
