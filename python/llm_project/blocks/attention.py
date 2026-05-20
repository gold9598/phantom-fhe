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


# ============================================================================
# K-1 RESTORE 2026-05-20: Production Cachemir §5.1 IRP-attention functions.
# Verbatim from origin/baseline:python/llm_project/blocks/attention.py
# (the n=83 F1=82.27 > Cachemir 82.19 pipeline). Names provably absent from
# the pre-restore dense-bootstrap17 attention.py (orchestrator-grep verified).
# Purely additive: no existing function modified.
# ============================================================================


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

