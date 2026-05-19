"""Dense token-major K/V layout — Stage 1 executable spec for the dense-layout rewrite.

PURE NUMPY. No phantom / FHE / GPU imports.

This module is the executable specification the later FHE stages are validated
against. Every contraction function simulates the EXACT slot-level operations
the C++ kernels perform on the flat slot vector — no reshape+matmul shortcuts.

===========================================================================
LAYOUT CONTRACT (source-proven from kernel inspection)
===========================================================================

Token-MAJOR, j-INNERMOST.  d_total = D = n_heads * d_head.

For a shard covering local token positions tok_local in [0, P):

  q_slot[tok_local * D + h*H + j] = Q[h, j]
      Query is pre-broadcast: the same Q[h,j] is copied into every
      tok_local offset so the elementwise multiply q*k is correctly aligned.
      (H = d_head, D = d_total = n_heads * d_head)

  k_slot[tok_local * D + h*H + j] = K[tok_local, h, j]   (GQA-expanded)
  v_slot[tok_local * D + h*H + j] = V[tok_local, h, j]   (GQA-expanded)

  score_slot[tok_local * D + h*H + j] = scores[tok_local, h]
      (replicated across the d_head block by inner_sum's cyclic rotation)

  o_slot[h*H + j] = sum_tok w[tok, h] * V[tok, h, j]
      (token-collapsed; lives at tok_local=0 offset, feeds Wo BSGS directly)

Shard / multi-ct parameters:
  P  = positions_per_ct = min(next_pow2(real_nt), NUM_SLOTS // D)
     = min(nt_pad, 32768 // 4096) = min(nt_pad, 8)
  Each shard ciphertext holds exactly P token positions.
  n_shards = ceil(nt_pad / P) = ceil(real_nt / P) rounded up to cover all.
  Tail positions (tok >= real_nt within a shard, or shard beyond real data)
  are padded with exact 0.0.
  Each shard slot-vector has length P * D.

GQA mapping: query head h -> kv head h // n_kv_groups  (block-repeat,
  matching np.repeat(K, n_kv_groups, axis=1) in llama3.py:328-329).

===========================================================================
CONTRACTION SEMANTICS (simulated exactly on flat slot vectors)
===========================================================================

compute_qkt  (src/attention.cu:21-25, src/linear.cu:17-24):
  prod = q * k   (elementwise on flat slots)
  inner_sum(prod, d_head):  for stride in [1,2,4,...,d_head/2]:
                                prod += rotate_left(prod, stride)
  Readout at slot tok_local*D + h*H (the d_head-aligned base):
    = sum_{j=0}^{H-1} q[tok_local*D + h*H + j] * k[tok_local*D + h*H + j]
    = sum_j Q[h,j] * K[tok_local, h, j]

score_times_v  (src/attention.cu:59-95):
  Step 1 — mask: zero all but base slots tok*D + h*H  (one value per tok,head).
  Step 2 — broadcast across d_head (src/attention.cu:66-72):
    bstride = d_head//2; while bstride >= 1: masked += rotate_right(masked, bstride); bstride //= 2
    Result: masked[tok*D + h*H + j] = w[tok,h] for all j in [0,H).
  Step 3 — multiply by V: prod[tok*D + h*H + j] = w[tok,h] * V[tok,h,j]
  Step 4 — accumulate over P positions (src/attention.cu:81-85):
    astride = D; while astride < P*D: prod += rotate_left(prod, astride); astride *= 2
    After loop, slot h*H + j (tok=0 offset) holds sum_{tok=0}^{P-1} w[tok,h]*V[tok,h,j].
  Step 5 — cross-shard add: sum the per-shard outputs.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def next_pow2(n: int) -> int:
    """Smallest power of two >= n (>=1).  next_pow2(0)=1, next_pow2(1)=1."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def positions_per_ct(nt_pad: int, num_slots: int, d_total: int) -> int:
    """P = min(nt_pad, num_slots // d_total); result is a power of two."""
    p = min(nt_pad, num_slots // d_total)
    assert p >= 1, f"positions_per_ct: p={p} < 1 (num_slots={num_slots}, d_total={d_total})"
    assert (p & (p - 1)) == 0, f"positions_per_ct: p={p} is not a power of 2"
    return p


def _kv_head_for_query(h: int, n_heads: int, n_kv_heads: int) -> int:
    """GQA block-repeat: query head h -> kv head h // n_kv_groups."""
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )
    return h // (n_heads // n_kv_heads)


# ---------------------------------------------------------------------------
# inner_sum simulation (src/linear.cu:17-24)
# Operates in-place on a numpy slot vector (or a copy); stride is LEFT rotation
# (positive index shift wraps cyclically over the slot vector length).
# ---------------------------------------------------------------------------

def _rotate_left(slots: np.ndarray, stride: int) -> np.ndarray:
    """Cyclic left rotation by stride positions: result[i] = slots[(i+stride) % N]."""
    return np.roll(slots, -stride)


def _inner_sum_slots(slots: np.ndarray, block_size: int) -> np.ndarray:
    """Simulate inner_sum(ct, block_size) from src/linear.cu:17-24.

    acc = slots.copy()
    for stride = 1, 2, 4, ..., block_size//2:
        acc += rotate_left(acc, stride)

    The value at every d_head-aligned base slot s = k*block_size becomes
        sum_{j=0}^{block_size-1} slots[s + j]
    (values at non-base positions are a partial prefix sum — not used).
    """
    assert block_size >= 1 and (block_size & (block_size - 1)) == 0, \
        f"block_size must be power of 2, got {block_size}"
    acc = slots.copy()
    stride = 1
    while stride < block_size:
        acc = acc + _rotate_left(acc, stride)
        stride <<= 1
    return acc


# ---------------------------------------------------------------------------
# score_times_v broadcast simulation (src/attention.cu:66-72)
# Negative-stride (RIGHT) rotations spread the base-slot value across the
# d_head block.
# ---------------------------------------------------------------------------

def _rotate_right(slots: np.ndarray, stride: int) -> np.ndarray:
    """Cyclic right rotation by stride: result[i] = slots[(i-stride) % N]."""
    return np.roll(slots, stride)


def _broadcast_within_heads(slots: np.ndarray, d_head: int) -> np.ndarray:
    """Simulate the broadcast step from src/attention.cu:66-72.

    bstride = d_head // 2
    while bstride >= 1:
        masked += rotate_right(masked, bstride)
        bstride //= 2

    Given that slots[s*D + h*H] = w for one value at the base slot and
    zeros elsewhere within the d_head block, after the loop
    slots[s*D + h*H + j] = w for all j in [0, H).
    """
    acc = slots.copy()
    bstride = d_head // 2
    while bstride >= 1:
        acc = acc + _rotate_right(acc, bstride)
        if bstride == 1:
            break
        bstride >>= 1
    return acc


def _accumulate_over_positions(slots: np.ndarray, P: int, d_total: int) -> np.ndarray:
    """Simulate token-accumulate step from src/attention.cu:81-85.

    max_accumulate = P * d_total
    astride = d_total
    while astride < max_accumulate:
        prod += rotate_left(prod, astride)
        astride *= 2

    Folds all P token positions into slot 0 (tok_local=0 offset) by
    summing rotate_left by d_total, 2*d_total, ..., (P//2)*d_total.
    After the loop, slot[0*D + h*H + j] holds sum_{tok=0}^{P-1} prod[tok*D + h*H + j].
    """
    max_acc = P * d_total
    acc = slots.copy()
    astride = d_total
    while astride < max_acc:
        acc = acc + _rotate_left(acc, astride)
        astride <<= 1
    return acc


# ---------------------------------------------------------------------------
# Q packing (single token; broadcast across P positions within a shard)
# ---------------------------------------------------------------------------

def pack_q_dense(Q_h_d: np.ndarray, P: int) -> np.ndarray:
    """Pack Q[n_heads, d_head] into a shard slot vector of length P * D.

    q_slot[tok_local * D + h*H + j] = Q[h, j]  for all tok_local in [0, P).
    The same Q value is broadcast across all P token-major slots so that
    the elementwise multiply q*k aligns Q[h,j] with K[tok_local,h,j].
    """
    Q = np.asarray(Q_h_d, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError(f"pack_q_dense: expected [n_heads, d_head], got {Q.shape}")
    n_heads, d_head = Q.shape
    D = n_heads * d_head
    slots = np.zeros(P * D, dtype=np.float64)
    # Q[h, :] -> contiguous block at h*d_head within each tok_local frame
    # Vectorised: flatten Q to D, tile P times
    q_frame = Q.ravel()  # [D] with layout Q[0,0..H-1] | Q[1,0..H-1] | ...
    for tok_local in range(P):
        slots[tok_local * D:(tok_local + 1) * D] = q_frame
    return slots


def unpack_q_dense(q_slots: np.ndarray, n_heads: int, d_head: int) -> np.ndarray:
    """Recover Q[n_heads, d_head] from the first token-frame of q_slots."""
    q = np.asarray(q_slots, dtype=np.float64)
    D = n_heads * d_head
    if q.shape[0] < D:
        raise ValueError(f"unpack_q_dense: need >= {D} slots, got {q.shape[0]}")
    return q[:D].reshape(n_heads, d_head).copy()


# ---------------------------------------------------------------------------
# K/V packing (multi-shard)
# ---------------------------------------------------------------------------

def pack_kv_shard(K_t_h_d: np.ndarray, V_t_h_d: np.ndarray,
                  shard_tok_start: int, P: int,
                  n_heads: int, n_kv_heads: int) -> tuple:
    """Pack one K/V shard: tokens [shard_tok_start, shard_tok_start+P) into a
    slot vector of length P * D.

    k_slot[tok_local * D + h*H + j] = K[shard_tok_start + tok_local, h, j]
    Tokens beyond real_nt are 0.0 (K/V arrays already zero-padded to full
    nt_pad by the caller if needed, or shard extends past the array end which
    is handled by slicing into a zero-padded copy inside pack_kv_dense_shards).
    """
    K = np.asarray(K_t_h_d, dtype=np.float64)  # [real_nt or nt_pad, n_kv_heads, d_head]
    V = np.asarray(V_t_h_d, dtype=np.float64)
    _, n_kv_heads_in, d_head = K.shape
    D = n_heads * d_head
    k_slots = np.zeros(P * D, dtype=np.float64)
    v_slots = np.zeros(P * D, dtype=np.float64)
    for tok_local in range(P):
        tok_abs = shard_tok_start + tok_local
        if tok_abs >= K.shape[0]:
            break  # pad remains 0.0
        frame_base = tok_local * D
        for h in range(n_heads):
            kvh = _kv_head_for_query(h, n_heads, n_kv_heads_in)
            h_base = frame_base + h * d_head
            k_slots[h_base:h_base + d_head] = K[tok_abs, kvh, :]
            v_slots[h_base:h_base + d_head] = V[tok_abs, kvh, :]
    return k_slots, v_slots


def pack_kv_dense_shards(K_t_h_d: np.ndarray, V_t_h_d: np.ndarray,
                         real_nt: int, P: int,
                         n_heads: int) -> tuple:
    """Pack the full K/V cache into a list of shard slot vectors.

    Args:
        K_t_h_d, V_t_h_d: [real_nt, n_kv_heads, d_head] (un-expanded).
        real_nt: number of valid token positions.
        P: positions_per_ct (power of 2).
        n_heads: number of query heads.

    Returns:
        (k_shards, v_shards): each a list of n_shards numpy arrays of length P*D.
        n_shards = ceil(real_nt / P).  Slots for tok >= real_nt are exact 0.0.
    """
    K = np.asarray(K_t_h_d, dtype=np.float64)
    V = np.asarray(V_t_h_d, dtype=np.float64)
    if K.ndim != 3 or V.ndim != 3:
        raise ValueError(
            f"pack_kv_dense_shards: expected [real_nt, n_kv_heads, d_head], "
            f"got K={K.shape} V={V.shape}"
        )
    if K.shape != V.shape:
        raise ValueError(f"pack_kv_dense_shards: K/V shape mismatch")
    real_nt_in, n_kv_heads, d_head = K.shape
    if real_nt_in != real_nt:
        raise ValueError(f"pack_kv_dense_shards: K dim0 ({real_nt_in}) != real_nt ({real_nt})")

    n_shards = math.ceil(real_nt / P)
    k_shards, v_shards = [], []
    for b in range(n_shards):
        ks, vs = pack_kv_shard(K, V, b * P, P, n_heads, n_kv_heads)
        k_shards.append(ks)
        v_shards.append(vs)
    return k_shards, v_shards


def unpack_kv_dense_shards(k_shards: list, v_shards: list,
                            real_nt: int, P: int,
                            n_heads: int, d_head: int) -> tuple:
    """Inverse of pack_kv_dense_shards.

    Returns (K, V) each [real_nt, n_heads, d_head] (GQA-expanded, float64).
    """
    D = n_heads * d_head
    K = np.zeros((real_nt, n_heads, d_head), dtype=np.float64)
    V = np.zeros((real_nt, n_heads, d_head), dtype=np.float64)
    for b, (ks, vs) in enumerate(zip(k_shards, v_shards)):
        for tok_local in range(P):
            tok_abs = b * P + tok_local
            if tok_abs >= real_nt:
                break
            frame_base = tok_local * D
            for h in range(n_heads):
                h_base = frame_base + h * d_head
                K[tok_abs, h, :] = ks[h_base:h_base + d_head]
                V[tok_abs, h, :] = vs[h_base:h_base + d_head]
    return K, V


# ---------------------------------------------------------------------------
# Score packing (token-major within a shard, base-slot only; replicated by
# inner_sum across the d_head block)
# ---------------------------------------------------------------------------

def pack_scores_shard(scores_t_h: np.ndarray, shard_tok_start: int,
                      P: int, n_heads: int, d_head: int) -> np.ndarray:
    """Pack one shard of scores into the score slot layout.

    After compute_qkt (inner_sum), the score for (tok_local, h) lives at
    slot tok_local*D + h*H and is replicated to all j in [0,H) by inner_sum's
    cyclic rotation. We pack only the base slot (j=0); the replication across
    j is already encoded in the inner_sum output.

    score_slot[tok_local * D + h * H] = scores[shard_tok_start + tok_local, h]
    Slot length: P * D.
    """
    D = n_heads * d_head
    slots = np.zeros(P * D, dtype=np.float64)
    for tok_local in range(P):
        tok_abs = shard_tok_start + tok_local
        if tok_abs >= scores_t_h.shape[0]:
            break
        for h in range(n_heads):
            slots[tok_local * D + h * d_head] = scores_t_h[tok_abs, h]
    return slots


def unpack_scores_shard(score_slots: np.ndarray, shard_tok_start: int,
                        P: int, n_heads: int, d_head: int,
                        total_nt: int) -> np.ndarray:
    """Recover scores[shard_tok_start:shard_tok_start+P, n_heads] from base slots."""
    D = n_heads * d_head
    n_tok = min(P, total_nt - shard_tok_start)
    out = np.zeros((n_tok, n_heads), dtype=np.float64)
    for tok_local in range(n_tok):
        for h in range(n_heads):
            out[tok_local, h] = score_slots[tok_local * D + h * d_head]
    return out


# ---------------------------------------------------------------------------
# dense_qkt: simulate compute_qkt on the slot vectors
# (src/attention.cu:21-25 + src/linear.cu:17-24)
# ---------------------------------------------------------------------------

def dense_qkt_shard(q_slots: np.ndarray, k_slots: np.ndarray,
                    n_heads: int, d_head: int, P: int,
                    inv_sqrt_d: float = None) -> np.ndarray:
    """Simulate compute_qkt for one shard on flat token-major slot vectors.

    Exact slot-level operations (NO reshape+matmul):
      prod = q_slots * k_slots               (elementwise)
      after_inner_sum = _inner_sum_slots(prod, d_head)
      score[tok_local, h] = after_inner_sum[tok_local*D + h*H]  (base slot readout)

    This proves the geometry: inner_sum over contiguous d_head slots at
    tok_local*D + h*H reduces sum_{j=0}^{H-1} q[tok*D+h*H+j]*k[tok*D+h*H+j].

    Returns scores_shard[P, n_heads] (float64). Padded tok positions give 0.
    """
    q = np.asarray(q_slots, dtype=np.float64)
    k = np.asarray(k_slots, dtype=np.float64)
    if inv_sqrt_d is None:
        inv_sqrt_d = 1.0 / np.sqrt(d_head)
    D = n_heads * d_head
    # Step 1: elementwise product (src/attention.cu:21)
    prod = q * k
    # Step 2: inner_sum over contiguous d_head blocks (src/linear.cu:17-24)
    after_sum = _inner_sum_slots(prod, d_head)
    # Step 3: readout at base slot of each (tok_local, head) block
    scores = np.zeros((P, n_heads), dtype=np.float64)
    for tok_local in range(P):
        for h in range(n_heads):
            base = tok_local * D + h * d_head
            scores[tok_local, h] = after_sum[base] * inv_sqrt_d
    return scores


def dense_qkt(q_slots_per_shard: list, k_shards: list,
              n_heads: int, d_head: int, real_nt: int, P: int,
              inv_sqrt_d: float = None) -> np.ndarray:
    """Full multi-shard dense_qkt -> scores[real_nt, n_heads].

    q_slots_per_shard: list of per-shard Q slot vectors (each length P*D),
        one per shard (same Q broadcast across tokens within each shard).
    k_shards: list of per-shard K slot vectors from pack_kv_dense_shards.
    """
    if inv_sqrt_d is None:
        inv_sqrt_d = 1.0 / np.sqrt(d_head)
    n_shards = len(k_shards)
    scores = np.zeros((real_nt, n_heads), dtype=np.float64)
    for b in range(n_shards):
        shard_scores = dense_qkt_shard(q_slots_per_shard[b], k_shards[b],
                                       n_heads, d_head, P, inv_sqrt_d)
        tok_start = b * P
        for tok_local in range(P):
            tok_abs = tok_start + tok_local
            if tok_abs >= real_nt:
                break
            scores[tok_abs, :] = shard_scores[tok_local, :]
    return scores


# ---------------------------------------------------------------------------
# dense_score_v: simulate score_times_v on the slot vectors
# (src/attention.cu:59-95)
# ---------------------------------------------------------------------------

def dense_score_v_shard(score_slots: np.ndarray, v_slots: np.ndarray,
                        n_heads: int, d_head: int, P: int) -> np.ndarray:
    """Simulate score_times_v for one shard on flat token-major slot vectors.

    Exact slot-level operations (NO reshape+matmul):
      Step 1 — mask: zero all slots except tok*D + h*H (base slots).
      Step 2 — broadcast: _broadcast_within_heads (negative-stride rotations).
      Step 3 — multiply by V.
      Step 4 — accumulate over P positions: _accumulate_over_positions.
      Readout: slot[0*D + h*H + j] = sum_{tok=0}^{P-1} w[tok,h]*V[tok,h,j].

    Returns out[n_heads, d_head] (the tok=0 frame after accumulation).
    """
    score = np.asarray(score_slots, dtype=np.float64)
    v = np.asarray(v_slots, dtype=np.float64)
    D = n_heads * d_head

    # Step 1: mask — keep only base slots (tok_local*D + h*H, j=0), zero the rest
    masked = np.zeros_like(score)
    for tok_local in range(P):
        for h in range(n_heads):
            base = tok_local * D + h * d_head
            masked[base] = score[base]

    # Step 2: broadcast w[tok,h] across the d_head block (src/attention.cu:66-72)
    masked = _broadcast_within_heads(masked, d_head)

    # Step 3: elementwise multiply by V
    prod = masked * v

    # Step 4: accumulate over P token positions by d_total strides (src/attention.cu:81-85)
    prod = _accumulate_over_positions(prod, P, D)

    # Readout: tok_local=0 frame holds sum over all positions
    out = np.zeros((n_heads, d_head), dtype=np.float64)
    for h in range(n_heads):
        h_base = h * d_head  # tok_local=0 offset is 0
        out[h, :] = prod[h_base:h_base + d_head]
    return out


def dense_score_v(score_shards: list, v_shards: list,
                  n_heads: int, d_head: int, P: int) -> np.ndarray:
    """Full multi-shard dense_score_v -> out[n_heads, d_head].

    Simulates cross-shard add (src/attention.cu:89-94).
    score_shards: list of per-shard score slot vectors (from pack_scores_shard).
    v_shards: list of per-shard V slot vectors (from pack_kv_dense_shards).
    """
    n_shards = len(v_shards)
    total = np.zeros((n_heads, d_head), dtype=np.float64)
    for b in range(n_shards):
        partial = dense_score_v_shard(score_shards[b], v_shards[b],
                                      n_heads, d_head, P)
        total += partial  # cross-shard add (src/attention.cu:89-94)
    return total
