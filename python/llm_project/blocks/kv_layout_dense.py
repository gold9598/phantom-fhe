"""Dense per-head K/V packing — Stage 1 executable spec for the dense-layout rewrite.

PURE NUMPY. No phantom / FHE / GPU imports. This module is the executable
specification the later FHE stages are validated against: the slot-layout
contraction functions here reproduce, on the flat slot vectors, the SAME
mathematical reduction the generic C++ kernels perform
(``src/attention.cu`` ``compute_qkt`` / ``score_times_v``,
``src/linear.cu`` ``inner_sum``).

Decided layout (token-minor; chosen to match the C++ kernels' existing
contraction order — NOT to be redesigned here):

* Query (single token):
      ``q_slot[h*d_head + j] = Q[h, j]``                 (d_total dense slots)
* K / V cache (heads already GQA-expanded to n_heads):
      ``k_slot[h*d_head*nt_pad + j*nt_pad + tok] = K[tok, h, j]``
      ``v_slot[h*d_head*nt_pad + j*nt_pad + tok] = V[tok, h, j]``
  with ``nt_pad = next_pow2(real_nt)``; slots for ``tok in [real_nt, nt_pad)``
  (and all unused tail slots) are exact ``0.0``.
* Scores / exp:
      ``score_slot[h*nt_pad + tok] = scores[tok, h]``     (n_heads*nt_pad slots)
* Attention out (feeds Wo):
      ``o_slot[h*d_head + j] = sum_tok w[tok, h] * V[tok, h, j]``  (d_total dense)

GQA mapping (LLaMA-3.1-8B): K/V come from ``n_kv_heads`` heads and are expanded
to ``n_heads`` query heads. The canonical replication used throughout this
repo is block-repeat (``np.repeat(..., n_kv_groups, axis=1)``; see
``llama3.py:328-329``), i.e. query head ``h`` reads kv head
``h // n_kv_groups``. ``pack_kv_dense`` accepts the *un-expanded*
``[real_nt, n_kv_heads, d_head]`` tensors and performs that mapping itself.

Contraction semantics being reproduced (the whole point of Stage 1):

* ``compute_qkt`` does ``q (elementwise*) k`` then ``inner_sum(d_head)``.
  ``inner_sum`` (``src/linear.cu:9-25``) is a power-of-two tree-sum with
  strides ``1,2,4,...,d_head/2`` (left rotation), so the value read at a
  ``d_head``-aligned base slot is ``sum_{j=0}^{d_head-1} q[base+j]*k[base+j]``.
  Mathematically this is ``sum_j Q[h,j] * K[tok,h,j]``. The 1/sqrt(d_head)
  softmax scale is applied here so ``dense_qkt`` returns the scaled scores.
* ``score_times_v`` masks the score, broadcasts it across the d_head block,
  multiplies by V, then tree-accumulates across token positions by d_total
  strides. Mathematically this is ``sum_tok w[tok,h] * V[tok,h,j]``.

Floating-point note: the C++ kernels accumulate via a balanced
power-of-two tree while these references use ``np.einsum`` (pairwise/serial).
Real summation is associative; the reorderings differ only at the ~1e-15
rounding level, far below the 1e-12 gate the tests assert.
"""

import numpy as np


def next_pow2(n: int) -> int:
    """Smallest power of two >= n (>=1). next_pow2(0)=1, next_pow2(1)=1."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def _kv_head_for_query(h: int, n_heads: int, n_kv_heads: int) -> int:
    """Map query head -> kv head using the repo's block-repeat GQA scheme.

    Matches ``np.repeat(K, n_kv_groups, axis=1)`` (llama3.py:328): query
    heads ``[g*n_kv_groups, (g+1)*n_kv_groups)`` all read kv head ``g``.
    """
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )
    n_kv_groups = n_heads // n_kv_heads
    return h // n_kv_groups


def pack_q_dense(Q_h_d: np.ndarray) -> np.ndarray:
    """Pack a single-token query ``Q[n_heads, d_head]`` into dense slots.

    Layout: ``q_slot[h*d_head + j] = Q[h, j]``. Output length is
    ``n_heads * d_head`` (== d_total). float64.
    """
    Q = np.asarray(Q_h_d, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError(f"pack_q_dense: expected [n_heads, d_head], got {Q.shape}")
    n_heads, d_head = Q.shape
    slots = np.zeros(n_heads * d_head, dtype=np.float64)
    for h in range(n_heads):
        base = h * d_head
        slots[base:base + d_head] = Q[h, :]
    return slots


def unpack_q_dense(q_slots: np.ndarray, n_heads: int, d_head: int) -> np.ndarray:
    """Inverse of :func:`pack_q_dense` -> ``Q[n_heads, d_head]`` (float64)."""
    q = np.asarray(q_slots, dtype=np.float64)
    if q.shape[0] < n_heads * d_head:
        raise ValueError(
            f"unpack_q_dense: need >= {n_heads * d_head} slots, got {q.shape[0]}"
        )
    Q = np.zeros((n_heads, d_head), dtype=np.float64)
    for h in range(n_heads):
        base = h * d_head
        Q[h, :] = q[base:base + d_head]
    return Q


def pack_kv_dense(K_t_h_d: np.ndarray, V_t_h_d: np.ndarray,
                  real_nt: int, nt_pad: int,
                  n_heads: int):
    """Pack the K/V cache into dense token-minor slot vectors.

    Args:
        K_t_h_d, V_t_h_d: ``[real_nt, n_kv_heads, d_head]`` (un-expanded;
            GQA replication is performed here).
        real_nt: number of valid token positions.
        nt_pad: padded token dimension; must be a power of two and
            ``>= real_nt``. Slots for ``tok in [real_nt, nt_pad)`` are 0.0.
        n_heads: number of query heads (K/V get block-repeated up to this).

    Returns:
        (k_slots, v_slots), each a float64 array of length
        ``n_heads * d_head * nt_pad``, laid out as
        ``slot[h*d_head*nt_pad + j*nt_pad + tok]``.
    """
    K = np.asarray(K_t_h_d, dtype=np.float64)
    V = np.asarray(V_t_h_d, dtype=np.float64)
    if K.ndim != 3 or V.ndim != 3:
        raise ValueError(
            f"pack_kv_dense: expected [real_nt, n_kv_heads, d_head], "
            f"got K={K.shape} V={V.shape}"
        )
    if K.shape != V.shape:
        raise ValueError(f"pack_kv_dense: K/V shape mismatch {K.shape} vs {V.shape}")
    t_in, n_kv_heads, d_head = K.shape
    if t_in != real_nt:
        raise ValueError(
            f"pack_kv_dense: K first dim ({t_in}) != real_nt ({real_nt})"
        )
    if nt_pad < real_nt:
        raise ValueError(
            f"pack_kv_dense: nt_pad ({nt_pad}) < real_nt ({real_nt})"
        )
    if nt_pad < 1 or (nt_pad & (nt_pad - 1)) != 0:
        raise ValueError(f"pack_kv_dense: nt_pad ({nt_pad}) must be a power of 2")

    head_stride = d_head * nt_pad
    total = n_heads * head_stride
    k_slots = np.zeros(total, dtype=np.float64)
    v_slots = np.zeros(total, dtype=np.float64)

    for h in range(n_heads):
        kvh = _kv_head_for_query(h, n_heads, n_kv_heads)
        h_base = h * head_stride
        for j in range(d_head):
            j_base = h_base + j * nt_pad
            # tail [real_nt, nt_pad) stays exactly 0.0 (np.zeros init)
            k_slots[j_base:j_base + real_nt] = K[:real_nt, kvh, j]
            v_slots[j_base:j_base + real_nt] = V[:real_nt, kvh, j]
    return k_slots, v_slots


def unpack_kv_dense(k_slots: np.ndarray, v_slots: np.ndarray,
                    real_nt: int, nt_pad: int,
                    n_heads: int, d_head: int):
    """Inverse of :func:`pack_kv_dense`.

    Returns ``(K, V)`` each ``[real_nt, n_heads, d_head]`` (GQA-expanded,
    float64) — the round-trip reconstructs the *expanded* tensors, which
    are bit-identical to ``np.repeat`` of the inputs.
    """
    k = np.asarray(k_slots, dtype=np.float64)
    v = np.asarray(v_slots, dtype=np.float64)
    head_stride = d_head * nt_pad
    need = n_heads * head_stride
    if k.shape[0] < need or v.shape[0] < need:
        raise ValueError(
            f"unpack_kv_dense: need >= {need} slots, got K={k.shape[0]} V={v.shape[0]}"
        )
    K = np.zeros((real_nt, n_heads, d_head), dtype=np.float64)
    V = np.zeros((real_nt, n_heads, d_head), dtype=np.float64)
    for h in range(n_heads):
        h_base = h * head_stride
        for j in range(d_head):
            j_base = h_base + j * nt_pad
            K[:, h, j] = k[j_base:j_base + real_nt]
            V[:, h, j] = v[j_base:j_base + real_nt]
    return K, V


def dense_qkt(q_slots: np.ndarray, k_slots: np.ndarray,
              n_heads: int, d_head: int, real_nt: int, nt_pad: int,
              inv_sqrt_d: float = None) -> np.ndarray:
    """Reproduce ``compute_qkt`` on the dense slot layout.

    For each head ``h`` and token ``tok``:
        ``scores[tok, h] = inv_sqrt_d * sum_{j} q_slot[h*d_head+j]
                                              * k_slot[h*d_head*nt_pad + j*nt_pad + tok]``
    which is the ``q (.) k`` + ``inner_sum(d_head)`` reduction of
    ``compute_qkt`` / ``inner_sum``. ``inv_sqrt_d`` defaults to
    ``1/sqrt(d_head)`` (the softmax scale applied at the QKT stage).

    Returns ``scores[real_nt, n_heads]`` (float64). Token positions in
    ``[real_nt, nt_pad)`` are excluded (their k slots are 0 anyway).
    """
    q = np.asarray(q_slots, dtype=np.float64)
    k = np.asarray(k_slots, dtype=np.float64)
    if inv_sqrt_d is None:
        inv_sqrt_d = 1.0 / np.sqrt(d_head)
    head_stride = d_head * nt_pad
    scores = np.zeros((real_nt, n_heads), dtype=np.float64)
    for h in range(n_heads):
        qh = q[h * d_head:(h + 1) * d_head]                  # [d_head]
        h_base = h * head_stride
        # kh[j, tok] = K[tok,h,j]; restrict to valid tokens
        kh = k[h_base:h_base + head_stride].reshape(d_head, nt_pad)[:, :real_nt]
        # sum_j q[j] * K[tok,h,j]  -> [real_nt]
        scores[:, h] = (qh @ kh) * inv_sqrt_d
    return scores


def dense_score_v(w_slots: np.ndarray, v_slots: np.ndarray,
                  n_heads: int, d_head: int, real_nt: int, nt_pad: int) -> np.ndarray:
    """Reproduce ``score_times_v`` on the dense slot layout.

    For each head ``h`` and dim ``j``:
        ``out[h, j] = sum_{tok} w_slot[h*nt_pad+tok]
                                 * v_slot[h*d_head*nt_pad + j*nt_pad + tok]``
    i.e. the mask + broadcast + V-multiply + token-accumulate of
    ``score_times_v``.

    Args:
        w_slots: softmax weights in score layout
            ``w_slot[h*nt_pad + tok] = w[tok, h]`` (length n_heads*nt_pad).
        v_slots: V in the :func:`pack_kv_dense` layout.

    Returns ``out[n_heads, d_head]`` (float64). Tokens in
    ``[real_nt, nt_pad)`` contribute 0 (their v / w slots are 0).
    """
    w = np.asarray(w_slots, dtype=np.float64)
    v = np.asarray(v_slots, dtype=np.float64)
    head_stride = d_head * nt_pad
    out = np.zeros((n_heads, d_head), dtype=np.float64)
    for h in range(n_heads):
        wh = w[h * nt_pad:h * nt_pad + nt_pad][:real_nt]      # [real_nt]
        h_base = h * head_stride
        vh = v[h_base:h_base + head_stride].reshape(d_head, nt_pad)[:, :real_nt]
        # sum_tok w[tok] * V[tok,h,j]  -> [d_head]
        out[h, :] = vh @ wh
    return out


def pack_scores_dense(scores_t_h: np.ndarray, real_nt: int, nt_pad: int,
                      n_heads: int) -> np.ndarray:
    """Pack ``scores[real_nt, n_heads]`` into the score slot layout.

    ``score_slot[h*nt_pad + tok] = scores[tok, h]``; tail
    ``[real_nt, nt_pad)`` is exact 0.0. Length ``n_heads * nt_pad``.
    """
    s = np.asarray(scores_t_h, dtype=np.float64)
    if s.shape != (real_nt, n_heads):
        raise ValueError(
            f"pack_scores_dense: expected [{real_nt}, {n_heads}], got {s.shape}"
        )
    slots = np.zeros(n_heads * nt_pad, dtype=np.float64)
    for h in range(n_heads):
        slots[h * nt_pad:h * nt_pad + real_nt] = s[:, h]
    return slots


def unpack_scores_dense(score_slots: np.ndarray, real_nt: int, nt_pad: int,
                        n_heads: int) -> np.ndarray:
    """Inverse of :func:`pack_scores_dense` -> ``scores[real_nt, n_heads]``."""
    s = np.asarray(score_slots, dtype=np.float64)
    need = n_heads * nt_pad
    if s.shape[0] < need:
        raise ValueError(
            f"unpack_scores_dense: need >= {need} slots, got {s.shape[0]}"
        )
    out = np.zeros((real_nt, n_heads), dtype=np.float64)
    for h in range(n_heads):
        out[:, h] = s[h * nt_pad:h * nt_pad + real_nt]
    return out
