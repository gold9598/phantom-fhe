"""Cachemir KV-cache-aware ct-ct attention (Section 5 of arXiv:2602.11470).

Implements:
  * KVCache  : interleaved-replicated K cache + column-strided V cache.
  * qkt_irp  : QK^T over an IRP-style Q ciphertext and the K cache.
  * softmax_v_irp : Softmax x V using the V cache; output is in interleaved
                   layout, ready to feed into the downstream Wo IRP matvec.
  * kv_cache_required_steps : galois rotation steps needed by the KV ops.

Slot layouts (single head, d divides N, t = N/d):

  Q (input to qkt_irp):
      slot[c * t]            = q[c]      for c in [0, d)
      else                   = 0
      (matches the output of irp.irp_matvec.)

  K cache ciphertext at chunk index `chunk` holds tokens
  [chunk*t, chunk*t + 1, ..., chunk*t + t - 1] in interleaved layout:
      slot[r * t + p]        = K[chunk*t + p, r]   for r in [0, d), p in [0, t)
      slots beyond n' are zero.
  Total ciphertexts in K cache: ceil(n' / t).

  V cache uses d ciphertexts per chunk of N tokens. For ciphertext index
  d_idx in [0, d) within a chunk:
      slot[i]                = V[(i + d_idx * t) mod N + chunk_offset, i // t]
                               if (i + d_idx * t) mod N + chunk_offset < n'
      else                   = 0
  This makes columns contiguous in t-blocks (slot[c*t..c*t+t) all hold
  column-c entries) and makes Softmax x V a d-1 rotations + d ct-ct mults.

  Attention map output of qkt_irp:
      slot[m]                = (Q . K[m, :])      for m in [0, n')
      else                   = 0
  (one ciphertext, contiguous packing.)

  Output of softmax_v_irp:
      slot[c * t]            = Att[c]             for c in [0, d)
      else                   = 0
      (interleaved layout, ready to feed into Wo via IRP.)
"""

from typing import List, Optional

import numpy as np
import pyPhantom as phantom


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _check_dims(N: int, d: int) -> tuple:
    if not _is_pow2(N) or not _is_pow2(d):
        raise ValueError(f"kv_cache: N and d must be powers of 2, got N={N}, d={d}")
    if d > N:
        raise ValueError(f"kv_cache: d must be <= N, got d={d}, N={N}")
    t = N // d
    return t


# ---------------------------------------------------------------------------
# Plaintext slot encodings
# ---------------------------------------------------------------------------

def _encode_slots_pt(ctx, encoder, slots: np.ndarray, scale: float, chain_index: int):
    """Encode a float64 slot array as a plaintext at the given chain/scale."""
    return encoder.encode_double_vector(ctx, slots.tolist(), scale, chain_index)


def _encode_zero_pt(ctx, encoder, N: int, scale: float, chain_index: int):
    """Encode an all-zero plaintext at the given chain/scale."""
    return _encode_slots_pt(ctx, encoder, np.zeros(N, dtype=np.float64), scale, chain_index)


def _encode_mask_pt(ctx, encoder, N: int, positions, scale: float, chain_index: int):
    """Encode a binary mask plaintext: 1.0 at each position in `positions`, 0 elsewhere."""
    slots = np.zeros(N, dtype=np.float64)
    slots[positions] = 1.0
    return _encode_slots_pt(ctx, encoder, slots, scale, chain_index)


# ---------------------------------------------------------------------------
# Galois rotation steps
# ---------------------------------------------------------------------------

def kv_cache_required_steps(d: int, N: int, max_n: int) -> List[int]:
    """All galois rotation steps the KV-cache ops use, for sequence lengths
    up to `max_n` tokens.

    Steps used:
      - Q full-replicate (qkt preprocess):     -1, -2, ..., -(t/2)
      - QK partial-sum reduce (across r):       t, 2t, ..., (d/2)*t
      - QK pack across chunks (rot left by    chunk*t for chunk in [1, ceil(max_n/t)))
      - K append intra-chunk shift:            -1, -2, ..., -(t-1)        (t-1 steps)
      - SoftmaxxV softmax rotations:           t, 2t, ..., (d-1)*t
      - SoftmaxxV reduce within t-block:       1, 2, ..., t/2
      - V append rotations:                    All d-many slot lands; we cover
                                               1..(N-1) with the same set used
                                               by QK pack (multiples of t up to
                                               (d-1)*t are needed and already
                                               included; intra-chunk shifts
                                               -1..-(t-1) are also already in).
    """
    t = _check_dims(N, d)
    if max_n < 0:
        raise ValueError("kv_cache_required_steps: max_n must be >= 0")

    steps = set()

    # Q full-replicate: right rotations to spread q[i] across t slots.
    p = 1
    while p <= t // 2:
        steps.add(-p)
        p <<= 1

    # QK partial-sum across the r-dimension: rotate left by t, 2t, ..., (d/2)*t.
    if d > 1:
        s = t
        while s < d * t:
            steps.add(s)
            s <<= 1

    # QK pack across chunks: rotate right by chunk * t for chunk = 1 .. ceil(max_n/t)-1.
    num_chunks = (max_n + t - 1) // t
    for chunk in range(1, num_chunks):
        steps.add(-chunk * t)

    # K append + V append intra-chunk shift (right rotate by p, p = 1..min(max_n-1, t-1)).
    # Token n uses intra-chunk offset n mod t; for sequence 0..max_n-1 we use
    # offsets 0..min(max_n-1, t-1). Both K and V append cover the same set:
    # K uses negative steps; V uses both signs (mask offsets) but the magnitude
    # of every V offset is in [0, t-1] as well (i_d mod t == chunk_pos mod t).
    max_intra = min(max(0, max_n - 1), t - 1)
    for p in range(1, max_intra + 1):
        steps.add(-p)
        steps.add(p)

    # Softmax x V rotations: rotate softmax left by d_idx * t for d_idx = 1..d-1.
    for d_idx in range(1, d):
        steps.add(d_idx * t)

    # Softmax x V reduce within t-block: 1, 2, ..., t/2.
    p = 1
    while p <= t // 2:
        steps.add(p)
        p <<= 1

    return sorted(steps)


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------

class KVCache:
    """Single-head Cachemir KV cache (K: one ct per t-token chunk; V: d cts per
    N-token chunk).  Append APIs accept IRP-interleaved layout ciphertexts.
    """

    def __init__(self, ctx, encoder, gk, sk, d: int, N: int, scale: float):
        """Construct an empty KV cache (sk encrypts zero-seed chunks)."""
        self.ctx = ctx
        self.encoder = encoder
        self.gk = gk
        self.sk = sk
        self.d = int(d)
        self.N = int(N)
        self.t = _check_dims(N, d)
        self.scale = float(scale)
        # K cache: list of phantom ciphertexts, one per chunk of t tokens.
        self.k_cts: List = []
        # V cache: list of lists; v_cts[chunk_idx] has length d ciphertexts.
        self.v_cts: List[List] = []
        # Current sequence length n'.
        self.n = 0
        # Chain index of the most recent append; new appends should be at
        # the same level. The first append seeds it.
        self._chain_index: Optional[int] = None

    # ------------------------------------------------------------------
    # Append K
    # ------------------------------------------------------------------

    def append_k(self, k_ct_interleaved):
        """Append K vector (IRP-interleaved layout).  0 rotations for a fresh
        chunk; 1 right-rotation for a partially-filled chunk.
        """
        p = self.n % self.t
        ci = k_ct_interleaved.chain_index()
        if self._chain_index is None:
            self._chain_index = ci

        if p == 0:
            # Start a new chunk; the standard interleaved layout already places
            # token's data at slot[r*t] = K[n, r], which matches the chunk's
            # p=0 entries.
            self.k_cts.append(k_ct_interleaved)
        else:
            # Rotate right by p so K[n, r] lands at slot[r*t + p], then add
            # to the partially-filled last chunk.
            shifted = phantom.rotate(self.ctx, k_ct_interleaved, -int(p), self.gk)
            last = self.k_cts[-1]
            self.k_cts[-1] = phantom.add(self.ctx, last, shifted)

        # Note: V/K appends share an n increment; do it from the caller after
        # both have appended to keep n consistent.

    # ------------------------------------------------------------------
    # Append V
    # ------------------------------------------------------------------

    def append_v(self, v_ct_interleaved):
        """Append V vector (IRP-interleaved layout).  Scatters columns into
        d V-cache ciphertexts via mask + rotate + add per d_idx.
        """
        n = self.n
        chunk = n // self.N
        chunk_pos = n - chunk * self.N        # n mod N

        # Lazily allocate a fresh chunk: d zero ciphertexts at the same
        # chain index as the input.
        if chunk >= len(self.v_cts):
            ci = v_ct_interleaved.chain_index()
            zero_pt = _encode_zero_pt(self.ctx, self.encoder, self.N, self.scale, ci)
            self.v_cts.append([
                self.sk.encrypt_symmetric(self.ctx, zero_pt) for _ in range(self.d)
            ])

        # Scatter V[n] into the d ciphertexts of the current chunk.
        for d_idx in range(self.d):
            i_d = (chunk_pos - d_idx * self.t) % self.N
            col_d = i_d // self.t
            ci = v_ct_interleaved.chain_index()
            mask_pt = _encode_mask_pt(self.ctx, self.encoder, self.N,
                                      col_d * self.t, self.scale, ci)
            nominal = v_ct_interleaved.scale()
            extracted = phantom.multiply_plain(self.ctx, v_ct_interleaved, mask_pt)
            extracted = phantom.rescale_to_next(self.ctx, extracted)
            extracted.set_scale(nominal)

            # Rotate from slot col_d*t to slot i_d.
            offset = i_d - col_d * self.t       # always in [-(t-1), t-1]
            if offset != 0:
                # phantom.rotate positive = left = slot[j] <- slot[j+offset];
                # to move data from slot col_d*t to slot col_d*t + offset we need
                # right rotation, i.e., negative step.
                extracted = phantom.rotate(self.ctx, extracted, -int(offset), self.gk)

            # Match the chain of the existing V chunk (mask+rescale dropped 1
            # level on extracted; the running target may already be deeper if
            # earlier appends rescaled it). chain_index INCREASES as you go
            # deeper, so the shallower (lower chain_index) ct must drop to the
            # deeper (higher chain_index) one.
            target = self.v_cts[chunk][d_idx]
            tci = target.chain_index()
            eci = extracted.chain_index()
            if tci < eci:
                target = phantom.mod_switch_to(self.ctx, target, eci)
            elif tci > eci:
                extracted = phantom.mod_switch_to(self.ctx, extracted, tci)

            self.v_cts[chunk][d_idx] = phantom.add(self.ctx, target, extracted)

    def commit_token(self):
        """Increment n' after both append_k and append_v have been called for
        the current token. Call exactly once per appended token."""
        self.n += 1


# ---------------------------------------------------------------------------
# QK^T using IRP-style Q + interleaved K cache
# ---------------------------------------------------------------------------

def qkt_irp(ctx, encoder, gk, relin_key, q_ct_interleaved, kv_cache: "KVCache"):
    """Compute Q . K^T for IRP-style Q against the K cache (Section 5.1).

    q_ct_interleaved: slot[i*t] = q[i], zeros elsewhere.
    Returns ciphertext with slot[m] = q . K[m, :] for m in [0, n').
    """
    d = kv_cache.d
    N = kv_cache.N
    t = kv_cache.t
    n = kv_cache.n
    if n == 0:
        raise ValueError("qkt_irp: KV cache is empty; nothing to attend to")

    # ---- Step 1: full replicate Q (log(t) right-rotations + adds) ----
    q_pp = q_ct_interleaved
    p = 1
    while p <= t // 2:
        rot = phantom.rotate(ctx, q_pp, -int(p), gk)
        q_pp = phantom.add(ctx, q_pp, rot)
        p <<= 1

    num_chunks = (n + t - 1) // t
    nominal = q_ct_interleaved.scale()

    attn_ct = None

    for chunk in range(num_chunks):
        k_ct = kv_cache.k_cts[chunk]
        # Match q_pp's chain to k_ct (or vice versa). chain_index increases
        # as the ct goes deeper (consumes levels); mod_switch_to can only
        # drop to a HIGHER chain_index. So we lower the shallower (smaller
        # chain_index) ct to the deeper (larger chain_index) one.
        qci = q_pp.chain_index()
        kci = k_ct.chain_index()
        if qci < kci:
            q_use = phantom.mod_switch_to(ctx, q_pp, kci)
            k_use = k_ct
        elif qci > kci:
            q_use = q_pp
            k_use = phantom.mod_switch_to(ctx, k_ct, qci)
        else:
            q_use = q_pp
            k_use = k_ct

        # ---- Step 2a: ct-ct multiply ----
        prod = phantom.multiply_and_relin(ctx, q_use, k_use, relin_key)
        prod = phantom.rescale_to_next(ctx, prod)
        prod.set_scale(nominal)

        # ---- Step 2b: sum across r (left rotate by t, 2t, ..., (d/2)*t) ----
        s = t
        while s < d * t:
            rot = phantom.rotate(ctx, prod, int(s), gk)
            prod = phantom.add(ctx, prod, rot)
            s <<= 1

        # ---- Step 2c: mask to keep only slots [0..valid_t) then rotate ----
        mci = prod.chain_index()
        valid_t = min(t, n - chunk * t)
        mask_pt = _encode_mask_pt(ctx, encoder, N, np.arange(valid_t), nominal, mci)
        prod = phantom.multiply_plain(ctx, prod, mask_pt)
        prod = phantom.rescale_to_next(ctx, prod)
        prod.set_scale(nominal)

        if chunk > 0:
            # We want data at slots [0, t) (m_{chunk*t}..m_{chunk*t+t-1}) to
            # land at global slots [chunk*t, chunk*t+t). That requires a RIGHT
            # rotation by chunk*t (negative step in phantom).
            prod = phantom.rotate(ctx, prod, -int(chunk * t), gk)

        # ---- Step 2d: accumulate ----
        if attn_ct is None:
            attn_ct = prod
        else:
            attn_ct = phantom.add(ctx, attn_ct, prod)

    return attn_ct


# ---------------------------------------------------------------------------
# Softmax x V using V cache; output in interleaved layout
# ---------------------------------------------------------------------------

def softmax_v_irp(ctx, encoder, gk, relin_key, softmax_ct, kv_cache: "KVCache"):
    """Compute Softmax x V using the V cache (Section 5.2).

    softmax_ct: slot[m] = softmax_attn[m] for m in [0, n').
    Returns ciphertext with slot[c*t] = sum_m softmax[m] * V[m, c] in
    IRP-interleaved layout, ready to feed Wo.
    """
    d = kv_cache.d
    N = kv_cache.N
    t = kv_cache.t
    n = kv_cache.n
    if n == 0:
        raise ValueError("softmax_v_irp: KV cache is empty")

    nominal = softmax_ct.scale()
    out_ct = None

    for chunk_idx, v_chunk in enumerate(kv_cache.v_cts):
        # Build d rotated softmax copies, then ct-ct multiply with each ct_d.
        # Match chain: bring the shallower (lower chain_index) one down to the
        # deeper (higher chain_index) one.
        s_ci = softmax_ct.chain_index()
        v_ci = v_chunk[0].chain_index()
        if s_ci < v_ci:
            s_use_base = phantom.mod_switch_to(ctx, softmax_ct, v_ci)
        else:
            s_use_base = softmax_ct
        s_base_ci = s_use_base.chain_index()
        v_use = []
        for d_idx in range(d):
            vc = v_chunk[d_idx]
            ci = vc.chain_index()
            if ci < s_base_ci:
                vc = phantom.mod_switch_to(ctx, vc, s_base_ci)
            v_use.append(vc)

        chunk_acc = None
        for d_idx in range(d):
            if d_idx == 0:
                s_rot = s_use_base
            else:
                s_rot = phantom.rotate(ctx, s_use_base, int(d_idx * t), gk)
            prod = phantom.multiply_and_relin(ctx, s_rot, v_use[d_idx], relin_key)
            prod = phantom.rescale_to_next(ctx, prod)
            prod.set_scale(nominal)
            if chunk_acc is None:
                chunk_acc = prod
            else:
                chunk_acc = phantom.add(ctx, chunk_acc, prod)

        # Reduce within t-blocks: log(t) left-rotations by 1, 2, ..., t/2.
        p = 1
        while p <= t // 2:
            rot = phantom.rotate(ctx, chunk_acc, int(p), gk)
            chunk_acc = phantom.add(ctx, chunk_acc, rot)
            p <<= 1

        # Mask to keep only slot[c*t] for c in [0, d).
        mci = chunk_acc.chain_index()
        mask_pt = _encode_mask_pt(ctx, encoder, N,
                                  np.arange(d) * t, nominal, mci)
        chunk_acc = phantom.multiply_plain(ctx, chunk_acc, mask_pt)
        chunk_acc = phantom.rescale_to_next(ctx, chunk_acc)
        chunk_acc.set_scale(nominal)

        if out_ct is None:
            out_ct = chunk_acc
        else:
            out_ct = phantom.add(ctx, out_ct, chunk_acc)

    return out_ct
