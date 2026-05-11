"""RoPE (LLaMA-3 interleaved form) plaintext-table construction and application.

Builds cos/sin plaintext tables and applies RoPE in 1 level + 2 rotations.
Python port of src/rope.cu; all CUDA ops dispatch through pyPhantom.
"""

import math
import sys

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom


def _is_power_of_two(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def _fill_head_block(
    cos_slots: np.ndarray,
    sin_neg_even_slots: np.ndarray,
    sin_pos_odd_slots: np.ndarray,
    base: int,
    d_head: int,
    pos: int,
    theta_base: float,
) -> None:
    """Fill one d_head-wide block at slot offset `base` for token position `pos`."""
    half = d_head // 2
    inv_dh = 1.0 / d_head
    for i in range(half):
        exponent = -2.0 * i * inv_dh
        inv_freq = theta_base ** exponent
        theta = pos * inv_freq
        c = math.cos(theta)
        s = math.sin(theta)
        even = base + 2 * i
        odd = base + 2 * i + 1
        cos_slots[even] = c
        cos_slots[odd] = c
        sin_neg_even_slots[even] = -s
        # sin_neg_even_slots[odd] stays 0
        # sin_pos_odd_slots[even] stays 0
        sin_pos_odd_slots[odd] = s


def _encode_pt(ctx, encoder, values: np.ndarray, chain_index: int, scale: float):
    """Encode a numpy float64 array as a PhantomPlaintext at given chain/scale."""
    return encoder.encode_double_vector(ctx, values, scale, chain_index)


class RopeTables:
    """Three PhantomPlaintexts needed to apply RoPE in one level."""
    __slots__ = ("cos_pt", "sin_neg_even_pt", "sin_pos_odd_pt")

    def __init__(self, cos_pt, sin_neg_even_pt, sin_pos_odd_pt):
        self.cos_pt = cos_pt
        self.sin_neg_even_pt = sin_neg_even_pt
        self.sin_pos_odd_pt = sin_pos_odd_pt


def rope_required_steps(d_head: int):
    """Returns the Galois rotation steps required by apply_rope: [-1, 1]."""
    if not _is_power_of_two(d_head) or d_head < 2:
        raise ValueError("rope_required_steps: d_head must be a power of 2 >= 2")
    return [-1, 1]


def build_rope_tables_single(
    ctx,
    encoder,
    d_head: int,
    d_total: int,
    pos: int,
    theta_base: float,
    chain_index: int,
    scale: float,
) -> RopeTables:
    """Build RoPE plaintexts for a single token at absolute position `pos`."""
    if not _is_power_of_two(d_head) or d_head < 2:
        raise ValueError("build_rope_tables_single: d_head must be a power of 2 >= 2")
    if d_total == 0 or d_total % d_head != 0:
        raise ValueError("build_rope_tables_single: d_total must be a positive multiple of d_head")

    num_slots = encoder.slot_count()
    if num_slots % d_total != 0:
        raise ValueError("build_rope_tables_single: num_slots must be a multiple of d_total")

    n_heads = d_total // d_head
    periods = num_slots // d_total

    cos_slots = np.zeros(num_slots, dtype=np.float64)
    sin_neg_even_slots = np.zeros(num_slots, dtype=np.float64)
    sin_pos_odd_slots = np.zeros(num_slots, dtype=np.float64)

    for k in range(periods):
        for h in range(n_heads):
            base = k * d_total + h * d_head
            _fill_head_block(cos_slots, sin_neg_even_slots, sin_pos_odd_slots,
                             base, d_head, pos, theta_base)

    return RopeTables(
        cos_pt=_encode_pt(ctx, encoder, cos_slots, chain_index, scale),
        sin_neg_even_pt=_encode_pt(ctx, encoder, sin_neg_even_slots, chain_index, scale),
        sin_pos_odd_pt=_encode_pt(ctx, encoder, sin_pos_odd_slots, chain_index, scale),
    )


def build_rope_tables_packed(
    ctx,
    encoder,
    d_head: int,
    d_total: int,
    num_tokens: int,
    pos_start: int,
    theta_base: float,
    chain_index: int,
    scale: float,
) -> RopeTables:
    """Build RoPE plaintexts for a packed multi-token layout (token t at pos_start+t)."""
    if not _is_power_of_two(d_head) or d_head < 2:
        raise ValueError("build_rope_tables_packed: d_head must be a power of 2 >= 2")
    if d_total == 0 or d_total % d_head != 0:
        raise ValueError("build_rope_tables_packed: d_total must be a positive multiple of d_head")
    if num_tokens == 0:
        raise ValueError("build_rope_tables_packed: num_tokens must be > 0")

    num_slots = encoder.slot_count()
    n_heads = d_total // d_head
    needed_slots = num_tokens * d_total
    if needed_slots > num_slots:
        raise ValueError("build_rope_tables_packed: num_tokens * d_total exceeds num_slots")

    cos_slots = np.zeros(num_slots, dtype=np.float64)
    sin_neg_even_slots = np.zeros(num_slots, dtype=np.float64)
    sin_pos_odd_slots = np.zeros(num_slots, dtype=np.float64)

    for t in range(num_tokens):
        pos = pos_start + t
        for h in range(n_heads):
            base = t * d_total + h * d_head
            _fill_head_block(cos_slots, sin_neg_even_slots, sin_pos_odd_slots,
                             base, d_head, pos, theta_base)

    return RopeTables(
        cos_pt=_encode_pt(ctx, encoder, cos_slots, chain_index, scale),
        sin_neg_even_pt=_encode_pt(ctx, encoder, sin_neg_even_slots, chain_index, scale),
        sin_pos_odd_pt=_encode_pt(ctx, encoder, sin_pos_odd_slots, chain_index, scale),
    )


def apply_rope(ctx, galois_key, q, tables: RopeTables, d_head: int = 0):
    """Apply RoPE to ciphertext `q` using precomputed tables. Consumes 1 level.

    `d_head` is accepted for API compatibility but unused (tables encode the pattern).
    """
    nominal = q.scale()

    q_plus = phantom.rotate(ctx, q, 1, galois_key)
    q_minus = phantom.rotate(ctx, q, -1, galois_key)

    prod1 = phantom.multiply_plain(ctx, q, tables.cos_pt)
    prod2 = phantom.multiply_plain(ctx, q_plus, tables.sin_neg_even_pt)
    prod3 = phantom.multiply_plain(ctx, q_minus, tables.sin_pos_odd_pt)

    result = phantom.add(ctx, prod1, prod2)
    result = phantom.add(ctx, result, prod3)

    result = phantom.rescale_to_next(ctx, result)
    result.set_scale(nominal)
    return result
