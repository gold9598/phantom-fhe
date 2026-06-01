"""
RoPE (LLaMA-3) round-trip test.

This implementation uses the *interleaved* (pair-based) RoPE form, matching
lazuli's `apply_rope` in `lazuli/scripts/extract_llama_attn.py`:

    x_re = x[..., 0::2]; x_im = x[..., 1::2]
    rot_re = x_re*cos - x_im*sin
    rot_im = x_re*sin + x_im*cos
    out = stack([rot_re, rot_im], dim=-1).flatten(-2)

i.e., pairs are (slot 2i, slot 2i+1) within each d_head block. This matches
Meta's original LLaMA RoPE form (and lazuli's reference `q_post_rope.bin`),
not the HuggingFace split-half form.

FHE trick: 1 level + 2 rotations.
  q_plus  = rotate(q, +1)
  q_minus = rotate(q, -1)
  prod1 = q       * cos_pt          (cos(theta_i) at slot 2i and 2i+1)
  prod2 = q_plus  * sin_neg_even_pt (-sin at even slots, 0 at odd)
  prod3 = q_minus * sin_pos_odd_pt  (+sin at odd slots, 0 at even)
  result = rescale(prod1 + prod2 + prod3)

Two test cases:
  Case A: single-token Q at pos=8, compared against lazuli q_post_rope.bin.
          Q_pre = W_q @ input is computed in numpy, then encrypted (with
          a pre-scale 1/32 to fit |slot|<0.5 at matched-prime CKKS), RoPE
          applied in FHE, and the decrypted result post-multiplied by 32
          to compare against q_post_rope.bin.
  Case B: packed K at pos=0..7 with random K_pre values — compared against
          numpy interleaved-RoPE reference (skip lazuli k_cache.bin compare
          since we don't have inputs to reproduce it).
"""

import math
import os
import sys
import time

import numpy as np
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project/blocks")
from rope import apply_rope, build_rope_tables_packed, build_rope_tables_single, rope_required_steps


LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2
SCALE = 2.0 ** 40
SPARSE_HW = 128

D_HEAD = 128
N_HEADS = 32
D_TOTAL = N_HEADS * D_HEAD     # 4096
THETA_BASE = 500000.0

# RoPE only consumes 1 level. Use a small chain.
BITS = [60, 40, 40, 60]

TOL = 5e-3

LAZULI_DATA_DIR = os.path.expanduser("~/lazuli/tests/data/llama3_8b_layer0")


def numpy_apply_rope_interleaved(x, pos, d_head, theta_base):
    """x shape (..., d_head). Returns same shape, RoPE applied at position `pos`."""
    inv_freq = 1.0 / (theta_base ** (np.arange(0, d_head, 2, dtype=np.float64) / d_head))
    theta = pos * inv_freq                  # (d_head/2,)
    cos = np.cos(theta)                     # (d_head/2,)
    sin = np.sin(theta)
    x = np.asarray(x, dtype=np.float64)
    x_re = x[..., 0::2]
    x_im = x[..., 1::2]
    rot_re = x_re * cos - x_im * sin
    rot_im = x_re * sin + x_im * cos
    out = np.stack([rot_re, rot_im], axis=-1).reshape(*x.shape)
    return out


def setup_phantom():
    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, BITS))
    steps = rope_required_steps(D_HEAD)
    galois_elts = phantom.get_elts_from_steps(steps, N)
    params.set_galois_elts(galois_elts)

    context = phantom.context(params)
    sk = phantom.secret_key()
    sk.generate_sparse(context, SPARSE_HW)
    encoder = phantom.ckks_encoder(context)
    relin_key = sk.gen_relinkey(context)
    galois_key = sk.create_galois_keys(context)
    return context, sk, encoder, relin_key, galois_key, steps


def case_a_single_token(context, sk, encoder, relin_key, galois_key, steps):
    print("=" * 60)
    print("Case A: single-token Q at pos=8 vs lazuli q_post_rope.bin")
    print("=" * 60)

    pos = 8
    # Load reference data.
    W_q_path = os.path.join(LAZULI_DATA_DIR, "W_q.bin")
    input_path = os.path.join(LAZULI_DATA_DIR, "input.bin")
    qref_path = os.path.join(LAZULI_DATA_DIR, "q_post_rope.bin")

    W_q = np.fromfile(W_q_path, dtype=np.float64).reshape(D_TOTAL, -1)  # (d_total, d_model)
    d_model = W_q.shape[1]
    inp = np.fromfile(input_path, dtype=np.float64)                      # (d_model,)
    if inp.size != d_model:
        raise SystemExit(f"input.bin size {inp.size} != d_model {d_model}")
    q_ref = np.fromfile(qref_path, dtype=np.float64)                     # (d_total,)
    if q_ref.size != D_TOTAL:
        raise SystemExit(f"q_post_rope.bin size {q_ref.size} != d_total {D_TOTAL}")

    # Q_pre = W_q @ x  (shape d_total)
    q_pre = W_q @ inp
    q_pre_max = float(np.max(np.abs(q_pre)))
    print(f"d_model={d_model} d_head={D_HEAD} n_heads={N_HEADS} d_total={D_TOTAL} "
          f"#steps={len(steps)}")
    print(f"|Q_pre|_inf = {q_pre_max:.3f}")

    # Pre-scale to fit |slot|<0.5 at matched-prime CKKS.
    PRESCALE = 1.0 / 32.0
    q_scaled = q_pre * PRESCALE
    q_scaled_max = float(np.max(np.abs(q_scaled)))
    print(f"prescale=1/32  |Q_scaled|_inf = {q_scaled_max:.3f}")
    if q_scaled_max >= 0.5:
        raise SystemExit(f"Q_scaled exceeds 0.5; pick a larger divisor")

    # Encode Q replicated across d_total periods.
    periods = NUM_SLOTS // D_TOTAL
    q_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    for k in range(periods):
        q_slots[k * D_TOTAL:(k + 1) * D_TOTAL] = q_scaled

    chain_index = 1
    q_pt = encoder.encode_double_vector(context, q_slots.tolist(), SCALE, chain_index)
    q_ct = sk.encrypt_symmetric(context, q_pt)

    # Build RoPE tables. The cos/sin pt scale should match q's coeff modulus
    # last prime so that q.scale * pt.scale rescales cleanly back to nominal.
    # For pt at chain_index 1, the standard pattern (used by other ops) is to
    # encode the pt at the same SCALE as the ct, so ct*pt has scale SCALE^2 and
    # rescale_to_next divides by the last prime ~2^40 -> ~SCALE.
    tables = build_rope_tables_single(
        context, encoder, D_HEAD, D_TOTAL, pos, THETA_BASE,
        q_ct.chain_index(), SCALE)

    print(f"q.chain_index={q_ct.chain_index()} q.scale=2^{math.log2(q_ct.scale()):.1f}")

    t0 = time.perf_counter()
    out_ct = apply_rope(context, galois_key, q_ct, tables, D_HEAD)
    runtime = time.perf_counter() - t0
    print(f"out.chain_index={out_ct.chain_index()} out.scale=2^{math.log2(out_ct.scale()):.1f} "
          f"runtime={runtime:.3f}s")

    out_pt = sk.decrypt(context, out_ct)
    decoded = np.array(encoder.decode_double_vector(context, out_pt), dtype=np.float64)

    # Take slots [0..d_total) from period 0, undo prescale.
    fhe_q_rope = decoded[:D_TOTAL] / PRESCALE

    err = fhe_q_rope - q_ref
    max_err = float(np.max(np.abs(err)))
    rel_rms = float(np.sqrt(np.sum(err ** 2) / max(np.sum(q_ref ** 2), 1e-30)))
    print(f"max |err| = {max_err:.3e}  rel-RMS = {rel_rms:.3e}  tol = {TOL:.0e}")
    passed = rel_rms <= TOL
    print("PASS" if passed else "FAIL")
    return passed, max_err, rel_rms


def case_b_packed_k(context, sk, encoder, relin_key, galois_key, steps):
    print("=" * 60)
    print("Case B: packed K at pos=0..7 vs numpy reference")
    print("=" * 60)

    num_tokens = 8
    pos_start = 0

    rng = np.random.default_rng(0xC0FFEE)
    K_pre = rng.uniform(-0.4, 0.4, size=(num_tokens, N_HEADS, D_HEAD))

    # Pack K into slot[t*D_TOTAL + h*D_HEAD + i] = K_pre[t][h][i].
    if num_tokens * D_TOTAL > NUM_SLOTS:
        raise SystemExit("num_tokens * D_TOTAL exceeds NUM_SLOTS")
    k_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    for t in range(num_tokens):
        for h in range(N_HEADS):
            base = t * D_TOTAL + h * D_HEAD
            k_slots[base:base + D_HEAD] = K_pre[t][h]

    chain_index = 1
    k_pt = encoder.encode_double_vector(context, k_slots.tolist(), SCALE, chain_index)
    k_ct = sk.encrypt_symmetric(context, k_pt)

    tables = build_rope_tables_packed(
        context, encoder, D_HEAD, D_TOTAL, num_tokens, pos_start, THETA_BASE,
        k_ct.chain_index(), SCALE)

    print(f"d_head={D_HEAD} n_heads={N_HEADS} d_total={D_TOTAL} num_tokens={num_tokens} "
          f"#steps={len(steps)}")
    print(f"k.chain_index={k_ct.chain_index()} k.scale=2^{math.log2(k_ct.scale()):.1f}")

    t0 = time.perf_counter()
    out_ct = apply_rope(context, galois_key, k_ct, tables, D_HEAD)
    runtime = time.perf_counter() - t0
    print(f"out.chain_index={out_ct.chain_index()} out.scale=2^{math.log2(out_ct.scale()):.1f} "
          f"runtime={runtime:.3f}s")

    out_pt = sk.decrypt(context, out_ct)
    decoded = np.array(encoder.decode_double_vector(context, out_pt), dtype=np.float64)

    # Numpy reference: apply interleaved RoPE per-token, per-head.
    K_ref = np.zeros_like(K_pre)
    for t in range(num_tokens):
        K_ref[t] = numpy_apply_rope_interleaved(K_pre[t], pos_start + t, D_HEAD, THETA_BASE)

    # Compare slot-by-slot for the packed region [0, num_tokens*D_TOTAL).
    expected = np.zeros(num_tokens * D_TOTAL, dtype=np.float64)
    for t in range(num_tokens):
        for h in range(N_HEADS):
            base = t * D_TOTAL + h * D_HEAD
            expected[base:base + D_HEAD] = K_ref[t][h]

    actual = decoded[:num_tokens * D_TOTAL]
    err = actual - expected
    max_err = float(np.max(np.abs(err)))
    rel_rms = float(np.sqrt(np.sum(err ** 2) / max(np.sum(expected ** 2), 1e-30)))
    print(f"max |err| = {max_err:.3e}  rel-RMS = {rel_rms:.3e}  tol = {TOL:.0e}")
    passed = rel_rms <= TOL
    print("PASS" if passed else "FAIL")
    return passed, max_err, rel_rms


def main():
    context, sk, encoder, relin_key, galois_key, steps = setup_phantom()

    a_pass, a_max_err, a_rel_rms = case_a_single_token(
        context, sk, encoder, relin_key, galois_key, steps)
    b_pass, b_max_err, b_rel_rms = case_b_packed_k(
        context, sk, encoder, relin_key, galois_key, steps)

    print("=" * 60)
    print(f"Summary: A {'PASS' if a_pass else 'FAIL'} (rel-RMS {a_rel_rms:.3e}), "
          f"B {'PASS' if b_pass else 'FAIL'} (rel-RMS {b_rel_rms:.3e})")

    if not (a_pass and b_pass):
        raise SystemExit(1)


main()
