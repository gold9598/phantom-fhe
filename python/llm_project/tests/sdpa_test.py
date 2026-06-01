"""
Scaled dot-product attention end-to-end round-trip test.

Pipeline:
  1. compute_qkt -> scores at slot[t*d_total + h*d_head].
  2. mask + scale by 1/sqrt(d_head).
  3. broadcast within d_head blocks.
  4. softmax over the t-axis with stride d_total.
  5. score_times_v -> attention output at slot[h*d_head + i].

Output: attention_out = softmax(Q·K^T / sqrt(d_head)) · V.
"""

import math
import os
import sys
import time

import numpy as np
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project/blocks")
from attention import scaled_dot_product_attention, sdpa_required_steps


LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2
SCALE = 2.0 ** 40
SPARSE_HW = 128

D_HEAD = 32
N_HEADS = 4
D_TOTAL = N_HEADS * D_HEAD
NUM_TOKENS = 4

TOL = 5e-2

# Chain budget: 1 msg + 26 scale + 1 special. Reduced from 28 after deg-4 +
# folded ps_exp_init optimizations save 2 levels in softmax (was depth 5,
# now depth 3).
BITS = [60] + [40] * 26 + [60]


def main():
    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, BITS))
    steps = sdpa_required_steps(D_HEAD, D_TOTAL, NUM_TOKENS, NUM_SLOTS)
    galois_elts = phantom.get_elts_from_steps(steps, N)
    params.set_galois_elts(galois_elts)

    context = phantom.context(params)
    sk = phantom.secret_key()
    sk.generate_sparse(context, SPARSE_HW)
    encoder = phantom.ckks_encoder(context)
    relin_key = sk.gen_relinkey(context)
    galois_key = sk.create_galois_keys(context)

    rng = np.random.default_rng(0xCAFEEAFE)
    Q = rng.uniform(-0.5, 0.5, size=(N_HEADS, D_HEAD))
    K = rng.uniform(-0.5, 0.5, size=(NUM_TOKENS, N_HEADS, D_HEAD))
    V = rng.uniform(-0.5, 0.5, size=(NUM_TOKENS, N_HEADS, D_HEAD))

    # Q replicated: slot[k*D_TOTAL + h*D_HEAD + i] = Q[h][i] for every k.
    periods = NUM_SLOTS // D_TOTAL
    q_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    for k in range(periods):
        for h in range(N_HEADS):
            base = k * D_TOTAL + h * D_HEAD
            q_slots[base:base + D_HEAD] = Q[h]

    # Packed K: slot[t*D_TOTAL + h*D_HEAD + i] = K[t][h][i], zeros for t >= NUM_TOKENS.
    k_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    for t in range(NUM_TOKENS):
        for h in range(N_HEADS):
            base = t * D_TOTAL + h * D_HEAD
            k_slots[base:base + D_HEAD] = K[t][h]

    # Packed V: same layout as K.
    v_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    for t in range(NUM_TOKENS):
        for h in range(N_HEADS):
            base = t * D_TOTAL + h * D_HEAD
            v_slots[base:base + D_HEAD] = V[t][h]

    chain_index = 1
    q_pt = encoder.encode_double_vector(context, q_slots.tolist(), SCALE, chain_index)
    q_ct = sk.encrypt_symmetric(context, q_pt)
    k_pt = encoder.encode_double_vector(context, k_slots.tolist(), SCALE, chain_index)
    k_ct = sk.encrypt_symmetric(context, k_pt)
    v_pt = encoder.encode_double_vector(context, v_slots.tolist(), SCALE, chain_index)
    v_ct = sk.encrypt_symmetric(context, v_pt)

    print(f"d_head={D_HEAD} n_heads={N_HEADS} d_total={D_TOTAL} num_tokens={NUM_TOKENS} "
          f"num_slots={NUM_SLOTS} #steps={len(steps)}")
    print(f"q.chain_index={q_ct.chain_index()} q.scale=2^{math.log2(q_ct.scale()):.1f}")

    t0 = time.perf_counter()
    out_ct = scaled_dot_product_attention(
        context, encoder, relin_key, galois_key,
        q_ct, k_ct, v_ct,
        D_HEAD, N_HEADS, NUM_TOKENS)
    runtime = time.perf_counter() - t0

    print(f"out.chain_index={out_ct.chain_index()} out.scale=2^{math.log2(out_ct.scale()):.1f}")

    out_pt = sk.decrypt(context, out_ct)
    decoded = np.array(encoder.decode_double_vector(context, out_pt), dtype=np.float64)

    # Reference: per (h, i): scores = Q[h]·K[t][h] / sqrt(d_head); weights = softmax(scores);
    # out[h][i] = Σ_t weights[t] * V[t][h][i].
    inv_sqrt_d = 1.0 / math.sqrt(D_HEAD)
    expected = np.zeros((N_HEADS, D_HEAD), dtype=np.float64)
    for h in range(N_HEADS):
        scores_h = np.array([float((Q[h] * K[t][h]).sum()) * inv_sqrt_d
                             for t in range(NUM_TOKENS)])
        m = scores_h.max()
        ex = np.exp(scores_h - m)
        weights_h = ex / ex.sum()
        for i in range(D_HEAD):
            expected[h, i] = float(np.sum(weights_h * V[:, h, i]))

    errors = []
    sq_errs = []
    sq_refs = []
    for h in range(N_HEADS):
        for i in range(D_HEAD):
            ref = expected[h, i]
            got = float(decoded[h * D_HEAD + i])
            err = got - ref
            errors.append(abs(err))
            sq_errs.append(err * err)
            sq_refs.append(ref * ref)

    errors = np.array(errors)
    max_err = float(errors.max())
    avg_err = float(errors.mean())
    rel_rms = float(math.sqrt(sum(sq_errs) / max(sum(sq_refs), 1e-30)))

    print(f"max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}  "
          f"rel-RMS = {rel_rms:.3e}  tol = {TOL:.0e}  runtime = {runtime:.3f}s  "
          f"positions = {len(errors)}")
    if max_err > TOL:
        raise SystemExit(f"FAIL: max abs err {max_err:.3e} > {TOL:.0e}")
    print("PASS")


main()
