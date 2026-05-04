"""
Softmax round-trip test (constant-T server-side init pipeline).
"""

import math
import random
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pyPhantom as phantom
from softmax import softmax_required_steps, reference_softmax


LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2
SCALE = 2.0 ** 40
SPARSE_HW = 128

NUM_TOKENS = 32
STRIDE = 1
NUM_SQUARINGS = 2
EXTRA_SCALE = 0.5
ITERS = 4

# Chain budget: ps_exp folded (deg=4; m=3,l=2,depth=3 -> 3 levels; the y=x/2^k
# scaling is folded into the polynomial coefficients) + squaring (NUM_SQUARINGS = 2)
# + softmax_correct (2*ITERS = 8) = 13 levels. 2 spare for headroom.
bits = [60] + [40] * 15 + [60]

params = phantom.params(phantom.scheme_type.ckks)
params.set_poly_modulus_degree(N)
params.set_special_modulus_size(1)
params.set_coeff_modulus(phantom.create_coeff_modulus(N, bits))

steps = softmax_required_steps(NUM_TOKENS, STRIDE)
galois_elts = phantom.get_elts_from_steps(steps, N)
params.set_galois_elts(galois_elts)

context = phantom.context(params)
sk = phantom.secret_key()
sk.generate_sparse(context, SPARSE_HW)
encoder = phantom.ckks_encoder(context)
relin_key = sk.gen_relinkey(context)
galois_key = sk.create_galois_keys(context)

rng = random.Random(0xC0FFEE32)
scores = [rng.uniform(-2.0, 2.0) for _ in range(NUM_TOKENS)]

# Tile scores into the slot vector at stride=1, repeating across periods.
scores_full = [0.0] * NUM_SLOTS
periods = NUM_SLOTS // NUM_TOKENS
for k in range(periods):
    for j in range(NUM_TOKENS):
        scores_full[k * NUM_TOKENS + j] = scores[j]

chain_index = 1
pt = encoder.encode_double_vector(context, scores_full, SCALE, chain_index=chain_index)
ct = sk.encrypt_symmetric(context, pt)

print(f"[softmax] num_tokens={NUM_TOKENS} num_squarings={NUM_SQUARINGS} iters={ITERS}  "
      f"ct.chain_index={ct.chain_index()} ct.scale=2^{math.log2(ct.scale()):.1f}")

# Stage A: ps_exp_init -> extra_scale * T^(-1/2^k) * exp(y)
e_ct = phantom.ps_exp_init(
    context, encoder, relin_key, ct,
    NUM_TOKENS, NUM_SQUARINGS, EXTRA_SCALE)
print(f"[softmax] post-ps_exp e_ct.chain_index={e_ct.chain_index()} "
      f"scale=2^{math.log2(e_ct.scale()):.1f}")

# Stage B: square k times -> extra_scale^(2^k) * exp(x) / T
phantom.square_iterations_inplace(context, relin_key, e_ct, NUM_SQUARINGS)
print(f"[softmax] post-square e_ct.chain_index={e_ct.chain_index()} "
      f"scale=2^{math.log2(e_ct.scale()):.1f}")

# Stage C: finalize via sum_reduce + Goldschmidt softmax_correct.
out = phantom.finalize_softmax(
    context, encoder, relin_key, galois_key, e_ct,
    NUM_TOKENS, STRIDE, ITERS)
print(f"[softmax] post-finalize out.chain_index={out.chain_index()} "
      f"scale=2^{math.log2(out.scale()):.1f}")

out_pt = sk.decrypt(context, out)
decoded = encoder.decode_double_vector(context, out_pt)

expected = reference_softmax(scores)
errors = np.abs(np.array(decoded[:NUM_TOKENS]) - expected)
max_err = float(errors.max())
avg_err = float(errors.mean())

print(f"[softmax] max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}")

# TOL relaxed from 1e-2 to 1.5e-2 to accommodate the deg-4 Chebyshev fit
# used in ps_exp_init (L_inf ≈ 2.7e-2 on [-2, 2] vs deg-5's 4.2e-3). The
# deg-4 fit saves 1 PS-depth level, and the y=x/2^k folding saves another;
# together they free 2 levels per attention block, enabling decoder pipeline
# to fit within CKKSEngine's bootstrap-friendly chain.
TOL = 1.5e-2
if max_err > TOL:
    raise SystemExit(f"FAIL [softmax]: max abs err {max_err:.3e} > {TOL:.0e}")
print(f"[softmax] PASS (tol={TOL:.0e})")
