"""
Replicate primitive round-trip test.

Encrypts a ct with values in slot[0..period) and zeros elsewhere, runs
`replicate` to broadcast that period-`period` block to fill all `num_slots`
slots, decrypts, and checks every replicated copy against the original.
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project/blocks")

import numpy as np
import pyPhantom as phantom
from linear import replicate_required_steps


LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2
SCALE = 2.0 ** 40
SPARSE_HW = 128

PERIOD = 64

# Replicate consumes 0 levels (rotations only): msg + 1 scale + special.
bits = [60, 40, 60]

params = phantom.params(phantom.scheme_type.ckks)
params.set_poly_modulus_degree(N)
params.set_special_modulus_size(1)
params.set_coeff_modulus(phantom.create_coeff_modulus(N, bits))

steps = replicate_required_steps(PERIOD, NUM_SLOTS)
galois_elts = phantom.get_elts_from_steps(steps, N)
params.set_galois_elts(galois_elts)

context = phantom.context(params)
sk = phantom.secret_key()
sk.generate_sparse(context, SPARSE_HW)
encoder = phantom.ckks_encoder(context)
galois_key = sk.create_galois_keys(context)

rng = random.Random(0xBEEFCAFE)
vals = [rng.uniform(-0.5, 0.5) for _ in range(PERIOD)]

slot_vec = [0.0] * NUM_SLOTS
for i in range(PERIOD):
    slot_vec[i] = vals[i]

chain_index = 1
pt = encoder.encode_double_vector(context, slot_vec, SCALE, chain_index=chain_index)
ct = sk.encrypt_symmetric(context, pt)

print(f"[replicate] period={PERIOD} num_slots={NUM_SLOTS} "
      f"required_steps={len(steps)} steps={steps}")
print(f"[replicate] in.chain_index={ct.chain_index()} "
      f"in.scale=2^{math.log2(ct.scale()):.1f}")

t0 = time.perf_counter()
out_ct = phantom.replicate(context, galois_key, ct, PERIOD, NUM_SLOTS)
runtime_ms = (time.perf_counter() - t0) * 1000.0

print(f"[replicate] out.chain_index={out_ct.chain_index()} "
      f"out.scale=2^{math.log2(out_ct.scale()):.1f}  runtime={runtime_ms:.2f} ms")

out_pt = sk.decrypt(context, out_ct)
decoded = encoder.decode_double_vector(context, out_pt)

decoded_arr = np.array(decoded[:NUM_SLOTS], dtype=np.float64)
expected = np.tile(np.array(vals, dtype=np.float64), NUM_SLOTS // PERIOD)
errors = np.abs(decoded_arr - expected)
max_err = float(errors.max())
avg_err = float(errors.mean())
argmax = int(errors.argmax())

TOL = 1e-4  # 9 chained rotations at 40-bit scale; matches qkt/score_v test bar.
print(f"[replicate] max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}  tol = {TOL:.0e}  "
      f"(worst slot={argmax}, k={argmax // PERIOD}, i={argmax % PERIOD})")

if max_err > TOL:
    raise SystemExit(f"FAIL: max abs err {max_err:.3e} > {TOL:.0e}")
print("PASS")
