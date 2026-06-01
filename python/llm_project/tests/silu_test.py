"""
SiLU (deg-8 Chebyshev fit on [-2, 2]) round-trip test.
"""

import math
import random
import sys

import numpy as np
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project/blocks")
from silu import silu, SILU_COEFFS_DEG5_R2


LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2
SCALE_LOG2 = 40
SCALE = 2.0 ** SCALE_LOG2
SPARSE_HW = 128

bits = [60] + [40] * 7 + [60]

params = phantom.params(phantom.scheme_type.ckks)
params.set_poly_modulus_degree(N)
params.set_special_modulus_size(1)
params.set_coeff_modulus(phantom.create_coeff_modulus(N, bits))

context = phantom.context(params)
sk = phantom.secret_key()
sk.generate_sparse(context, SPARSE_HW)
encoder = phantom.ckks_encoder(context)
relin_key = sk.gen_relinkey(context)

rng = random.Random(0xCAFEBABE)
NUM_TEST = 1024
x_test = [rng.uniform(-2.0, 2.0) for _ in range(NUM_TEST)]
x_full = list(x_test) + [0.0] * (NUM_SLOTS - NUM_TEST)

chain_index = 1
pt = encoder.encode_double_vector(context, x_full, SCALE, chain_index=chain_index)
ct = sk.encrypt_symmetric(context, pt)

degree = 8
m = int(math.ceil(math.sqrt(degree + 1)))
l = (degree + 1 + m - 1) // m
depth = m + l - 1

print(f"[silu] degree={degree} m={m} l={l} depth={depth}  "
      f"ct.chain_index={ct.chain_index()} ct.scale=2^{math.log2(ct.scale()):.1f}")

out = silu(context, encoder, relin_key, ct)
out_pt = sk.decrypt(context, out)
decoded = encoder.decode_double_vector(context, out_pt)

x_arr = np.array(x_test, dtype=np.float64)
expected = x_arr / (1.0 + np.exp(-x_arr))

errors = np.abs(np.array(decoded[:NUM_TEST]) - expected)
max_err = float(errors.max())
avg_err = float(errors.mean())

print(f"[silu] out.chain_index={out.chain_index()} out.scale=2^{math.log2(out.scale()):.1f}")
print(f"[silu] max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}")

TOL = 5e-3  # deg-5 Cheb fit on [-2, 2] has L_inf 2.45e-3 (was deg-8: 2.7e-5)
if max_err > TOL:
    raise SystemExit(f"FAIL [silu]: max abs err {max_err:.3e} > {TOL:.0e}")
print(f"[silu] PASS (tol={TOL:.0e})")
