"""
Paterson-Stockmeyer polynomial evaluation round-trip tests.

For each test case:
  - encrypt random x in [-0.4, 0.4] across all slots
  - run phantom.eval_polynomial
  - decrypt and compare to numpy's polyval reference
"""

import math
import random

import numpy as np
import pyPhantom as phantom


def ceil_sqrt(n):
    return int(math.ceil(math.sqrt(n)))


def run_case(coeffs, bits, scale_log2, label, tol=1e-3, hw=128, seed=0xCAFEBABE):
    LOG_N = 16
    N = 1 << LOG_N
    NUM_SLOTS = N // 2
    SCALE = 2.0 ** scale_log2

    d = len(coeffs) - 1
    m = ceil_sqrt(d + 1)
    l = (d + 1 + m - 1) // m
    depth = m + l - 1

    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, bits))

    context = phantom.context(params)
    sk = phantom.secret_key()
    sk.generate_sparse(context, hw)
    encoder = phantom.ckks_encoder(context)
    relin_key = sk.gen_relinkey(context)

    rng = random.Random(seed)
    x_vals = [rng.uniform(-0.4, 0.4) for _ in range(NUM_SLOTS)]

    chain_index = 1
    pt = encoder.encode_double_vector(context, x_vals, SCALE, chain_index=chain_index)
    ct = sk.encrypt_symmetric(context, pt)

    print(f"[{label}] degree={d} m={m} l={l} depth={depth}  "
          f"ct.chain_index={ct.chain_index()} ct.scale=2^{math.log2(ct.scale()):.1f}")

    out = phantom.eval_polynomial(context, encoder, relin_key, ct, coeffs)
    out_pt = sk.decrypt(context, out)
    decoded = encoder.decode_double_vector(context, out_pt)

    coeffs_desc = list(reversed(coeffs))
    expected = np.polyval(coeffs_desc, np.array(x_vals, dtype=np.float64))

    errors = np.abs(np.array(decoded) - expected)
    max_err = float(errors.max())
    avg_err = float(errors.mean())

    print(f"[{label}] out.chain_index={out.chain_index()} out.scale=2^{math.log2(out.scale()):.1f}")
    print(f"[{label}] max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}  tol = {tol:.0e}")
    if max_err > tol:
        raise SystemExit(f"FAIL [{label}]: max abs err {max_err:.3e} > {tol:.0e}")
    print(f"[{label}] PASS")


# ---- Degree 2: depth = m+l-1 = 2+1-1 = 2 ----
# bits = msg(60) + 3*scale(40) + special(60)
run_case(
    coeffs=[-0.2, 0.5, 0.3],
    bits=[60, 40, 40, 40, 60],
    scale_log2=40,
    label="deg2",
    tol=1e-3,
    seed=0xCAFE0002,
)

# ---- Degree 7: depth = m+l-1 = 3+3-1 = 5 ----
# Taylor expansion of sin(2πx)/(2π) up to x^7:
#   sin(2πx)/(2π) = x - (2π)^2 x^3 /6 + (2π)^4 x^5 /120 - (2π)^6 x^7 /5040
deg7_coeffs = [
    0.0,
    1.0,
    0.0,
    -((2 * math.pi) ** 2) / 6.0,
    0.0,
    ((2 * math.pi) ** 4) / 120.0,
    0.0,
    -((2 * math.pi) ** 6) / 5040.0,
]
run_case(
    coeffs=deg7_coeffs,
    bits=[60] + [40] * 7 + [60],
    scale_log2=40,
    label="deg7",
    tol=1e-3,
    seed=0xCAFE0007,
)

# ---- Degree 31: depth = m+l-1 = 6+6-1 = 11 ----
rng = random.Random(0xC0FFEE31)
deg31_coeffs = [rng.uniform(-1.0, 1.0) for _ in range(32)]
run_case(
    coeffs=deg31_coeffs,
    bits=[60] + [40] * 13 + [60],
    scale_log2=40,
    label="deg31",
    tol=1e-3,
    seed=0xCAFE0031,
)

print("ALL PASS")
