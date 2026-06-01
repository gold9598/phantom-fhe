"""
RMSNorm round-trip test.

Two cases:
  - Smoke d_model=128
  - LLaMA-scale d_model=4096

Layout: x is replicated in period-d_model blocks across the slot vector.
"""

import math
import random
import sys
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project/blocks")

import numpy as np
import pyPhantom as phantom
from rmsnorm import rmsnorm_required_steps, setup_rmsnorm_weights, reference_rmsnorm


def run_case(d_model, label, seed, tol=1e-3):
    LOG_N = 16
    N = 1 << LOG_N
    NUM_SLOTS = N // 2
    SCALE = 2.0 ** 40
    SPARSE_HW = 128

    EPSILON = 1e-5
    # Band bracketed around E[x^2] = 1/12 ~= 0.0833 for x ~ U[-0.5, 0.5].
    # A tight 3:1 band keeps the deg-8 Chebyshev fit of z^(-1/2) under 1e-3.
    Z_MIN = 0.05
    Z_MAX = 0.15
    POLY_DEG = 8

    # Chain budget: 2 + ps_level_cost(8) + 2 = 9 levels.
    # 1 msg + 11 scale + 1 special leaves headroom from chain_index=1.
    bits = [60] + [40] * 11 + [60]

    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, bits))
    steps = rmsnorm_required_steps(d_model)
    galois_elts = phantom.get_elts_from_steps(steps, N)
    params.set_galois_elts(galois_elts)

    context = phantom.context(params)
    sk = phantom.secret_key()
    sk.generate_sparse(context, SPARSE_HW)
    encoder = phantom.ckks_encoder(context)
    relin_key = sk.gen_relinkey(context)
    galois_key = sk.create_galois_keys(context)

    rng = random.Random(seed)
    x = [rng.uniform(-0.5, 0.5) for _ in range(d_model)]
    g = [rng.uniform(0.5, 1.5) for _ in range(d_model)]

    # Tile x in period-d_model replicated blocks.
    periods = NUM_SLOTS // d_model
    x_full = [0.0] * NUM_SLOTS
    for k in range(periods):
        for j in range(d_model):
            x_full[k * d_model + j] = x[j]

    rms_params = phantom.rmsnorm_params()
    rms_params.d_model = d_model
    rms_params.epsilon = EPSILON
    rms_params.z_min = Z_MIN
    rms_params.z_max = Z_MAX
    rms_params.poly_degree = POLY_DEG

    weights = setup_rmsnorm_weights(context, encoder, rms_params, g)

    chain_index = 1
    pt = encoder.encode_double_vector(context, x_full, SCALE, chain_index=chain_index)
    ct = sk.encrypt_symmetric(context, pt)

    print(f"[{label}] d_model={d_model} ct.chain_index={ct.chain_index()} "
          f"ct.scale=2^{math.log2(ct.scale()):.1f}")

    out = phantom.rmsnorm_forward(
        context, encoder, relin_key, galois_key, ct, weights, rms_params)
    out_pt = sk.decrypt(context, out)
    decoded = encoder.decode_double_vector(context, out_pt)

    expected = reference_rmsnorm(x, g, d_model, EPSILON)
    expected = np.array(expected, dtype=np.float64)

    errors = np.abs(np.array(decoded[:d_model]) - expected)
    max_err = float(errors.max())
    avg_err = float(errors.mean())

    print(f"[{label}] out.chain_index={out.chain_index()} "
          f"out.scale=2^{math.log2(out.scale()):.1f}")
    print(f"[{label}] max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}  tol={tol:.0e}")
    if max_err > tol:
        raise SystemExit(f"FAIL [{label}]: max abs err {max_err:.3e} > {tol:.0e}")
    print(f"[{label}] PASS")


run_case(d_model=128, label="smoke-128", seed=0xCAFE0128)
run_case(d_model=4096, label="llama-4096", seed=0xCAFE4096)

print("ALL PASS")
