"""
SwiGLU MLP forward round-trip test.

y = W_down @ ( silu(W_gate @ x) * (W_up @ x) )

Two cases (smoke + LLaMA scale). Both use logN=16, sparse_hw=128,
scale=2^40 with chain budget for: gate(1) + silu(4) + ct*ct(1) + down(1) = 7
plus headroom.
"""

import math
import sys
import time

import numpy as np
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project/blocks")
from mlp import setup_mlp_weights, mlp_required_steps, reference_mlp_forward


LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2
SCALE = 2.0 ** 40
SPARSE_HW = 128

# 1 msg + 10 scale + 1 special = 10 chain levels available from chain_index=1.
BITS = [60] + [40] * 10 + [60]


def run_case(label, d_model, d_hidden, d_pad, baby_steps, weight_lo, weight_hi,
             tol, seed):
    giant_steps = d_pad // baby_steps
    assert baby_steps * giant_steps == d_pad
    assert d_pad >= max(d_model, d_hidden)
    assert (d_pad & (d_pad - 1)) == 0
    assert NUM_SLOTS % d_pad == 0

    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, BITS))
    steps = mlp_required_steps(baby_steps)
    galois_elts = phantom.get_elts_from_steps(steps, N)
    params.set_galois_elts(galois_elts)

    context = phantom.context(params)
    sk = phantom.secret_key()
    sk.generate_sparse(context, SPARSE_HW)
    encoder = phantom.ckks_encoder(context)
    relin_key = sk.gen_relinkey(context)
    galois_key = sk.create_galois_keys(context)

    rng = np.random.default_rng(seed)
    Wgate = rng.uniform(weight_lo, weight_hi, size=(d_hidden, d_model))
    Wup = rng.uniform(weight_lo, weight_hi, size=(d_hidden, d_model))
    Wdown = rng.uniform(weight_lo, weight_hi, size=(d_model, d_hidden))
    x = rng.uniform(-0.5, 0.5, size=d_model)

    weights = setup_mlp_weights(
        context, encoder,
        Wgate.flatten().tolist(),
        Wup.flatten().tolist(),
        Wdown.flatten().tolist(),
        d_model, d_hidden, d_pad, baby_steps, SCALE)

    # Replicated-block layout: slot[k * d_pad + j] = x[j] for j in [0, d_model)
    # (and 0 in pad slots [d_model..d_pad)) for each period k.
    periods = NUM_SLOTS // d_pad
    slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    for k in range(periods):
        slots[k * d_pad : k * d_pad + d_model] = x
    chain_index = 1
    pt = encoder.encode_double_vector(context, slots.tolist(), SCALE, chain_index)
    x_ct = sk.encrypt_symmetric(context, pt)

    print(f"[{label}] d_model={d_model} d_hidden={d_hidden} d_pad={d_pad} "
          f"M={baby_steps} G={giant_steps}")
    print(f"[{label}] x.chain_index={x_ct.chain_index()} "
          f"x.scale=2^{math.log2(x_ct.scale()):.1f}")

    t0 = time.perf_counter()
    y_ct = phantom.mlp_forward(
        context, encoder, relin_key, galois_key, x_ct, weights)
    runtime = time.perf_counter() - t0

    y_pt = sk.decrypt(context, y_ct)
    decoded = encoder.decode_double_vector(context, y_pt)
    decoded_real = np.array(decoded[:d_model], dtype=np.float64)

    expected = np.array(
        reference_mlp_forward(
            x,
            Wgate,
            Wup,
            Wdown),
        dtype=np.float64)

    errors = np.abs(decoded_real - expected)
    max_err = float(errors.max())
    avg_err = float(errors.mean())
    rms_norm = float(np.linalg.norm(expected))
    rel_rms = float(np.linalg.norm(decoded_real - expected) / rms_norm) if rms_norm > 0 else float("nan")

    print(f"[{label}] y.chain_index={y_ct.chain_index()} "
          f"y.scale=2^{math.log2(y_ct.scale()):.1f}")
    print(f"[{label}] runtime={runtime:.2f}s  max|err|={max_err:.3e}  "
          f"avg|err|={avg_err:.3e}  rel_rms={rel_rms:.3e}  tol={tol:.0e}")

    if max_err > tol:
        raise SystemExit(f"FAIL [{label}]: max abs err {max_err:.3e} > {tol:.0e}")
    print(f"[{label}] PASS")


# ---- Smoke: d_model=128, d_hidden=512, d_pad=512, M=16, G=32 ----
run_case(
    label="smoke-128x512",
    d_model=128, d_hidden=512, d_pad=512, baby_steps=16,
    weight_lo=-0.1, weight_hi=0.1,
    tol=2e-2,  # bumped from 5e-3 after silu deg-5 substitution
    seed=0xCAFE0001,
)

# ---- LLaMA MLP scale: d_model=4096, d_hidden=14336, d_pad=16384, M=G=128 ----
run_case(
    label="llama-4096x14336",
    d_model=4096, d_hidden=14336, d_pad=16384, baby_steps=128,
    weight_lo=-0.05, weight_hi=0.05,
    tol=2e-2,
    seed=0xBEEF0001,
)

print("ALL PASS")
