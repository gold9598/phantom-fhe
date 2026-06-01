"""
Complex-folded SwiGLU MLP forward round-trip test.

Same target function as mlp_test.py:
    y = W_down @ ( silu(W_gate @ x) * (W_up @ x) )

But uses complex-slot folding to halve d_pad. The complex MLP path packs
two reals per CKKS slot via:
  - Row-fold on W_gate, W_up: (d_hidden x d_model) -> (d_hidden/2 x d_model)
    encoded as complex matrix M' with M'[i] = M[i] + i*M[i + d_hidden/2].
  - Col-fold-conjugate on W_down: (d_model x d_hidden) -> (d_model x d_hidden/2)
    encoded as M' with M'[i][j] = M[i][j] - i*M[i][j + d_hidden/2].

Cost relative to real MLP:
  + 4 levels for 2 BSGS x 2 extractions (each extract = +1 level: scalar *0.5
    after conjugate)
  + 2 levels for the -i twist on g_bot/u_bot before silu (both halves)
  + 1 level for re-pack (h_bot * +i so it occupies imag slot)
  - matmul cost halved (d_pad halved)

Chain budget: real MLP needs gate(1) + silu(5) + ct*ct(1) + down(1) = 8 levels.
Complex MLP adds +7 levels overhead -> 15 levels. We use BITS = [60] + [40]*22
+ [60] for plenty of headroom.
"""

import math
import sys
import time

import numpy as np
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project/blocks")
from mlp import setup_mlp_weights_complex, mlp_complex_required_steps, reference_mlp_forward


LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2
SCALE = 2.0 ** 40
SPARSE_HW = 128

# +5 levels of slack vs the 18 used by mlp_test.py to cover extract/repack.
BITS = [60] + [40] * 22 + [60]


def run_case(label, d_model, d_hidden, d_pad, baby_steps, weight_lo, weight_hi,
             tol, seed):
    giant_steps = d_pad // baby_steps
    assert baby_steps * giant_steps == d_pad, "baby * giant must equal d_pad"
    assert d_hidden % 2 == 0, "d_hidden must be even (rows folded into pairs)"
    folded_rows = d_hidden // 2
    folded_cols = d_hidden // 2  # for col-fold of W_down (d_model x d_hidden)
    assert d_pad >= max(folded_rows, d_model), "d_pad too small for row fold"
    assert d_pad >= max(d_model, folded_cols), "d_pad too small for col fold"
    assert (d_pad & (d_pad - 1)) == 0, "d_pad must be power of 2"
    assert NUM_SLOTS % d_pad == 0

    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, BITS))
    steps = mlp_complex_required_steps(baby_steps)
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

    weights = setup_mlp_weights_complex(
        context, encoder,
        Wgate.flatten().tolist(),
        Wup.flatten().tolist(),
        Wdown.flatten().tolist(),
        d_model, d_hidden, d_pad, baby_steps, SCALE)

    # Replicated-block layout, period d_pad.
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
    y_ct = phantom.mlp_forward_complex(
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
    return runtime


# ---- Smoke: d_model=128, d_hidden=512 ----
# d_hidden/2 = 256, d_pad must be >= max(256, 128) = 256.
run_case(
    label="smoke-128x512",
    d_model=128, d_hidden=512, d_pad=256, baby_steps=16,
    weight_lo=-0.1, weight_hi=0.1,
    tol=2e-2,
    seed=0xCAFE0001,
)

# ---- LLaMA scale: d_model=4096, d_hidden=14336, d_pad halved to 8192 ----
# d_hidden/2 = 7168 < 8192, d_model = 4096 < 8192 -- d_pad=8192 is enough.
run_case(
    label="llama-4096x14336",
    d_model=4096, d_hidden=14336, d_pad=8192, baby_steps=128,
    weight_lo=-0.05, weight_hi=0.05,
    tol=2e-2,
    seed=0xBEEF0001,
)

print("ALL PASS")
