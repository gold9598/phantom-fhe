"""
BSGS-diagonal matmul round-trip test.

Encodes a (num_rows x num_cols) matrix into d_pad BSGS diagonals,
encrypts x in replicated-block layout (period = d_pad), runs
`bsgs_matmul_preencoded`, decrypts, and compares slots [0..num_rows)
against numpy M @ x.
"""

import time

import numpy as np
import pyPhantom as phantom


LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2
SCALE = 2.0 ** 40
BITS = [60, 40, 40, 60]


def run_case(num_rows, num_cols, d_pad, baby_steps, tol, label, seed):
    giant_steps = d_pad // baby_steps
    assert baby_steps * giant_steps == d_pad

    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, BITS))

    steps = phantom.bsgs_required_steps(baby_steps)
    galois_elts = phantom.get_elts_from_steps(steps, N)
    params.set_galois_elts(galois_elts)

    context = phantom.context(params)
    sk = phantom.secret_key(context)
    encoder = phantom.ckks_encoder(context)
    galois_key = sk.create_galois_keys(context)

    rng = np.random.default_rng(seed)
    M = rng.uniform(-0.5, 0.5, size=(num_rows, num_cols))
    x = rng.uniform(-0.5, 0.5, size=num_cols)

    matrix_flat = M.flatten().tolist()
    diags = phantom.pre_encode_bsgs_diagonals(
        context, encoder, matrix_flat, num_rows, num_cols, d_pad, baby_steps, SCALE)

    periods = NUM_SLOTS // d_pad
    # Replicated-block layout: slot[k * d_pad + j] = x[j] for j in [0, num_cols)
    # (and 0 in pad slots [num_cols..d_pad)) for each period k in [0, periods).
    slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    for k in range(periods):
        slots[k * d_pad : k * d_pad + num_cols] = x
    chain_index = 1
    pt = encoder.encode_double_vector(context, slots.tolist(), SCALE, chain_index)
    x_ct = sk.encrypt_symmetric(context, pt)

    print(f"[{label}] d_pad={d_pad} M={baby_steps} G={giant_steps} "
          f"shape=({num_rows}x{num_cols}) num_slots={NUM_SLOTS} periods={periods}")

    t0 = time.perf_counter()
    out_ct = phantom.bsgs_matmul_preencoded(context, galois_key, x_ct, diags)
    runtime = time.perf_counter() - t0

    out_pt = sk.decrypt(context, out_ct)
    decoded = encoder.decode_double_vector(context, out_pt)
    decoded_real = np.array(decoded[:num_rows], dtype=np.float64)

    expected = M @ x
    errors = np.abs(decoded_real - expected)
    max_err = float(errors.max())
    avg_err = float(errors.mean())

    print(f"[{label}] max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}  "
          f"tol = {tol:.0e}  runtime = {runtime:.2f}s")
    if max_err > tol:
        raise SystemExit(f"FAIL [{label}]: max abs err {max_err:.3e} > {tol:.0e}")
    print(f"[{label}] PASS")


# Smoke: 256x256, d_pad=256, M=G=16. 128 periods over 32768 slots.
# Tolerance 2e-5: CKKS noise at 40-bit scale with 256 accumulations; avg err ~1e-6.
run_case(num_rows=256, num_cols=256, d_pad=256, baby_steps=16,
         tol=2e-5, label="smoke-256", seed=0xCAFE)

# LLaMA MLP scale: 4096x16384 (Up/Gate weight shape), d_pad=16384, M=G=128. 2 periods.
run_case(num_rows=4096, num_cols=16384, d_pad=16384, baby_steps=128,
         tol=1e-3, label="llama-up-4096x16384", seed=0xBEEF)

print("ALL PASS")
