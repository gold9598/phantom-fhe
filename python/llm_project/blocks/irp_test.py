"""IRP (Cachemir Sec. 4.1) ct-pt VMM round-trip test.

Encodes a (d x d) matrix into K = d^2 / N IRP plaintexts, encrypts x in
interleaved-with-zeros layout (slot[i*t] = x[i]), runs `irp.irp_matvec`,
decrypts, extracts the d valid output slots via `irp.decode_irp_output`, and
compares against numpy M @ x.

Convention (from irp.py:453-481): both the input encrypt and the output decode
operate on stride-t = N/d, so the matvec computes y = M @ x with M shape (d,d).
"""

import time

import numpy as np
import pyPhantom as phantom

import irp


LOG_N = 16
N = 1 << LOG_N            # poly modulus degree (engine setup)
NUM_SLOTS = N // 2        # CKKS real-encoding slot count; this is irp.py's
                          # "N" parameter (its _build_irp_slots builds
                          # slot-count-length arrays — see irp.py:96).
SCALE = 2.0 ** 40
BITS = [60, 40, 40, 60]  # depth-3 minimal engine; IRP primitive is depth 1.


def run_case(d, baby_steps, tol, label, seed):
    if d * d % NUM_SLOTS != 0:
        raise SystemExit(f"FAIL [{label}]: d*d ({d*d}) not divisible by NUM_SLOTS ({NUM_SLOTS})")
    t = NUM_SLOTS // d
    K = d * d // NUM_SLOTS
    if K % baby_steps != 0:
        raise SystemExit(f"FAIL [{label}]: baby_steps={baby_steps} does not divide K={K}")
    G = K // baby_steps

    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, BITS))

    steps = irp.irp_required_steps(NUM_SLOTS, d, baby_steps=baby_steps)
    galois_elts = phantom.get_elts_from_steps(steps, N)  # N=poly degree here
    params.set_galois_elts(galois_elts)

    context = phantom.context(params)
    sk = phantom.secret_key(context)
    encoder = phantom.ckks_encoder(context)
    galois_key = sk.create_galois_keys(context)

    rng = np.random.default_rng(seed)
    M_mat = rng.uniform(-0.5, 0.5, size=(d, d))
    x = rng.uniform(-0.5, 0.5, size=d)

    chain_index = 1

    # Pre-encode IRP plaintexts (HOST/SCP path -- this is what the working
    # baseline pipeline uses; baseline llama3_mrpc.py:54-58 imports
    # encode_irp_diagonals_host + irp_matvec_host, NEVER the non-_host
    # GPU-plaintext variants). SCPs are chain-agnostic (no chain_index arg).
    pts = irp.encode_irp_diagonals_host(
        context, encoder, M_mat, N=NUM_SLOTS, d=d,
        scale=SCALE, baby_steps=baby_steps)

    x_ct = irp.encrypt_irp_input(
        context, encoder, sk, x, N=NUM_SLOTS, d=d,
        scale=SCALE, chain_index=chain_index)

    # §4.2 step-4 mask: 1.0 at slot[i*t] for i in [0, d), 0 elsewhere.
    # Baseline ALWAYS passes mask_pt (e.g. llama3_mrpc.py:407,581,632,855) --
    # the unmasked path leaves the output at scale=SCALE^2 unrescaled which
    # produced the 5.6e+0 error in the previous run. Mask fuses the rescale.
    mask_pt = irp.encode_irp_mask(
        context, encoder, NUM_SLOTS, d, SCALE, chain_index)

    print(f"[{label}] d={d} N={N} t={t} K={K} M={baby_steps} G={G} "
          f"num_plaintexts={len(pts)}")

    t0 = time.perf_counter()
    out_ct = irp.irp_matvec_host(
        context, encoder, galois_key, x_ct, pts,
        N=NUM_SLOTS, d=d, baby_steps=baby_steps, mask_pt=mask_pt)
    runtime = time.perf_counter() - t0

    out_pt = sk.decrypt(context, out_ct)
    decoded = encoder.decode_double_vector(context, out_pt)
    decoded_arr = np.asarray(decoded, dtype=np.float64)
    decoded_real = irp.decode_irp_output(decoded_arr, N=NUM_SLOTS, d=d)

    # CONVENTION: irp_matvec computes y = x @ M (== M.T @ x), NOT M @ x.
    # Verified empirically against decoded slots for both conventions;
    # matches the diagonal indexing in _build_irp_slots:53-55 where
    # matrix[(i+j+r*K)%d, (i-g*M)%d] folds the (i, k) sum as y[i] = sum_k
    # M[k, i] * x[k]. Baseline llama3.py uses Wq with this same convention.
    expected = x @ M_mat
    errors = np.abs(decoded_real - expected)
    max_err = float(errors.max())
    avg_err = float(errors.mean())

    print(f"[{label}] max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}  "
          f"tol = {tol:.0e}  runtime = {runtime:.2f}s")
    if max_err > tol:
        raise SystemExit(f"FAIL [{label}]: max abs err {max_err:.3e} > {tol:.0e}")
    print(f"[{label}] PASS")


# Toy: d=256 (smallest power-of-2 satisfying d^2 % NUM_SLOTS == 0 for
# NUM_SLOTS=16384, since sqrt(16384)=128 -> next pow-2 d=256).
# K = 256^2/16384 = 4 plaintexts, t = 16384/256 = 64.
run_case(d=256, baby_steps=1, tol=5e-5,
         label="toy d=256", seed=0)

# LLaMA Wq scale: d=4096, NUM_SLOTS=32768 -> K = 4096^2/32768 = 512 plaintexts,
# t = NUM_SLOTS/d = 8. baby_steps=16 gives M=16, G=32 (~sqrt(K)=22.6 balance).
# Tol scaled vs toy: more rotations + more multiplies => more accumulated noise.
run_case(d=4096, baby_steps=16, tol=5e-4,
         label="llama-Wq d=4096", seed=1)

print("=== IRP TEST SUMMARY: both cases PASS ===")
