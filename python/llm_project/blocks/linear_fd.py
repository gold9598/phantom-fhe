"""
FD-packed matrix-vector multiply round-trip test.

Encodes a matrix M (r x d) using feature-dimension packing, encrypts a vector v
(length d), runs `multiply_matrix_vector_fd`, decrypts, and compares slots
[0..r) against the numpy reference M @ v.
"""

import math
import random
import sys
import os

import numpy as np
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyPhantom as phantom
from linear import (
    encode_matrix_fd,
    encrypt_vector_fd,
    matvec_fd_required_steps,
    multiply_matrix_vector_fd,
)


def run_case(num_rows, num_cols, tol, label, seed):
    LOG_N = 16
    N = 1 << LOG_N
    NUM_SLOTS = N // 2
    SCALE = 2.0 ** 40

    bits = [60, 40, 40, 40, 40, 60]

    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, bits))

    cols_per_chunk = NUM_SLOTS // num_rows
    steps = matvec_fd_required_steps(num_rows, cols_per_chunk)
    galois_elts = phantom.get_elts_from_steps(steps, N)
    params.set_galois_elts(galois_elts)

    context = phantom.context(params)
    sk = phantom.secret_key(context)
    encoder = phantom.ckks_encoder(context)
    galois_key = sk.create_galois_keys(context)

    rng = random.Random(seed)
    matrix = [rng.uniform(-0.5, 0.5) for _ in range(num_rows * num_cols)]
    vector = [rng.uniform(-0.5, 0.5) for _ in range(num_cols)]

    chain_index = 1
    enc_matrix = encode_matrix_fd(
        context, encoder, matrix, num_rows, num_cols, SCALE)
    enc_vector = encrypt_vector_fd(
        context, encoder, sk, vector, num_rows, SCALE, chain_index)

    print(f"[{label}] r={num_rows} d={num_cols} num_chunks={enc_matrix.num_chunks} "
          f"cols_per_chunk={enc_matrix.cols_per_chunk}")

    out_ct = multiply_matrix_vector_fd(context, galois_key, enc_matrix, enc_vector)
    out_pt = sk.decrypt(context, out_ct)
    decoded = encoder.decode_double_vector(context, out_pt)

    M = np.array(matrix, dtype=np.float64).reshape(num_rows, num_cols)
    v = np.array(vector, dtype=np.float64)
    expected = M @ v

    errors = np.abs(np.array(decoded[:num_rows]) - expected)
    max_err = float(errors.max())
    avg_err = float(errors.mean())

    print(f"[{label}] max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}  tol = {tol:.0e}")
    if max_err > tol:
        raise SystemExit(f"FAIL [{label}]: max abs err {max_err:.3e} > {tol:.0e}")
    print(f"[{label}] PASS")


# Smoke test first so failures surface fast.
run_case(num_rows=32, num_cols=128, tol=1e-5, label="smoke", seed=0xCAFE)

# LLaMA-scale.
run_case(num_rows=4096, num_cols=4096, tol=1e-3, label="llama-4096", seed=0xBEEF)


def report_host_memory_savings():
    LOG_N = 16
    N = 1 << LOG_N
    NUM_SLOTS = N // 2
    SCALE = 2.0 ** 40
    bits = [60, 40, 40, 40, 40, 60]
    chain_index = 1
    num_rows = 4096
    num_cols = 4096

    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, bits))
    cols_per_chunk = NUM_SLOTS // num_rows
    steps = matvec_fd_required_steps(num_rows, cols_per_chunk)
    params.set_galois_elts(phantom.get_elts_from_steps(steps, N))
    context = phantom.context(params)
    encoder = phantom.ckks_encoder(context)

    rng = random.Random(0xFEED)
    matrix = [rng.uniform(-0.5, 0.5) for _ in range(num_rows * num_cols)]
    enc_matrix = encode_matrix_fd(
        context, encoder, matrix, num_rows, num_cols, SCALE)

    num_chunks = enc_matrix.num_chunks
    # SingleChainPlaintext: N * 8 B per chunk.
    sc_bytes = num_chunks * (N * 8)
    # Equivalent full-RNS storage at chain_index: num_active_towers * N * 8 B.
    # num_active_towers = total_parm_size - chain_index (special modulus excluded
    # at the data level, matching what multiply_plain_ntt sees here).
    num_active_towers = context.total_parm_size() - chain_index
    full_bytes = num_chunks * num_active_towers * N * 8
    sc_mib = sc_bytes / (1024 * 1024)
    full_mib = full_bytes / (1024 * 1024)
    ratio = full_bytes / sc_bytes if sc_bytes else 0.0
    print(f"[host-mem] num_chunks={num_chunks} num_active_towers={num_active_towers}")
    print(f"[host-mem] SingleChain={sc_mib:.2f} MiB, "
          f"full-RNS-equivalent={full_mib:.2f} MiB, ratio={ratio:.2f}x")


report_host_memory_savings()

print("ALL PASS")
