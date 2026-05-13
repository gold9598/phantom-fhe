"""Round-trip tests for the multi-ct K/V cache layout (Stage 3b-a).

Verifies pack_kv_blocks / unpack_kv_blocks at multiple NUM_TOKENS values, both
in pure numpy and with an actual FHE encrypt/decrypt cycle.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from blocks.kv_layout import pack_kv_blocks, unpack_kv_blocks


# Match llama3.py constants
LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2  # 32768
SCALE = 2.0 ** 40
SPARSE_HW = 128
D_HEAD = 128
N_HEADS = 32
D_MODEL = 4096
T_MODEL = NUM_SLOTS // D_MODEL  # = 8
NUM_SCALE_LEVELS = 14
NUM_SPECIAL_PRIMES = 6


def test_numpy_roundtrip():
    """K_full -> pack -> unpack should reproduce K_full exactly (no FHE)."""
    rng = np.random.default_rng(42)
    fail = 0
    for T in [1, 4, 7, 8, 9, 16, 33, 75, 128]:
        K = rng.standard_normal((T, N_HEADS, D_HEAD))
        V = rng.standard_normal((T, N_HEADS, D_HEAD))
        k_blocks, v_blocks = pack_kv_blocks(K, V, T, T_MODEL, NUM_SLOTS, N_HEADS, D_HEAD)
        K_back, V_back = unpack_kv_blocks(k_blocks, v_blocks, T, T_MODEL, N_HEADS, D_HEAD)
        ok_k = np.allclose(K, K_back)
        ok_v = np.allclose(V, V_back)
        n_blocks = len(k_blocks)
        block_size_last = T - (n_blocks - 1) * T_MODEL if n_blocks else 0
        print(f"  T={T:>3d}  n_blocks={n_blocks:>2d}  last_block_active={block_size_last}  "
              f"K_ok={ok_k}  V_ok={ok_v}")
        if not (ok_k and ok_v):
            fail += 1
            print(f"    K diff max: {np.abs(K - K_back).max():.3e}")
            print(f"    V diff max: {np.abs(V - V_back).max():.3e}")
    return fail


def _build_minimal_engine():
    """Smallest engine with matching CKKS params; no user_rotations needed for
    encrypt/decrypt round-trip."""
    cfg = phantom.ckks_engine_config()
    cfg.log_n = LOG_N
    cfg.user_scale = SCALE
    cfg.num_scale_levels = NUM_SCALE_LEVELS
    cfg.sparse_hw = SPARSE_HW
    cfg.num_special_primes = NUM_SPECIAL_PRIMES
    cfg.include_user_rotations = False
    cfg.user_rotation_steps = []
    cfg.user_rotation_target_chain_indices = []
    print("  Building engine (no user rotations)...")
    t0 = time.perf_counter()
    eng = phantom.ckks_engine(cfg)
    print(f"  engine built in {time.perf_counter()-t0:.1f}s")
    return eng


def test_fhe_roundtrip():
    """Encrypt each slot vector, decrypt, reconstruct K_full / V_full, compare."""
    eng = _build_minimal_engine()
    ctx = eng.context()
    encoder = eng.encoder()
    sk = eng.secret_key()
    fresh_ci = eng.user_level_chain_index(0)
    print(f"  fresh_ci={fresh_ci}")

    rng = np.random.default_rng(42)
    fail = 0
    for T in [4, 8, 16, 75, 128]:
        K = rng.standard_normal((T, N_HEADS, D_HEAD))
        V = rng.standard_normal((T, N_HEADS, D_HEAD))
        k_blocks, v_blocks = pack_kv_blocks(K, V, T, T_MODEL, NUM_SLOTS, N_HEADS, D_HEAD)
        # Encrypt
        k_cts = []
        v_cts = []
        for kb, vb in zip(k_blocks, v_blocks):
            kp = encoder.encode_double_vector(ctx, kb.tolist(), SCALE, fresh_ci)
            vp = encoder.encode_double_vector(ctx, vb.tolist(), SCALE, fresh_ci)
            k_cts.append(sk.encrypt_symmetric(ctx, kp))
            v_cts.append(sk.encrypt_symmetric(ctx, vp))
        # Decrypt
        k_blocks_dec = [np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)))
                        for ct in k_cts]
        v_blocks_dec = [np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)))
                        for ct in v_cts]
        # Unpack
        K_back, V_back = unpack_kv_blocks(k_blocks_dec, v_blocks_dec, T, T_MODEL, N_HEADS, D_HEAD)
        K_err = float(np.abs(K - K_back).max())
        V_err = float(np.abs(V - V_back).max())
        n_blocks = len(k_blocks)
        ok = K_err < 1e-3 and V_err < 1e-3
        print(f"  T={T:>3d}  n_blocks={n_blocks:>2d}  "
              f"K max|err|={K_err:.3e}  V max|err|={V_err:.3e}  {'OK' if ok else 'FAIL'}")
        if not ok:
            fail += 1
    return fail


if __name__ == "__main__":
    print("=== Numpy round-trip ===")
    nfail_np = test_numpy_roundtrip()
    print()
    print("=== FHE encrypt/decrypt round-trip ===")
    nfail_fhe = test_fhe_roundtrip()
    print()
    if nfail_np or nfail_fhe:
        print(f"FAIL: {nfail_np} numpy, {nfail_fhe} FHE failures")
        sys.exit(1)
    else:
        print("All round-trip tests pass.")
