"""Round-trip test for multi_ct_softmax_finalize (Stage 3b-c).

Bypasses the ps_exp polynomial path (which has its own approximation error)
and feeds true exp values directly. This isolates the cross-block aggregation
+ within-block reduce + Goldschmidt 1/x logic.
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from blocks.attention import multi_ct_softmax_finalize


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
ITERS = 6


def _build_engine_with_softmax_rots():
    """Engine with rotation steps multi_ct_softmax_finalize needs:
    {1, 2, 4} for within-block sum_reduce, {-1, -2, -4} for the broadcast
    doubling after the per-head mask."""
    user_steps = [1, 2, 4, -1, -2, -4]
    fresh_ci = 16
    target_chains = [fresh_ci] * len(user_steps)
    cfg = phantom.ckks_engine_config()
    cfg.log_n = LOG_N
    cfg.user_scale = SCALE
    cfg.num_scale_levels = NUM_SCALE_LEVELS
    cfg.sparse_hw = SPARSE_HW
    cfg.num_special_primes = NUM_SPECIAL_PRIMES
    cfg.include_user_rotations = False
    cfg.user_rotation_steps = user_steps
    cfg.user_rotation_target_chain_indices = target_chains
    print(f"  Building engine (rotation steps: {user_steps})...")
    t0 = time.perf_counter()
    eng = phantom.ckks_engine(cfg)
    print(f"  engine built in {time.perf_counter()-t0:.1f}s")
    return eng


def _pack_e_blocks(e_full, num_tokens):
    """Pack e_full[T, N_HEADS] into n_blocks slot vectors using the layout
    slot[h*D_HEAD*T_MODEL + tok_local] = e[k*T_MODEL + tok_local, h] for valid
    (k, tok_local) pairs; zeros elsewhere within each (h, *) t-stride block."""
    n_blocks = math.ceil(num_tokens / T_MODEL)
    e_blocks_slots = []
    for blk in range(n_blocks):
        block_start = blk * T_MODEL
        block_size = min(T_MODEL, num_tokens - block_start)
        slots = np.zeros(NUM_SLOTS, dtype=np.float64)
        for h in range(N_HEADS):
            base = h * D_HEAD * T_MODEL
            for tok_local in range(block_size):
                abs_tok = block_start + tok_local
                slots[base + tok_local] = e_full[abs_tok, h]
        e_blocks_slots.append(slots)
    return e_blocks_slots


def test_multi_ct_softmax():
    eng = _build_engine_with_softmax_rots()
    ctx = eng.context()
    encoder = eng.encoder()
    sk = eng.secret_key()
    relin_key = eng.relin_key()
    galois_key = eng.galois_key()
    fresh_ci = eng.user_level_chain_index(0)
    print(f"  fresh_ci={fresh_ci}")

    rng = np.random.default_rng(0xBEEF)
    fail = 0
    for T in [4, 8, 16, 32, 64, 128]:
        # Generate scores in [-2, -0.5] (matches post-C-sub range; positive max=-0.5)
        scores = -rng.uniform(0.5, 2.0, (T, N_HEADS))
        # True softmax weights
        e_true = np.exp(scores - scores.max(0, keepdims=True))
        weights_np = e_true / e_true.sum(0, keepdims=True)  # [T, N_HEADS]
        # Use exp directly (skip the ps_exp polynomial). For Goldschmidt
        # convergence, per-head sum must fall in (0, 2). The real damped-
        # squarings cascade includes a 1/T cancellation factor (where T is
        # total num_tokens) — match that here so the test isolates the
        # cross-block aggregation logic, not the polynomial damping.
        e_pre = e_true / T  # per-head sum is ≈ 1 by construction

        e_slots = _pack_e_blocks(e_pre, T)
        e_cts = [sk.encrypt_symmetric(ctx,
            encoder.encode_double_vector(ctx, eb.tolist(), SCALE, fresh_ci))
            for eb in e_slots]
        # Build the per-head first-slot mask plaintext at the chain `a` will
        # be at after the within-block sum_reduce (= fresh_ci, since reduce
        # uses only rotates+adds which don't change chain).
        mask_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
        for h in range(N_HEADS):
            mask_slots[h * D_HEAD * T_MODEL] = 1.0
        mask_pt = encoder.encode_double_vector(ctx, mask_slots.tolist(), SCALE, fresh_ci)

        weights_blocks = multi_ct_softmax_finalize(
            ctx, encoder, relin_key, galois_key, e_cts,
            mask_pt, N_HEADS, D_HEAD, T_MODEL, ITERS, SCALE)

        # Decrypt and reconstruct weights_fhe[T, N_HEADS]
        weights_fhe = np.zeros((T, N_HEADS), dtype=np.float64)
        n_blocks = len(e_cts)
        for blk, w_ct in enumerate(weights_blocks):
            block_start = blk * T_MODEL
            block_size = min(T_MODEL, T - block_start)
            raw = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, w_ct)),
                           dtype=np.float64)
            for h in range(N_HEADS):
                base = h * D_HEAD * T_MODEL
                for tok_local in range(block_size):
                    abs_tok = block_start + tok_local
                    weights_fhe[abs_tok, h] = raw[base + tok_local]

        diff_max = float(np.abs(weights_fhe - weights_np).max())
        diff_rms = float(np.linalg.norm(weights_fhe - weights_np)
                          / math.sqrt(weights_np.size))
        ok = diff_max < 1e-2
        print(f"  T={T:>3d}  n_blocks={n_blocks:>2d}  "
              f"max|err|={diff_max:.3e}  rms|err|={diff_rms:.3e}  "
              f"weights_sum_check={weights_fhe.sum(0).mean():.4f} (target 1.0)  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fail += 1
    return fail


if __name__ == "__main__":
    print("=== multi_ct_softmax_finalize round-trip ===")
    n = test_multi_ct_softmax()
    sys.exit(1 if n else 0)
