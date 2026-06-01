"""Round-trip test for compute_qkt_irp_multi (Stage 3b-b).

Verifies that QK^T computed across multiple K-block ciphertexts matches the
numpy reference (Q . K^T) at multiple NUM_TOKENS values.
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from blocks.attention import compute_qkt_irp_multi
from blocks.kv_layout import pack_kv_blocks


# Match llama3.py constants
LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2  # 32768
SCALE = 2.0 ** 40
SPARSE_HW = 128
D_HEAD = 128
N_HEADS = 32
D_MODEL = 4096
D_TOTAL = N_HEADS * D_HEAD
T_MODEL = NUM_SLOTS // D_MODEL  # = 8
NUM_SCALE_LEVELS = 14
NUM_SPECIAL_PRIMES = 6


def _build_engine_with_rots():
    """Engine with the rotation steps compute_qkt_irp_multi needs for our
    config: log_t = log2(8) = 3 negative powers (-1, -2, -4) for Q preprocess,
    log_d_head = 7 positive multiples of T_MODEL (8, 16, ..., 512) for j-reduce."""
    user_steps = []
    log_t = int(round(math.log2(T_MODEL)))
    for s in range(log_t):
        user_steps.append(-(1 << s))  # -1, -2, -4
    log_d_head = int(round(math.log2(D_HEAD)))
    for s in range(log_d_head):
        user_steps.append(T_MODEL * (1 << s))  # 8, 16, 32, 64, 128, 256, 512
    fresh_ci = 16  # = 1 + (NSL-1) + 0; actually depends on engine
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


def _encode_q_layout(Q, num_slots, t_model):
    """Encode Q[N_HEADS, D_HEAD] into a slot vector in the post-Wq-IRP layout.
    slot[i*t_model] = Q.flat[i]  for i in [0, D_TOTAL).
    Other slots in each (h, j) group are zero — Q is single-token at slot 0
    of each t-stride block, matching what irp_matvec_host produces."""
    slots = np.zeros(num_slots, dtype=np.float64)
    Q_flat = Q.reshape(-1)  # [N_HEADS * D_HEAD]
    slots[::t_model][:Q_flat.size] = Q_flat
    return slots


def test_compute_qkt_multi():
    eng = _build_engine_with_rots()
    ctx = eng.context()
    encoder = eng.encoder()
    sk = eng.secret_key()
    relin_key = eng.relin_key()
    galois_key = eng.galois_key()
    fresh_ci = eng.user_level_chain_index(0)
    print(f"  fresh_ci={fresh_ci}")

    rng = np.random.default_rng(0xCAFE)
    fail = 0
    for T in [4, 8, 16, 75, 128]:
        K = rng.standard_normal((T, N_HEADS, D_HEAD)) * 0.1
        Q = rng.standard_normal((N_HEADS, D_HEAD)) * 0.1
        # Numpy reference
        scores_np = np.einsum('hd,thd->th', Q, K)  # [T, N_HEADS]

        # FHE: pack K, encode Q, encrypt, run compute_qkt_irp_multi
        k_blocks, _v_blocks = pack_kv_blocks(K, K, T, T_MODEL, NUM_SLOTS, N_HEADS, D_HEAD)
        q_slots = _encode_q_layout(Q, NUM_SLOTS, T_MODEL)
        q_ct = sk.encrypt_symmetric(ctx,
            encoder.encode_double_vector(ctx, q_slots.tolist(), SCALE, fresh_ci))
        k_cts = [sk.encrypt_symmetric(ctx,
            encoder.encode_double_vector(ctx, kb.tolist(), SCALE, fresh_ci))
            for kb in k_blocks]

        score_blocks, block_sizes = compute_qkt_irp_multi(
            ctx, encoder, relin_key, galois_key,
            q_ct, k_cts, D_HEAD, D_TOTAL, T_MODEL, num_tokens=T)

        # Decrypt blocks; reconstruct scores at slot[h*D_HEAD*T_MODEL + tok_local]
        scores_fhe = np.zeros((T, N_HEADS), dtype=np.float64)
        for blk, (s_ct, blk_size) in enumerate(zip(score_blocks, block_sizes)):
            raw = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, s_ct)),
                           dtype=np.float64)
            for h in range(N_HEADS):
                base = h * D_HEAD * T_MODEL
                for tok_local in range(blk_size):
                    abs_tok = blk * T_MODEL + tok_local
                    scores_fhe[abs_tok, h] = raw[base + tok_local]

        diff_max = float(np.abs(scores_fhe - scores_np).max())
        diff_rms = float(np.linalg.norm(scores_fhe - scores_np) / math.sqrt(scores_np.size))
        ok = diff_max < 5e-3
        n_blocks = len(k_blocks)
        print(f"  T={T:>3d}  n_blocks={n_blocks:>2d}  "
              f"max|err|={diff_max:.3e}  rms|err|={diff_rms:.3e}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fail += 1
            print(f"    FHE max: {np.abs(scores_fhe).max():.3e}")
            print(f"    NP  max: {np.abs(scores_np).max():.3e}")
    return fail


if __name__ == "__main__":
    print("=== compute_qkt_irp_multi round-trip ===")
    n = test_compute_qkt_multi()
    sys.exit(1 if n else 0)
