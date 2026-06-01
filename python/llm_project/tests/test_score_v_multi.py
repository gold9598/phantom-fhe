"""Round-trip test for score_times_v_irp_multi (Stage 3b-d).

Verifies attn[h, j] = Σ_t weights[t, h] * V[t, h, j] computed across
n_blocks ciphertexts matches the numpy reference at NUM_TOKENS up to 128.
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from blocks.attention import score_times_v_irp_multi
from blocks.kv_layout import pack_kv_blocks


# Match llama3.py constants
LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2  # 32768
SCALE = 2.0 ** 40
SPARSE_HW = 128
D_HEAD = 128
N_HEADS = 32
D_TOTAL = N_HEADS * D_HEAD
D_MODEL = 4096
T_MODEL = NUM_SLOTS // D_MODEL  # = 8
NUM_SCALE_LEVELS = 14
NUM_SPECIAL_PRIMES = 6


def _build_engine_with_score_v_rots():
    """Engine with rotation steps score_times_v_irp_multi needs:
    {-T_MODEL * 2^s for s in [0, log2(D_HEAD))} for the j-axis broadcast,
    and {2^s for s in [0, log2(T_MODEL))} for the tok-axis reduce.
    """
    user_steps = []
    log_d_head = int(round(math.log2(D_HEAD)))
    for s in range(log_d_head):
        user_steps.append(-int(T_MODEL * (1 << s)))  # -8, -16, ..., -512
    log_t = int(round(math.log2(T_MODEL)))
    for s in range(log_t):
        user_steps.append(int(1 << s))  # 1, 2, 4
    user_steps = sorted(set(user_steps))
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
    print(f"  Building engine ({len(user_steps)} rotation steps)...")
    t0 = time.perf_counter()
    eng = phantom.ckks_engine(cfg)
    print(f"  engine built in {time.perf_counter()-t0:.1f}s")
    return eng


def _pack_weights_blocks(weights_full, num_tokens):
    """Pack weights_full[T, N_HEADS] into n_blocks slot vectors using the
    same layout multi_ct_softmax_finalize emits:
    slot[h*D_HEAD*T_MODEL + tok_local] = weights[k*T_MODEL + tok_local, h]
    for valid (k, tok_local); other slots are zero (representing the
    masked-out junk + non-meaningful positions)."""
    n_blocks = math.ceil(num_tokens / T_MODEL)
    blocks_slots = []
    for blk in range(n_blocks):
        block_start = blk * T_MODEL
        block_size = min(T_MODEL, num_tokens - block_start)
        slots = np.zeros(NUM_SLOTS, dtype=np.float64)
        for h in range(N_HEADS):
            base = h * D_HEAD * T_MODEL
            for tok_local in range(block_size):
                abs_tok = block_start + tok_local
                slots[base + tok_local] = weights_full[abs_tok, h]
        blocks_slots.append(slots)
    return blocks_slots


def _build_output_mask_slots():
    """stride-T_MODEL mask: 1 at slot i*T_MODEL for i in [0, D_TOTAL)."""
    slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    slots[::T_MODEL][:D_TOTAL] = 1.0
    return slots


def test_score_v_multi():
    eng = _build_engine_with_score_v_rots()
    ctx = eng.context()
    encoder = eng.encoder()
    sk = eng.secret_key()
    relin_key = eng.relin_key()
    galois_key = eng.galois_key()
    fresh_ci = eng.user_level_chain_index(0)
    print(f"  fresh_ci={fresh_ci}")

    rng = np.random.default_rng(0xF00D)
    fail = 0
    for T in [4, 8, 16, 32, 64, 128]:
        # Random softmax-like weights (sum to 1 per head) and random V values
        logits = rng.standard_normal((T, N_HEADS))
        ex = np.exp(logits - logits.max(0, keepdims=True))
        weights_np = ex / ex.sum(0, keepdims=True)  # [T, N_HEADS]
        V = rng.standard_normal((T, N_HEADS, D_HEAD)) * 0.5  # [T, N_HEADS, D_HEAD]
        # Numpy reference: attn[h, j] = Σ_t weights[t, h] * V[t, h, j]
        attn_np = np.einsum('th,thd->hd', weights_np, V)  # [N_HEADS, D_HEAD]

        # Pack into multi-ct
        w_blocks_slots = _pack_weights_blocks(weights_np, T)
        v_blocks_slots, _ = pack_kv_blocks(V, V, T, T_MODEL, NUM_SLOTS, N_HEADS, D_HEAD)
        # Encrypt all
        w_cts = [sk.encrypt_symmetric(ctx,
            encoder.encode_double_vector(ctx, wb.tolist(), SCALE, fresh_ci))
            for wb in w_blocks_slots]
        v_cts = [sk.encrypt_symmetric(ctx,
            encoder.encode_double_vector(ctx, vb.tolist(), SCALE, fresh_ci))
            for vb in v_blocks_slots]
        # Output mask. score_times_v_irp_multi calls score_times_v_irp which
        # does ct*ct (rescale to fresh_ci+1) then multiply_plain mask + rescale.
        # So the mask must be at chain fresh_ci+1.
        mask_slots = _build_output_mask_slots()
        mask_pt = encoder.encode_double_vector(ctx, mask_slots.tolist(),
                                                 SCALE, fresh_ci + 1)

        attn_irp = score_times_v_irp_multi(
            ctx, encoder, relin_key, galois_key,
            w_cts, v_cts,
            D_HEAD, D_TOTAL, T_MODEL, mask_pt)

        # Decrypt; output has attn[h, j] at slot[(h*D_HEAD+j)*T_MODEL]
        raw = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, attn_irp)),
                       dtype=np.float64)
        attn_fhe = np.zeros((N_HEADS, D_HEAD), dtype=np.float64)
        for h in range(N_HEADS):
            for j in range(D_HEAD):
                slot = (h * D_HEAD + j) * T_MODEL
                attn_fhe[h, j] = raw[slot]

        diff_max = float(np.abs(attn_fhe - attn_np).max())
        diff_rms = float(np.linalg.norm(attn_fhe - attn_np) / math.sqrt(attn_np.size))
        ok = diff_max < 5e-3
        n_blocks = len(w_cts)
        print(f"  T={T:>3d}  n_blocks={n_blocks:>2d}  "
              f"max|err|={diff_max:.3e}  rms|err|={diff_rms:.3e}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fail += 1
    return fail


if __name__ == "__main__":
    print("=== score_times_v_irp_multi round-trip ===")
    n = test_score_v_multi()
    sys.exit(1 if n else 0)
