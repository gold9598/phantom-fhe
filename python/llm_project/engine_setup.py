"""Engine / user-step setup + per-layer numpy calibration split out of
llama3_mrpc.py.

design: doc/design/engine_setup.md#module-contents
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from blocks.attention import (
    qkt_required_steps, score_v_required_steps, broadcast_required_steps,
)
from blocks.rmsnorm import rmsnorm_required_steps_stride_t
from blocks.softmax import (
    softmax_damping_schedule, softmax_required_steps,
)
from llama3 import (
    LOG_N, N, NUM_SLOTS, SCALE,
    D_MODEL, D_HEAD, N_HEADS, N_KV_HEADS, N_KV_GROUPS, D_TOTAL,
    T_MODEL,
    EPSILON, NUM_SQUARINGS, EXTRA_SCALE, TARGET_MAG, RMS_POLY_DEG,
    NUM_SCALE_LEVELS, NUM_SPECIAL_PRIMES, SPARSE_HW,
    USER_LEVEL_IRP_ATTN,
    BOOT_CALIB_MARGIN,
    rmsnorm_np, apply_rope_np, silu_np,
)


def _make_rms_params_local(zmin, zmax):
    """Local rmsnorm_params builder (mirrors llama3.main()'s nested fn)."""
    p = phantom.rmsnorm_params()
    p.d_model = D_MODEL
    p.epsilon = EPSILON
    p.z_min = zmin
    p.z_max = zmax
    p.poly_degree = RMS_POLY_DEG
    return p


# design: doc/design/engine_setup.md#fixed-nt-causal-invariant
def _real_nt(num_tokens, query_position):
    """Real token count (incl. query). Pad slots [real_nt, num_tokens) are
    excluded from all reductions. No-op (== num_tokens) for variable-nt."""
    if query_position is None:
        return num_tokens
    return min(num_tokens, query_position + 1)


# design: doc/design/engine_setup.md#exp-cheb-coeffs-provenance
_EXP_CHEB_DEG4_R2 = np.array([
    1.0000000000000002,
    0.9999999011179665,
    0.49999999014536933,
    0.16666798420023443,
    0.04166679798739991,
    0.008328598903862764,
    0.001388416857145537,
    0.00020469833492755798,
    2.542872206845459e-05,
])


def _sim_pre_finsmx_mean(scores_post_C_ht, num_tokens, real_nt=None):
    """Empirical mean of Stage-B softmax output over ALL NUM_SLOTS slots.

    design: doc/design/engine_setup.md#sim-pre-finsmx-mean-realnt
    """
    if real_nt is None:
        real_nt = num_tokens
    # design: doc/design/engine_setup.md#ps-exp-init-token-count
    # --- ps_exp_init on populated slots ---
    t_factor = float(real_nt) ** (-1.0 / float(2 ** NUM_SQUARINGS))
    lead = EXTRA_SCALE * t_factor
    inv_se = 1.0 / float(2 ** NUM_SQUARINGS)
    inv_pow = 1.0
    coeffs = np.empty(len(_EXP_CHEB_DEG4_R2))
    for i in range(len(_EXP_CHEB_DEG4_R2)):
        coeffs[i] = lead * _EXP_CHEB_DEG4_R2[i] * inv_pow
        inv_pow *= inv_se

    # design: doc/design/engine_setup.md#pad-columns-masked
    scores_flat = scores_post_C_ht[:, :real_nt].ravel().astype(np.float64)
    # Horner evaluation
    y_pop = np.zeros(len(scores_flat), dtype=np.float64)
    for i in range(len(coeffs) - 1, -1, -1):
        y_pop = y_pop * scores_flat + coeffs[i]

    # Damped squarings on populated slots
    damps = softmax_damping_schedule(NUM_SQUARINGS, real_nt, EXTRA_SCALE, TARGET_MAG)
    for d in damps:
        y_pop = y_pop * y_pop * d

    # --- ps_exp_init on score=0 (unpopulated slots) ---
    y0 = np.zeros(1, dtype=np.float64)
    for i in range(len(coeffs) - 1, -1, -1):
        y0 = y0 * np.zeros(1) + coeffs[i]
    for d in damps:
        y0 = y0 * y0 * d
    v0 = float(y0[0])

    n_populated = N_HEADS * real_nt
    n_unpopulated = NUM_SLOTS - n_populated
    global_mean = (y_pop.sum() + n_unpopulated * v0) / NUM_SLOTS
    return float(global_mean)


def compute_layer_calib_n(x_btd, w, cos_all, sin_all, num_tokens, query_position,
                            margin=BOOT_CALIB_MARGIN):
    """num_tokens-aware version of compute_layer_z + compute_layer_max_abs.

    Returns:
      z1, z2: rmsnorm input variance estimates for rms1 / rms2.
      max_abs: dict of bootstrap max_abs values for the in-block sites.
    """
    g1, g2 = w["g1"], w["g2"]
    Wq, Wk, Wv, Wo = w["Wq"], w["Wk"], w["Wv"], w["Wo"]
    Wgate, Wup, Wdown = w["Wgate"], w["Wup"], w["Wdown"]
    P_q = query_position
    # design: doc/design/engine_setup.md#calib-realnt-clip
    real_nt = _real_nt(num_tokens, P_q)

    # rms1 input variance
    z1 = float((x_btd[P_q] ** 2).mean() + EPSILON)

    xn = rmsnorm_np(x_btd, g1)
    Q_full = (xn @ Wq.T).reshape(num_tokens, N_HEADS, D_HEAD)
    K_full = (xn @ Wk.T).reshape(num_tokens, N_KV_HEADS, D_HEAD)
    V_full = (xn @ Wv.T).reshape(num_tokens, N_KV_HEADS, D_HEAD)
    Q_full = apply_rope_np(Q_full, cos_all, sin_all)
    K_full = apply_rope_np(K_full, cos_all, sin_all)
    K_full = np.repeat(K_full, N_KV_GROUPS, axis=1)
    V_full = np.repeat(V_full, N_KV_GROUPS, axis=1)
    # design: doc/design/engine_setup.md#exclude-eos-pad-kv
    K_full = K_full[:real_nt]
    V_full = V_full[:real_nt]
    q_max = float(np.abs(Q_full[P_q]).max())
    # scores: shape (N_HEADS, real_nt). Per-head max for c_per_head.
    scores = np.einsum('hd,thd->ht', Q_full[P_q], K_full) / math.sqrt(D_HEAD)
    c_per_head = scores.max(-1) + 0.5  # (N_HEADS,) — per-head softmax shift
    scores_post_C = scores - c_per_head[:, None]  # broadcast over T
    scores_max = float(np.abs(scores_post_C).max())
    weights = np.exp(scores_post_C - scores_post_C.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)
    # design: doc/design/engine_setup.md#softmax-safety-goldschmidt-noise
    SOFTMAX_TARGET = 1.5
    sum_t_exp = np.exp(scores_post_C).sum(axis=-1)  # (N_HEADS,) — per-head sum
    expected_max_sum = float(sum_t_exp.max() * 0.45)  # TARGET_MAG
    if expected_max_sum > SOFTMAX_TARGET:
        softmax_safety_scale = SOFTMAX_TARGET / expected_max_sum
    else:
        softmax_safety_scale = 1.0
    attn_p = np.einsum('ht,thd->hd', weights, V_full).reshape(N_HEADS * D_HEAD)
    o_p = attn_p @ Wo.T
    x_mid_full = x_btd.copy(); x_mid_full[P_q] = x_btd[P_q] + o_p
    z2 = float((x_mid_full[P_q] ** 2).mean() + EPSILON)
    x_mid_max = float(np.abs(x_mid_full[P_q]).max())
    x_mid_n = rmsnorm_np(x_mid_full, g2)
    rms2_out_max = float(np.abs(x_mid_n[P_q]).max())
    gate_pre = x_mid_n[P_q] @ Wgate.T
    gate_max = float(np.abs(gate_pre).max())
    gate_silu = silu_np(gate_pre)
    up = x_mid_n[P_q] @ Wup.T
    up_max = float(np.abs(up).max())
    h = gate_silu * up
    h_max = float(np.abs(h).max())
    mlp_out_vec = h @ Wdown.T
    mlp_out_max = float(np.abs(mlp_out_vec).max())
    mlp_out_mean = float(mlp_out_vec.mean())
    o_max = float(np.abs(o_p).max())
    o_mean_val = float(o_p.mean())

    max_abs = {
        "x_in":         float(np.abs(x_btd[P_q]).max()) * margin,
        "rms1_out":     float(np.abs(xn[P_q]).max()) * margin,
        "x_mid":        x_mid_max * margin,
        "rms2_out":     rms2_out_max * margin,
        "q":            q_max * margin,
        "scores":       scores_max * margin,
        "gate":         gate_max * margin,
        "up":           up_max * margin,
        "h":            h_max * margin,
        "o":            o_max * margin,
        "o_mean":       o_mean_val,
        "mlp_out":      mlp_out_max * margin,
        "mlp_out_mean": mlp_out_mean,
        "softmax_safety_scale": softmax_safety_scale,
        "pre_finsmx_mean": _sim_pre_finsmx_mean(
            scores_post_C, real_nt, real_nt=real_nt),
    }
    return z1, z2, max_abs


def build_user_steps_mrpc():
    """Galois rotation steps for the dense token-major MRPC pipeline.

    design: doc/design/engine_setup.md#build-user-steps-overview
    """
    # design: doc/design/engine_setup.md#p-frames-layout
    P_frames = NUM_SLOTS // D_TOTAL

    rms_steps      = list(rmsnorm_required_steps_stride_t(D_MODEL, T_MODEL))
    bsgs_steps     = [int(s) for s in phantom.bsgs_required_steps(64)]
    qkt_steps      = list(qkt_required_steps(D_HEAD))
    smx_steps      = list(softmax_required_steps(P_frames, D_TOTAL))
    score_v_steps  = list(score_v_required_steps(D_HEAD, D_TOTAL, P_frames))
    # design: doc/design/engine_setup.md#bcast-provision-unconditional
    bcast_steps    = list(broadcast_required_steps(N_HEADS))

    # design: doc/design/engine_setup.md#irp-attn-rotations
    from blocks import irp as _irp
    from blocks import attention as _attn_steps
    _BABY_STEPS_IRP = 16
    _t_k = NUM_SLOTS // D_TOTAL
    irp_steps = list(_irp.irp_required_steps(
        NUM_SLOTS, D_TOTAL, baby_steps=_BABY_STEPS_IRP))
    sdpa_steps = list(_attn_steps.sdpa_irp_required_steps(
        D_HEAD, D_TOTAL, 512, _t_k))

    # design: doc/design/engine_setup.md#irp-rect-mlp-rotations
    _BABY_STEPS_IRP_MLP_RECT = 16
    _D_PAD_OUT_MLP = 16384
    _D_OUT_FOLD_MLP = _D_PAD_OUT_MLP // 2  # 8192
    irp_rect_wide_steps = list(_irp.irp_required_steps_rect(
        NUM_SLOTS, D_MODEL, _D_PAD_OUT_MLP, baby_steps=_BABY_STEPS_IRP_MLP_RECT))
    irp_rect_tall_steps = list(_irp.irp_required_steps_rect(
        NUM_SLOTS, _D_PAD_OUT_MLP, D_MODEL, baby_steps=_BABY_STEPS_IRP_MLP_RECT))
    # design: doc/design/engine_setup.md#irp-mlp-gateup-fold
    irp_rect_fold_steps = list(_irp.irp_required_steps_rect(
        NUM_SLOTS, D_MODEL, _D_OUT_FOLD_MLP, baby_steps=_BABY_STEPS_IRP_MLP_RECT))
    _t_fold = NUM_SLOTS // _D_OUT_FOLD_MLP
    irp_fold_recombine_steps = [
        (NUM_SLOTS - _t_fold // 2) % NUM_SLOTS,  # interleave right-rotate
    ]
    # design: doc/design/engine_setup.md#irp-mlp-wdown-fold
    _D_OUT_FOLD_DOWN = D_MODEL // 2  # 2048
    irp_rect_tall_fold_steps = list(_irp.irp_required_steps_rect(
        NUM_SLOTS, _D_PAD_OUT_MLP, _D_OUT_FOLD_DOWN,
        baby_steps=_BABY_STEPS_IRP_MLP_RECT))
    # design: doc/design/engine_setup.md#irp-wq-wo-square-fold
    irp_sq_fold_steps = list(_irp.irp_required_steps_rect(
        NUM_SLOTS, D_TOTAL, D_TOTAL // 2, baby_steps=_BABY_STEPS_IRP))

    user_steps = sorted(set(
        rms_steps + bsgs_steps + qkt_steps + smx_steps
        + score_v_steps + bcast_steps + irp_steps + sdpa_steps
        + irp_rect_wide_steps + irp_rect_tall_steps
        + irp_rect_fold_steps + irp_fold_recombine_steps
        + irp_rect_tall_fold_steps + irp_sq_fold_steps
    ))
    # design: doc/design/engine_setup.md#chain-depth-buckets
    qkt_bucket = set(bsgs_steps) | set(qkt_steps)
    smx_bucket = set(smx_steps)
    score_v_neg = {s for s in score_v_steps if s < 0} | set(bcast_steps)
    step_categories = {
        "rms": set(rms_steps),
        "sdpa": set(),
        "irp_attn": set(),
        "irp_mlp_w": set(),
        "irp_mlp_t": set(),
        "softmax_within_block": smx_bucket,
        "softmax_cross_block_doubling": set(),
        "qkt_q_preprocess": qkt_bucket,
        "sdpa_score_v_broadcast": score_v_neg,
        "packed_softmax_sum_reduce": set(),
        "packed_softmax_broadcast": set(),
        "packed_unpack": set(),
    }
    return user_steps, step_categories


def setup_engine(user_steps, step_categories=None, target_chain_default=16):
    """Build engine with per-step Galois target chain assignment (Stage 3b-f-4).

    design: doc/design/engine_setup.md#setup-engine-target-chain
    """
    if step_categories is not None:
        # design: doc/design/engine_setup.md#freshest-chain-invariant
        FRESHEST_CHAIN = 16
        TARGET_RMS      = FRESHEST_CHAIN + 0   # 16
        TARGET_FINALIZE = FRESHEST_CHAIN + 1   # 17
        TARGET_SCORE_V  = FRESHEST_CHAIN + 7   # 23 (6 Goldschmidt + 1 mask)
        TARGET_IRP      = FRESHEST_CHAIN + USER_LEVEL_IRP_ATTN  # 26

        rms_set         = step_categories["rms"]
        sdpa_set        = step_categories["sdpa"]
        irp_all_set     = (step_categories["irp_attn"]
                           | step_categories["irp_mlp_w"]
                           | step_categories["irp_mlp_t"])
        irp_only_set    = irp_all_set - rms_set - sdpa_set
        qkt_q_set       = step_categories["qkt_q_preprocess"]
        finalize_set    = step_categories["softmax_within_block"]   # {1, 2, 4}
        score_v_set     = step_categories["sdpa_score_v_broadcast"]
        cross_block_set = step_categories["softmax_cross_block_doubling"]  # {-1,-2,-4}
        # design: doc/design/engine_setup.md#packed-softmax-chains
        packed_sum_reduce_set = step_categories.get("packed_softmax_sum_reduce", set())
        packed_broadcast_set  = step_categories.get("packed_softmax_broadcast", set())
        packed_unpack_set     = step_categories.get("packed_unpack", set())

        target_chain_indices = []
        for s in user_steps:
            if s in rms_set:
                target_chain_indices.append(TARGET_RMS)
            elif s in qkt_q_set:
                target_chain_indices.append(TARGET_RMS)         # post-bootstrap qkt
            elif s in cross_block_set:
                target_chain_indices.append(TARGET_FINALIZE)    # 17
            elif s in finalize_set:
                target_chain_indices.append(TARGET_FINALIZE)    # 17
            elif s in packed_sum_reduce_set:
                target_chain_indices.append(TARGET_FINALIZE)    # 17
            elif s in packed_broadcast_set:
                target_chain_indices.append(TARGET_FINALIZE)    # 17
            elif s in packed_unpack_set:
                target_chain_indices.append(TARGET_SCORE_V)     # 23
            elif s in score_v_set:
                target_chain_indices.append(TARGET_SCORE_V)     # 23
            elif s in irp_only_set:
                target_chain_indices.append(TARGET_IRP)         # 26
            else:
                target_chain_indices.append(TARGET_RMS)         # safe fallback

        # Resolve galois-element collisions with min-wins.
        def _galois_elt(step):
            m = 2 * N
            power = (step % (N // 2)) + (N // 2) if step < 0 else step % (N // 2)
            return pow(3, power, m)

        elt_min_target = {}
        for s, t in zip(user_steps, target_chain_indices):
            e = _galois_elt(s)
            if e not in elt_min_target or t < elt_min_target[e]:
                elt_min_target[e] = t
        target_chain_indices = [elt_min_target[_galois_elt(s)] for s in user_steps]

        by_target = {}
        for s, t in zip(user_steps, target_chain_indices):
            by_target.setdefault(t, []).append(s)
        print(f"  Per-step galois target chain assignment:")
        for t in sorted(by_target):
            print(f"    chain={t}: {len(by_target[t]):3d} steps")
    else:
        # Fallback: uniform target_chain (slow path retained for compatibility)
        target_chain_indices = [target_chain_default] * len(user_steps)

    cfg = phantom.ckks_engine_config()
    cfg.log_n = LOG_N
    cfg.user_scale = SCALE
    cfg.num_scale_levels = NUM_SCALE_LEVELS
    cfg.sparse_hw = SPARSE_HW
    cfg.num_special_primes = NUM_SPECIAL_PRIMES
    cfg.include_user_rotations = False
    cfg.user_rotation_steps = user_steps
    cfg.user_rotation_target_chain_indices = target_chain_indices
    # design: doc/design/engine_setup.md#bootstrap-17-levels
    cfg.use_bootstrap_to_17_levels = os.environ.get("USE_BOOTSTRAP_17") == "1"
    print(f"  Engine: logN={LOG_N} NSL={NUM_SCALE_LEVELS} #user_steps={len(user_steps)}")
    t0 = time.perf_counter()
    eng = phantom.ckks_engine(cfg)
    print(f"  engine built in {time.perf_counter()-t0:.1f}s")
    return eng
