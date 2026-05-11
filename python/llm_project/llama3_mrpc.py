"""End-to-end MRPC single-example FHE forward via multi-ct K/V cache.

Stage 3b-f-1: skeleton that runs at NUM_TOKENS up to T_MODEL=8 (single
block, n_blocks=1). Verifies the multi-ct attention path reduces to the
single-ct path on a known input. Same prompt as the existing llama3.py
sanity check: [BOS, "The", " quick", " brown"] (4 tokens).

3b-f-2 will scale to NUM_TOKENS=64 (n_blocks=8) on a real MRPC prompt.
"""
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from blocks.attention import (
    compute_qkt_irp_multi, multi_ct_softmax_finalize, score_times_v_irp_multi,
    qkt_irp_mask_scale_plaintext, qkt_irp_per_head_sub_plaintext,
    score_v_irp_output_mask_plaintext,
    sdpa_irp_required_steps,
)
from blocks.bootstrap import bootstrap_safe
from blocks.bootstrap_placement import (
    build_layers_from_table, find_optimal_placement, render_plan_table,
)
from blocks.irp import (
    encode_irp_diagonals_host, irp_matvec_host,
    encode_irp_mask, irp_required_steps,
    encode_irp_diagonals_rect_host, irp_matvec_rect_host,
    encode_irp_mask_rect, irp_required_steps_rect,
)
from blocks.kv_layout import pack_kv_blocks
from blocks.lm_head import yes_no_logits_np
from blocks.residual import residual
from blocks.rmsnorm import (
    rmsnorm_forward_stride_t, rmsnorm_required_steps_stride_t,
    setup_rmsnorm_weights,
)
from blocks.silu import silu, fit_silu_coeffs, fit_silu_chebyshev_basis
from blocks.softmax import softmax_damping_schedule
from llama3 import (
    LOG_N, N, NUM_SLOTS, SCALE, SPARSE_HW,
    D_MODEL, D_HEAD, N_HEADS, N_KV_HEADS, N_KV_GROUPS, D_TOTAL,
    T_MODEL, BABY_STEPS_IRP_SQUARE, D_HIDDEN, D_PAD_MLP, BABY_STEPS_IRP_MLP,
    EPSILON, P, NUM_SQUARINGS, EXTRA_SCALE, ITERS, TARGET_MAG, RMS_POLY_DEG,
    NUM_SCALE_LEVELS, NUM_SPECIAL_PRIMES,
    USER_LEVEL_IRP_ATTN, USER_LEVEL_IRP_MLP,
    PROBE, PROBE_FULL,
    BOOT_CALIB_MARGIN,
    rmsnorm_np, apply_rope_np, rope_matrix_np, silu_np,
    rms_z_window, compute_layer_z, compute_layer_max_abs,
    forward_decoder_np,
    load_layer_weights, encode_layer_irps,
    fhe_mlp_irp_bootstrap,
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


def compute_layer_z_n(x_btd, w, num_tokens, query_position):
    """num_tokens-aware version of llama3.compute_layer_z (which hardcodes
    NUM_TOKENS=4). Returns (z_rms1, z_rms2)."""
    z1 = float((x_btd[query_position] ** 2).mean() + EPSILON)
    xn = rmsnorm_np(x_btd, w["g1"])
    Q_full = (xn @ w["Wq"].T).reshape(num_tokens, N_HEADS, D_HEAD)
    K_full = (xn @ w["Wk"].T).reshape(num_tokens, N_KV_HEADS, D_HEAD)
    V_full = (xn @ w["Wv"].T).reshape(num_tokens, N_KV_HEADS, D_HEAD)
    return z1, None  # rms2 z computed below


def compute_layer_calib_n(x_btd, w, cos_all, sin_all, num_tokens, query_position,
                            margin=BOOT_CALIB_MARGIN):
    """num_tokens-aware version of compute_layer_z + compute_layer_max_abs.

    Returns:
      z1, z2: rmsnorm input variance estimates for rms1 / rms2.
      max_abs: dict of bootstrap_safe max_abs values for the in-block sites.
    """
    g1, g2 = w["g1"], w["g2"]
    Wq, Wk, Wv, Wo = w["Wq"], w["Wk"], w["Wv"], w["Wo"]
    Wgate, Wup, Wdown = w["Wgate"], w["Wup"], w["Wdown"]
    P_q = query_position

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
    q_max = float(np.abs(Q_full[P_q]).max())
    # scores: shape (N_HEADS, num_tokens). Per-head max for c_per_head.
    scores = np.einsum('hd,thd->ht', Q_full[P_q], K_full) / math.sqrt(D_HEAD)
    c_per_head = scores.max(-1) + 0.5  # (N_HEADS,) — per-head softmax shift
    scores_post_C = scores - c_per_head[:, None]  # broadcast over T
    scores_max = float(np.abs(scores_post_C).max())
    weights = np.exp(scores_post_C - scores_post_C.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)
    # Softmax safety scale: post-damped per-head sum is approximately
    # TARGET_MAG (0.45) * sum_t exp(score_post_C[h, t]). When scores are
    # tightly clustered (typical at L0 with many similar embeddings), this
    # sum can exceed Goldschmidt's convergence range (0, 2). Scale e_blocks
    # by safety_scale before softmax_correct so the per-head sum stays under
    # 1.5; weights are scale-invariant so this changes nothing in the math.
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

    max_abs = {
        "x_in":     float(np.abs(x_btd[P_q]).max()) * margin,
        "rms1_out": float(np.abs(xn[P_q]).max()) * margin,
        "x_mid":    x_mid_max * margin,
        "rms2_out": rms2_out_max * margin,
        "q":        q_max * margin,
        "scores":   scores_max * margin,
        "gate":     gate_max * margin,
        "up":       up_max * margin,
        "h":        h_max * margin,
        "softmax_safety_scale": softmax_safety_scale,
    }
    return z1, z2, max_abs


def build_user_steps_mrpc():
    """Galois rotation steps needed for the multi-ct MRPC pipeline at
    NUM_TOKENS_PER_BLOCK=T_MODEL=8.

    Returns (user_steps, step_categories) where step_categories is a dict
    of step subsets used by setup_engine for per-step target chain
    assignment.
    """
    log_t = int(round(math.log2(T_MODEL)))
    log_d_head = int(round(math.log2(D_HEAD)))

    rms_steps = rmsnorm_required_steps_stride_t(D_MODEL, T_MODEL)
    sdpa_steps = sdpa_irp_required_steps(D_HEAD, D_TOTAL, T_MODEL, T_MODEL)
    irp_attn_steps = irp_required_steps(NUM_SLOTS, D_TOTAL, baby_steps=BABY_STEPS_IRP_SQUARE)
    irp_mlp_w_steps = irp_required_steps_rect(NUM_SLOTS, D_MODEL, D_PAD_MLP,
                                                baby_steps=BABY_STEPS_IRP_MLP)
    irp_mlp_t_steps = irp_required_steps_rect(NUM_SLOTS, D_PAD_MLP, D_MODEL,
                                                baby_steps=BABY_STEPS_IRP_MLP)
    # Multi-ct softmax: within-block + broadcast doubling
    softmax_steps = []
    for s in range(log_t):
        softmax_steps.append(int(1 << s))
        softmax_steps.append(-int(1 << s))

    user_steps = sorted(set(
        list(rms_steps) + list(sdpa_steps) +
        list(irp_attn_steps) + list(irp_mlp_w_steps) + list(irp_mlp_t_steps) +
        softmax_steps
    ))
    step_categories = {
        "rms": set(rms_steps),
        "sdpa": set(sdpa_steps),
        "irp_attn": set(irp_attn_steps),
        "irp_mlp_w": set(irp_mlp_w_steps),
        "irp_mlp_t": set(irp_mlp_t_steps),
        "softmax_within_block": {int(1 << s) for s in range(log_t)},
        "softmax_cross_block_doubling": {-int(1 << s) for s in range(log_t)},
        "qkt_q_preprocess": {-int(1 << s) for s in range(log_t)},
        "sdpa_score_v_broadcast": {-int(T_MODEL * (1 << s)) for s in range(log_d_head)},
    }
    return user_steps, step_categories


def setup_engine(user_steps, step_categories=None, target_chain_default=16):
    """Build engine with per-step Galois target chain assignment (Stage 3b-f-4).

    Mirrors the optimization in llama3.py main(): each step's target_chain
    is set to the SHALLOWEST chain at which it actually fires in the pipeline.
    Smaller-target keys are larger; larger-target keys are smaller. Empirically
    on this 5090 build the savings are storage-only (per-layer compute time
    is unchanged from uniform target_chain=16) — phantom rotations cost scales
    with the ciphertext's chain, not the key's coverage size. Kept for memory
    correctness and parity with main()'s structure.

    Pipeline chain trace (between bootstraps each stage restarts at 16):

      rms steps fire at chain 16 (sum_reduce inside rmsnorm)              -> 16
      qkt_q_preprocess {-1,-2,-4} fires at chain 16 (post-Wq bootstrap)   -> 16
      finalize_softmax sum_reduce {1,2,4} fires at chain 17 (post mask)   -> 17
      cross-block doubling {-1,-2,-4} fires at chain 17 (post mask)       -> 17
        (collides with qkt_q_preprocess on same galois elt; min wins -> 16)
      score_v broadcast {-T_MODEL*2^s} fires at chain 23 (post softmax)   -> 23
      IRP-only steps fire at chain 26 (USER_LEVEL_IRP_ATTN=10)            -> 26

    Galois-element collisions are resolved with min-target-wins: the engine
    generates one key per distinct galois element, so two steps mapping to
    the same element must share the smaller (= shallowest-chain) target,
    otherwise a key sized for a deep chain would be silently used at a
    shallower chain and cause out-of-bounds reads.
    """
    if step_categories is not None:
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
    print(f"  Engine: logN={LOG_N} NSL={NUM_SCALE_LEVELS} #user_steps={len(user_steps)}")
    t0 = time.perf_counter()
    eng = phantom.ckks_engine(cfg)
    print(f"  engine built in {time.perf_counter()-t0:.1f}s")
    return eng


def encrypt_layer_inputs_multi(ctx, encoder, sk, fresh_ci, x_btd, w, R_P,
                                 num_tokens, cos_all, sin_all, query_position):
    """Compute K, V at all NUM_TOKENS positions, RoPE, pack into n_blocks
    slot vectors, encrypt. Also encrypt x at query position P.

    Returns:
      x_ct: 1 ciphertext (single-token query in stride-T_MODEL layout)
      k_cts: list of n_blocks K ciphertexts
      v_cts: list of n_blocks V ciphertexts
      c_per_head: numpy array (N_HEADS,) — per-head softmax shift constant
      Wq_baked: numpy (D_TOTAL, D_MODEL) — Wq with R_P pre-applied
    """
    g1 = w["g1"]; Wq = w["Wq"]; Wk = w["Wk"]; Wv = w["Wv"]
    Wq_baked = Wq.copy()
    for h in range(N_HEADS):
        s, e = h * D_HEAD, (h + 1) * D_HEAD
        Wq_baked[s:e, :] = R_P @ Wq[s:e, :]

    xn = rmsnorm_np(x_btd, g1)
    K = (xn @ Wk.T).reshape(num_tokens, N_KV_HEADS, D_HEAD)
    V = (xn @ Wv.T).reshape(num_tokens, N_KV_HEADS, D_HEAD)
    K = apply_rope_np(K, cos_all, sin_all)
    K_full_h = np.repeat(K, N_KV_GROUPS, axis=1)  # (num_tokens, N_HEADS, D_HEAD)
    V_full_h = np.repeat(V, N_KV_GROUPS, axis=1)

    Q_np = (xn[query_position] @ Wq_baked.T).reshape(N_HEADS, D_HEAD)
    scores_np = np.einsum('hd,thd->th', Q_np, K_full_h) / math.sqrt(D_HEAD)
    c_per_head = scores_np.max(0) + 0.5

    k_blocks_slots, v_blocks_slots = pack_kv_blocks(
        K_full_h, V_full_h, num_tokens, T_MODEL, NUM_SLOTS, N_HEADS, D_HEAD)
    k_cts = [sk.encrypt_symmetric(ctx,
        encoder.encode_double_vector(ctx, kb.tolist(), SCALE, fresh_ci))
        for kb in k_blocks_slots]
    v_cts = [sk.encrypt_symmetric(ctx,
        encoder.encode_double_vector(ctx, vb.tolist(), SCALE, fresh_ci))
        for vb in v_blocks_slots]

    x_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    x_slots[::T_MODEL][:D_MODEL] = x_btd[query_position]
    x_ct = sk.encrypt_symmetric(ctx,
        encoder.encode_double_vector(ctx, x_slots.tolist(), SCALE, fresh_ci))

    return x_ct, k_cts, v_cts, c_per_head, Wq_baked


def fhe_attention_multi_ct(engine, ctx, encoder, relin_key, galois_key, sk,
                             x_norm, diag_wq_irp, diag_wo_irp, mask_attn_pt,
                             k_cts, v_cts, c_per_head,
                             num_tokens, max_abs_calib, head_first_slot_mask_slots,
                             stage_times=None, verbose=False):
    """Multi-ct attention block. Same shape as fhe_attention_irp_bootstrap
    but using compute_qkt_irp_multi + per-block softmax pipeline +
    multi_ct_softmax_finalize + score_times_v_irp_multi.

    For a single block (n_blocks=1, NUM_TOKENS<=T_MODEL=8) this reduces to
    the single-ct flow (within FHE noise tolerance) — Stage 3b-f-1 sanity.
    """
    def _t(): return time.perf_counter()
    def _rec(name, t0):
        if stage_times is None: return
        stage_times.setdefault(name, 0.0)
        stage_times[name] += (time.perf_counter() - t0) * 1000.0

    n_blocks = len(k_cts)
    _calib = {"q": 2.5, "scores": 45.10}
    if max_abs_calib is not None:
        _calib.update({k: max_abs_calib[k] for k in ("q", "scores") if k in max_abs_calib})

    # mod_switch x_norm to USER_LEVEL_IRP_ATTN chain
    t0 = _t()
    irp_attn_ci = engine.user_level_chain_index(USER_LEVEL_IRP_ATTN)
    if x_norm.chain_index() < irp_attn_ci:
        x_irp = phantom.mod_switch_to(ctx, x_norm, irp_attn_ci)
    else:
        x_irp = x_norm
    _rec("layout_shift", t0)

    # Wq IRP -> q_ct
    t0 = _t()
    q_ct = irp_matvec_host(ctx, encoder, galois_key, x_irp, diag_wq_irp,
                            NUM_SLOTS, D_TOTAL, baby_steps=BABY_STEPS_IRP_SQUARE,
                            mask_pt=mask_attn_pt)
    q_ct = phantom.rescale_to_next(ctx, q_ct)
    q_ct.set_scale(SCALE)
    _rec("wq_irp", t0)

    if verbose: _probe("attn post-Wq", ctx, encoder, sk, q_ct)
    # Bootstrap q_ct
    t0 = _t()
    q_ct = bootstrap_safe(engine, ctx, encoder, q_ct,
                           max_abs=_calib["q"], slot_count=NUM_SLOTS)
    _rec("bootstrap", t0)
    if verbose: _probe("attn post-q-boot", ctx, encoder, sk, q_ct)

    # compute_qkt_irp_multi — produces n_blocks score blocks
    t0 = _t()
    for k_ct in k_cts:
        phantom.mod_switch_to_inplace(ctx, k_ct, q_ct.chain_index())
    score_blocks, block_sizes = compute_qkt_irp_multi(
        ctx, encoder, relin_key, galois_key,
        q_ct, k_cts, D_HEAD, D_TOTAL, T_MODEL, num_tokens=num_tokens)
    if verbose:
        for kk, sb in enumerate(score_blocks):
            _probe(f"attn post-qkt[{kk}]", ctx, encoder, sk, sb)

    # Per-block softmax pipeline, restructured to pair bootstraps across
    # consecutive blocks via merge_bootstrap (saves 6/12 bootstraps for
    # n_blocks=6: 3 stage-A pairs + 3 stage-B pairs, ~1s/layer).
    from blocks.bootstrap import merge_bootstrap
    inv_sqrt_d = 1.0 / math.sqrt(float(D_HEAD))
    safety_scale = max_abs_calib.get("softmax_safety_scale", 1.0) if max_abs_calib else 1.0
    damps = softmax_damping_schedule(NUM_SQUARINGS, num_tokens, EXTRA_SCALE, TARGET_MAG)
    _PRE_FINSMX_MEAN = 0.4487

    def _stage_a_premask(sb, blk_size):
        """mask*scale + sub_C (block-aware)."""
        nominal = sb.scale()
        ms_pt = qkt_irp_mask_scale_plaintext(
            ctx, encoder, D_HEAD, D_TOTAL, blk_size, T_MODEL,
            inv_sqrt_d, sb.chain_index(), SCALE)
        sb = phantom.multiply_plain(ctx, sb, ms_pt)
        sb = phantom.rescale_to_next(ctx, sb)
        sb.set_scale(nominal)
        sub_pt = qkt_irp_per_head_sub_plaintext(
            ctx, encoder, D_HEAD, D_TOTAL, blk_size, T_MODEL,
            c_per_head, sb.chain_index(), sb.scale())
        return phantom.sub_plain(ctx, sb, sub_pt)

    def _stage_b_ps_exp(sb):
        """ps_exp_init + damped squarings + mean-sub."""
        e_ct = phantom.ps_exp_init(
            ctx, encoder, relin_key, sb,
            num_tokens, NUM_SQUARINGS, EXTRA_SCALE)
        phantom.square_iterations_damped_inplace(
            ctx, encoder, relin_key, e_ct, damps)
        mean_pt = encoder.encode_double_vector(
            ctx, [_PRE_FINSMX_MEAN] * NUM_SLOTS, e_ct.scale(), e_ct.chain_index())
        return phantom.sub_plain(ctx, e_ct, mean_pt)

    def _stage_c_post(e_ct, blk_size):
        """mean-add + mask*safety_scale."""
        mean_pt = encoder.encode_double_vector(
            ctx, [_PRE_FINSMX_MEAN] * NUM_SLOTS, e_ct.scale(), e_ct.chain_index())
        e_ct = phantom.add_plain(ctx, e_ct, mean_pt)
        e_nominal = e_ct.scale()
        mask_pt = qkt_irp_mask_scale_plaintext(
            ctx, encoder, D_HEAD, D_TOTAL, blk_size, T_MODEL,
            safety_scale, e_ct.chain_index(), SCALE)
        e_ct = phantom.multiply_plain(ctx, e_ct, mask_pt)
        e_ct = phantom.rescale_to_next(ctx, e_ct)
        e_ct.set_scale(e_nominal)
        return e_ct

    n_blocks = len(score_blocks)
    e_blocks = [None] * n_blocks
    # Process blocks in PAIRS to share bootstrap_inplace calls.
    pair_idx = 0
    while pair_idx < n_blocks:
        i = pair_idx
        j = pair_idx + 1 if pair_idx + 1 < n_blocks else None

        # ---- Stage A: mask + sub_C for each block in pair. ----
        sb_i = _stage_a_premask(score_blocks[i], block_sizes[i])
        if j is not None:
            sb_j = _stage_a_premask(score_blocks[j], block_sizes[j])

        # ---- Bootstrap before damped squarings (merge-paired). ----
        if j is not None:
            sb_i, sb_j = merge_bootstrap(
                engine, ctx, encoder, sb_i, sb_j,
                max_abs=_calib["scores"], slot_count=NUM_SLOTS,
                galois_key=galois_key)
        else:
            sb_i = bootstrap_safe(engine, ctx, encoder, sb_i,
                                    max_abs=_calib["scores"], slot_count=NUM_SLOTS)

        # ---- Stage B: ps_exp + damped + mean-sub. ----
        e_i = _stage_b_ps_exp(sb_i)
        if j is not None:
            e_j = _stage_b_ps_exp(sb_j)

        # ---- Bootstrap after damped (NOT merge-paired here).
        # Stage B e_ct enters at user_level = max_user_level (12 levels
        # consumed by ps_exp + 4 damped squarings from a fresh stage-A
        # bootstrap), leaving no level headroom for merge_bootstrap's
        # pre-scale multiplies. Use separate bootstrap_safe — at max_abs
        # = TARGET_MAG = 0.45 (< target_mag 0.49) it skips scaling and
        # runs bootstrap_inplace directly with no level cost.
        # Opt 3 attempted: pair these bootstraps via merge_bootstrap to
        # halve the stage-B bootstrap count (10 -> 5 at nt=75). Rejected
        # because merge_bootstrap unconditionally multiplies ct2 by sd*i
        # via multiply_plain+rescale_to_next (bootstrap.py:186-192) —
        # consumes 1 level even when sd==1.0. At max_user_level inputs,
        # this raises before bootstrap_inplace. Skipping this opt keeps
        # correctness; a zero-level pack would need a phantom API for
        # i-multiplication without rescale, which doesn't exist today. ----
        e_i = bootstrap_safe(engine, ctx, encoder, e_i,
                               max_abs=TARGET_MAG, slot_count=NUM_SLOTS)
        if j is not None:
            e_j = bootstrap_safe(engine, ctx, encoder, e_j,
                                   max_abs=TARGET_MAG, slot_count=NUM_SLOTS)

        # ---- Stage C: mean-add + mask*safety_scale. ----
        e_blocks[i] = _stage_c_post(e_i, block_sizes[i])
        if j is not None:
            e_blocks[j] = _stage_c_post(e_j, block_sizes[j])

        pair_idx += 2
    _rec("attn_blocks", t0)
    if verbose:
        if safety_scale < 0.999:
            print(f"    [softmax-scale] safety_scale={safety_scale:.4f} folded into mask")
        for kk, eb in enumerate(e_blocks):
            _probe(f"attn post-e[{kk}]", ctx, encoder, sk, eb)

    # Multi-ct softmax aggregation. Encode the per-head first-slot mask
    # at e_blocks[0]'s chain so multiply_plain inside the call accepts it.
    t0 = _t()
    a_chain_guess = e_blocks[0].chain_index()
    head_first_slot_mask_pt = encoder.encode_double_vector(
        ctx, head_first_slot_mask_slots.tolist(), SCALE, a_chain_guess)
    weights_blocks = multi_ct_softmax_finalize(
        ctx, encoder, relin_key, galois_key,
        e_blocks, head_first_slot_mask_pt,
        N_HEADS, D_HEAD, T_MODEL, ITERS, SCALE,
        sk=sk if verbose else None, verbose=verbose)
    _rec("softmax_finalize", t0)
    if verbose:
        for kk, wb in enumerate(weights_blocks):
            _probe(f"attn post-weights[{kk}]", ctx, encoder, sk, wb)

    # score_times_v_irp_multi
    t0 = _t()
    weights_ci = weights_blocks[0].chain_index()
    for v_ct in v_cts:
        phantom.mod_switch_to_inplace(ctx, v_ct, weights_ci)
    sv_mask = score_v_irp_output_mask_plaintext(
        ctx, encoder, D_HEAD, D_TOTAL, T_MODEL,
        weights_ci + 1, SCALE)
    attn_irp = score_times_v_irp_multi(
        ctx, encoder, relin_key, galois_key,
        weights_blocks, v_cts,
        D_HEAD, D_TOTAL, T_MODEL, sv_mask)
    _rec("score_v", t0)
    if verbose: _probe("attn post-score_v", ctx, encoder, sk, attn_irp)

    # Wo IRP
    t0 = _t()
    irp_attn_ci = engine.user_level_chain_index(USER_LEVEL_IRP_ATTN)
    if attn_irp.chain_index() < irp_attn_ci:
        attn_irp = phantom.mod_switch_to(ctx, attn_irp, irp_attn_ci)
    o_ct = irp_matvec_host(ctx, encoder, galois_key, attn_irp, diag_wo_irp,
                            NUM_SLOTS, D_TOTAL, baby_steps=BABY_STEPS_IRP_SQUARE,
                            mask_pt=mask_attn_pt)
    o_ct = phantom.rescale_to_next(ctx, o_ct)
    o_ct.set_scale(SCALE)
    _rec("wo_irp", t0)
    return o_ct


def _probe(tag, ctx, encoder, sk, ct):
    v = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                 dtype=np.float64)
    print(f"    [probe] {tag:30s} chain={ct.chain_index():2d} "
          f"max|.|={np.abs(v).max():.4e} mean|.|={np.abs(v).mean():.4e}")


def run_classifier_fhe(num_tokens, query_position, pytorch_ref, pytorch_pre_norm,
                         cos_all_full, sin_all_full, label="prompt",
                         debug_layer=None, max_layer=None, min_layer=None,
                         rp_indep_cache=None, engine=None):
    """End-to-end FHE classifier: 32 decoder layers + LM head -> Yes/No logits.

    Args:
      num_tokens: actual number of tokens in the prompt (NUM_TOKENS).
      query_position: position to query for next-token logit (typically num_tokens-1).
      pytorch_ref: (33, num_tokens, D_MODEL) per-layer hidden states from PyTorch.
      pytorch_pre_norm: (num_tokens, D_MODEL) pre-final-norm last hidden state.
      cos_all_full / sin_all_full: RoPE tables of shape (>=num_tokens, D_HEAD).
      label: short string for printing.
    """
    print(f"=== run_classifier_fhe: {label}, NUM_TOKENS={num_tokens}, P={query_position} ===")
    P_local = query_position
    cos_all = cos_all_full[:num_tokens]
    sin_all = sin_all_full[:num_tokens]
    R_P = rope_matrix_np(cos_all[P_local], sin_all[P_local])

    final_norm_g = np.load(f"{PROBE_FULL}/final_norm_g.npy").astype(np.float64)
    lm_head_yesno = np.load(f"{PROBE_FULL}/lm_head_yesno.npy").astype(np.float64)
    meta = json.loads(open(f"{PROBE_FULL}/meta.json").read())

    # ---- Engine. If caller supplies one, reuse it (required when sharing
    # an rp_indep_cache of plaintexts across calls — plaintexts are bound to
    # the engine's (ctx, encoder) and become invalid if the engine is rebuilt).
    if engine is None:
        user_steps, step_categories = build_user_steps_mrpc()
        print(f"User steps ({len(user_steps)}): first 10 = {user_steps[:10]}")
        engine = setup_engine(user_steps, step_categories=step_categories)
    ctx = engine.context()
    encoder = engine.encoder()
    sk = engine.secret_key()
    relin_key = engine.relin_key()
    galois_key = engine.galois_key()
    fresh_ci = engine.user_level_chain_index(0)

    # ---- Layer-independent IRP masks
    irp_attn_chain = engine.user_level_chain_index(USER_LEVEL_IRP_ATTN)
    irp_mlp_chain = engine.user_level_chain_index(USER_LEVEL_IRP_MLP)
    mask_attn_pt = encode_irp_mask(ctx, encoder, NUM_SLOTS, D_TOTAL, SCALE, irp_attn_chain)
    sub_mask_mlp_wide_pt = encode_irp_mask_rect(
        ctx, encoder, NUM_SLOTS, D_MODEL, D_PAD_MLP, SCALE, irp_mlp_chain)
    sub_mask_mlp_tall_pt = encode_irp_mask_rect(
        ctx, encoder, NUM_SLOTS, D_PAD_MLP, D_MODEL, SCALE, irp_mlp_chain + 1)
    input_mask_mlp_pt = encode_irp_mask(ctx, encoder, NUM_SLOTS, D_MODEL,
                                          SCALE, irp_mlp_chain)

    # head_first_slot_mask: 1.0 at slot[h*D_HEAD*T_MODEL + 0] for h in [0, N_HEADS).
    # Encode at runtime at the correct chain (depends on intermediate level
    # usage; passed as raw slot vector to fhe_attention_multi_ct).
    hf_mask_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    for h in range(N_HEADS):
        hf_mask_slots[h * D_HEAD * T_MODEL] = 1.0

    # ---- Bootstrap placement (same as llama3.py)
    NSL_MAX = NUM_SCALE_LEVELS - 1
    T_BOOT_MS = 182.0
    OUTPUT_LEVEL_AFTER_IRP = USER_LEVEL_IRP_ATTN + 2
    placement_table = [
        ("rms1",      7,  29.4, True,  None, True),
        ("attention", 0, 521.0, True,  OUTPUT_LEVEL_AFTER_IRP, False),
        ("residual1", 0,   1.0, True,  None, False),
        ("rms2",      7,  27.4, True,  None, True),
        ("mlp",       0, 624.1, True,  OUTPUT_LEVEL_AFTER_IRP, False),
        ("residual2", 0,   1.0, True,  None, False),
    ]
    layers_for_dag = build_layers_from_table(placement_table)
    plan = find_optimal_placement(layers_for_dag, NSL_MAX, T_BOOT_MS)
    boot_before = {plan.layers[s.layer_idx].name: s.bootstrap_before for s in plan.steps}

    # ---- Per-layer FHE forward
    NUM_DECODERS = 32
    # Opt 1: pre-encode all 32 layers' R_P-dependent Wq IRPs (and prime
    # rp_indep_cache on first example) BEFORE the layer loop. Moves
    # ~1.5s/layer of encoding work out of the per-layer timer; total work
    # is unchanged but the per-layer printed time becomes a clean view of
    # FHE compute. rp_indep_cache fills lazily inside encode_layer_irps;
    # on subsequent examples the call is Wq-only (~1.5s/layer).
    t_wq_encode0 = time.perf_counter()
    wq_cache = {}  # layer_idx -> (Wq_baked, diag_wq_irp, diag_wo_irp,
                   #               diag_gate_irp, diag_up_irp, diag_down_irp)
    layer_weights = {}  # layer_idx -> w (cache weight loads too)
    for _li in range(NUM_DECODERS):
        if min_layer is not None and _li < min_layer:
            continue
        if max_layer is not None and _li > max_layer:
            break
        _w = load_layer_weights(_li)
        layer_weights[_li] = _w
        wq_cache[_li] = encode_layer_irps(ctx, encoder, _w, R_P,
                                            rp_indep_cache=rp_indep_cache,
                                            layer_idx=_li)
    t_wq_encode = time.perf_counter() - t_wq_encode0
    print(f"[wq encode: {t_wq_encode:.1f}s]")

    print(f"\nRunning {NUM_DECODERS} decoder layers...")
    layer_times = []
    y_p_fhe = None  # final hidden state at P (post-residual2 of last layer)

    for layer_idx in range(NUM_DECODERS):
        if min_layer is not None and layer_idx < min_layer:
            continue
        if max_layer is not None and layer_idx > max_layer:
            print(f"  early exit after layer {max_layer}")
            break
        t_layer_start = time.perf_counter()
        verbose = (debug_layer is not None and layer_idx == debug_layer)
        x_btd = pytorch_ref[layer_idx]  # (NUM_TOKENS, D_MODEL) — input to layer L

        # Per-layer real weights + IRP encoding (pre-encoded above)
        w = layer_weights[layer_idx]
        Wq_baked, diag_wq_irp, diag_wo_irp, diag_gate_irp, diag_up_irp, diag_down_irp = \
            wq_cache[layer_idx]

        # Per-layer rmsnorm + bootstrap_safe calibration (num_tokens-aware)
        z1_l, z2_l, max_abs_calib = compute_layer_calib_n(
            x_btd, w, cos_all, sin_all, num_tokens, P_local)
        z1_min, z1_max = rms_z_window(z1_l)
        z2_min, z2_max = rms_z_window(z2_l)
        rms1_p = _make_rms_params_local(z1_min, z1_max)
        rms2_p = _make_rms_params_local(z2_min, z2_max)
        rms1_w = setup_rmsnorm_weights(ctx, encoder, rms1_p, w["g1"].tolist(), stride=T_MODEL)
        rms2_w = setup_rmsnorm_weights(ctx, encoder, rms2_p, w["g2"].tolist(), stride=T_MODEL)

        silu_max = max_abs_calib["gate"] / BOOT_CALIB_MARGIN
        silu_domain = (-silu_max * 1.2, silu_max * 1.2)
        # Use NORMALIZED monomial fit when an adaptive degree <= 20 meets
        # the error threshold (~1e-3); falls back to the deg=32 Chebyshev
        # Clenshaw path otherwise. Clenshaw adds 2 extra bootstraps + ~30
        # ct-ct multiplies (~840ms/layer), so prefer eval_polynomial when
        # the simpler path's accuracy is comparable.
        _silu_D = silu_domain[1]
        _silu_xs = np.linspace(silu_domain[0], silu_domain[1], 1001)
        _silu_zs = _silu_xs / _silu_D
        _silu_actual = silu_np(_silu_xs)
        _SILU_ENC_SCALE = SCALE
        silu_deg = 14
        silu_coeffs = fit_silu_coeffs(silu_domain, deg=14, normalized=True)
        silu_norm_factor = 1.0 / _silu_D
        _best_err = float(np.abs(np.polyval(
            [round(c * _SILU_ENC_SCALE) / _SILU_ENC_SCALE
             for c in silu_coeffs[::-1]], _silu_zs) - _silu_actual).max())
        # Test degrees up to 20 (PS depth 5; +1 for normalization = 6 levels).
        # deg=24 with normalized coeffs has c_top ~ 8e4 (encoded ~9e16, within
        # prime 2^60 ≈ 1.15e18 but apparently triggers a slow path in Phantom's
        # eval_polynomial — observed to hang on L31 silu). deg=28+ even worse.
        # Higher degrees would also push PS depth to 6, busting chain budget.
        for _d in (10, 12, 16, 18, 20):
            _c = fit_silu_coeffs(silu_domain, deg=_d, normalized=True)
            _cq = [round(c * _SILU_ENC_SCALE) / _SILU_ENC_SCALE for c in _c]
            _err = float(np.abs(np.polyval(_cq[::-1], _silu_zs) - _silu_actual).max())
            if _err < _best_err:
                _best_err = _err
                silu_deg = _d
                silu_coeffs = _c
        # Opt 2: dispatch silu_clenshaw only when the adaptive winner is
        # still over the error budget. The deg=32 Chebyshev BASIS path
        # (Clenshaw) bounds intermediates by max|t_k| ~ silu_max — needed
        # when the normalized poly fit can't hit ~1e-3 Linf at deg <= 20.
        # Threshold = 5e-3 (matches the error budget the existing pipeline
        # tolerates at deg=32 Clenshaw on wide silu domains).
        _SILU_POLY_ERR_BUDGET = 5e-3
        if silu_deg <= 20 and _best_err <= _SILU_POLY_ERR_BUDGET:
            silu_t_coeffs = None  # gates fhe_mlp_irp_bootstrap to eval_polynomial
            silu_D = None
            _silu_path = f"poly{silu_deg}"
        else:
            silu_D = silu_domain[1]
            silu_t_coeffs = fit_silu_chebyshev_basis(silu_domain, deg=32)
            _silu_path = "clenshaw"
        if verbose or layer_idx == (min_layer if min_layer is not None else 0):
            print(f"  [silu: deg={silu_deg} path={_silu_path} Linf={_best_err:.2e}]")
        if verbose:
            margin = BOOT_CALIB_MARGIN
            ks = ("x_in", "rms1_out", "x_mid", "rms2_out",
                   "q", "scores", "gate", "up", "h")
            np_str = "  ".join(f"{k}={max_abs_calib[k]/margin:.3f}" for k in ks)
            print(f"  [calib] z1={z1_l:.3e} z2={z2_l:.3e}  np-max-abs (pre-margin):  {np_str}")
            print(f"  [calib] silu polynomial domain: [{-silu_max*1.2:.2f}, {silu_max*1.2:.2f}] "
                  f"(deg={silu_deg}, Linf-at-CKKS={_best_err:.3e})")
            print(f"  [calib] softmax_safety_scale={max_abs_calib.get('softmax_safety_scale', 1.0):.4f}")

        # Encrypt inputs (multi-ct K, V)
        x_ct, k_cts, v_cts, c_per_head, _ = encrypt_layer_inputs_multi(
            ctx, encoder, sk, fresh_ci, x_btd, w, R_P,
            num_tokens, cos_all, sin_all, P_local)

        # ---- FHE forward through one decoder layer ----
        if verbose:
            _probe("input x_ct", ctx, encoder, sk, x_ct)
            for kk, kct in enumerate(k_cts):
                _probe(f"input k_ct[{kk}]", ctx, encoder, sk, kct)
            for kk, vct in enumerate(v_cts):
                _probe(f"input v_ct[{kk}]", ctx, encoder, sk, vct)
        # rms1
        if boot_before.get("rms1", False):
            x_ct = bootstrap_safe(engine, ctx, encoder, x_ct,
                                    max_abs=max_abs_calib.get("x_in", 1.0),
                                    slot_count=NUM_SLOTS)
        x_norm = rmsnorm_forward_stride_t(ctx, encoder, relin_key, galois_key,
                                            x_ct, rms1_w, rms1_p, t=T_MODEL)
        if verbose: _probe("post-rms1", ctx, encoder, sk, x_norm)
        # attention (multi-ct)
        if boot_before.get("attention", False):
            x_norm = bootstrap_safe(engine, ctx, encoder, x_norm,
                                      max_abs=max_abs_calib.get("rms1_out", 1.0),
                                      slot_count=NUM_SLOTS)
        attn_out = fhe_attention_multi_ct(
            engine, ctx, encoder, relin_key, galois_key, sk,
            x_norm, diag_wq_irp, diag_wo_irp, mask_attn_pt,
            k_cts, v_cts, c_per_head,
            num_tokens, max_abs_calib, hf_mask_slots, verbose=verbose)
        if verbose: _probe("post-attention", ctx, encoder, sk, attn_out)
        # residual1
        x_mid_ct = residual(ctx, x_ct, attn_out)
        if verbose: _probe("post-residual1", ctx, encoder, sk, x_mid_ct)
        # rms2
        if boot_before.get("rms2", False):
            x_mid_ct = bootstrap_safe(engine, ctx, encoder, x_mid_ct,
                                        max_abs=max_abs_calib.get("x_mid", 1.0),
                                        slot_count=NUM_SLOTS)
        x_mid_norm = rmsnorm_forward_stride_t(ctx, encoder, relin_key, galois_key,
                                                x_mid_ct, rms2_w, rms2_p, t=T_MODEL)
        if verbose: _probe("post-rms2", ctx, encoder, sk, x_mid_norm)
        # mlp
        if boot_before.get("mlp", False):
            x_mid_norm = bootstrap_safe(engine, ctx, encoder, x_mid_norm,
                                           max_abs=max_abs_calib.get("rms2_out", 1.0),
                                           slot_count=NUM_SLOTS)
        mlp_out = fhe_mlp_irp_bootstrap(
            engine, ctx, encoder, relin_key, galois_key,
            x_mid_norm,
            diag_gate_irp, diag_up_irp, diag_down_irp,
            sub_mask_mlp_wide_pt, sub_mask_mlp_tall_pt, input_mask_mlp_pt,
            max_abs_calib=max_abs_calib, silu_coeffs=silu_coeffs,
            silu_norm_factor=silu_norm_factor,
            silu_t_coeffs=silu_t_coeffs, silu_D=silu_D,
            sk=sk if verbose else None, verbose_mag=verbose)
        if verbose: _probe("post-mlp", ctx, encoder, sk, mlp_out)
        # residual2
        y_ct = residual(ctx, x_mid_ct, mlp_out)
        if verbose: _probe("post-residual2 y_ct", ctx, encoder, sk, y_ct)
        layer_ms = (time.perf_counter() - t_layer_start) * 1000
        layer_times.append(layer_ms)

        # Decrypt for accuracy check (vs pre-norm reference for L=31, vs pytorch_ref[L+1] for others)
        y_full = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, y_ct)),
                           dtype=np.float64)
        y_p = y_full[::T_MODEL][:D_MODEL]
        if layer_idx < NUM_DECODERS - 1:
            ref = pytorch_ref[layer_idx + 1, P_local]
        else:
            ref = pytorch_pre_norm[P_local]  # pre-final-norm for L=31
        max_err = float(np.abs(y_p - ref).max())
        rel_rms = float(np.linalg.norm(y_p - ref) / np.linalg.norm(ref))
        print(f"  Layer {layer_idx:2d}: ‖y_fhe‖={np.linalg.norm(y_p):.4f}  "
              f"‖y_ref‖={np.linalg.norm(ref):.4f}  max|err|={max_err:.3e}  "
              f"rel-RMS={rel_rms:.3e}  t={layer_ms:.0f}ms")
        y_p_fhe = y_p

    # ---- LM head (host-side)
    yes_logit, no_logit = yes_no_logits_np(y_p_fhe, final_norm_g, lm_head_yesno,
                                              eps=meta["rms_norm_eps"])
    print(f"\n--- LM head: FHE yes_logit={yes_logit:.4f}  no_logit={no_logit:.4f} ---")
    print(f"--- Total layer time: {sum(layer_times)/1000:.1f}s "
          f"(avg {sum(layer_times)/len(layer_times):.0f}ms/layer) ---")
    return yes_logit, no_logit


def capture_pytorch_ref(token_ids):
    """Run PyTorch LLaMA-3.1-8B forward on token_ids and capture all hidden
    states + the pre-final-norm last hidden state. Returns:
      pytorch_ref:      (n_layers+1, num_tokens, D_MODEL) post-final-norm at idx -1
      pytorch_pre_norm: (num_tokens, D_MODEL) — pre-final-norm last hidden state
      yes_logit, no_logit: PyTorch reference logits at the last token position
    """
    import torch
    from transformers import AutoModelForCausalLM
    print(f"  Loading PyTorch model (fp16)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3.1-8B",
                                                  torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    input_ids = torch.tensor([token_ids], device="cuda:0")
    pre_norm_capture = {}
    h = model.model.norm.register_forward_pre_hook(
        lambda m, i: (pre_norm_capture.update(x=i[0].clone()), None)[1])
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    h.remove()
    pytorch_ref = np.stack([
        h_.squeeze(0).detach().cpu().to(torch.float32).numpy().astype(np.float64)
        for h_ in out.hidden_states
    ], axis=0)
    pytorch_pre_norm = pre_norm_capture['x'].squeeze(0).detach().cpu().to(torch.float32).numpy().astype(np.float64)
    last_logits = out.logits[0, -1].to(torch.float32).cpu().numpy()
    meta = json.loads(open(f"{PROBE_FULL}/meta.json").read())
    yes_pt = float(last_logits[meta["yes_token_id"]])
    no_pt = float(last_logits[meta["no_token_id"]])
    del model
    torch.cuda.empty_cache()
    return pytorch_ref, pytorch_pre_norm, yes_pt, no_pt


DEBUG_LAYER = None
MAX_LAYER = None
MIN_LAYER = None


def _cached_pytorch_ref(idx, truncate_to, token_ids):
    """Load cached PT reference for (idx, truncate_to) from disk if present;
    otherwise run capture_pytorch_ref and save to disk. Saves ~3 min of PT
    model load+forward when iterating on a specific layer's FHE accuracy."""
    cache_path = f"/tmp/mrpc_ptref_idx{idx}_n{len(token_ids)}.npz"
    if __import__("os").path.exists(cache_path):
        print(f"  [cache hit] loading PT ref from {cache_path}")
        z = np.load(cache_path)
        return z["ref"], z["prenorm"], float(z["yes"]), float(z["no"])
    print(f"  [cache miss] running PT and saving to {cache_path}")
    ref, prenorm, yes_pt, no_pt = capture_pytorch_ref(token_ids)
    np.savez(cache_path, ref=ref, prenorm=prenorm,
             yes=np.float64(yes_pt), no=np.float64(no_pt))
    return ref, prenorm, yes_pt, no_pt


def run_mrpc_example(idx, truncate_to=None):
    """Tokenize MRPC dev example #idx, run FHE pipeline, compare to PyTorch.
    If truncate_to is set, use only the first `truncate_to` tokens (for
    num_tokens-vs-error sweep)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    print(f"--- run_mrpc_example idx={idx} truncate_to={truncate_to} ---")
    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
    ds = load_dataset("nyu-mll/glue", "mrpc")["validation"]
    row = ds[idx]
    PROMPT_FMT = ("Are these two sentences paraphrases of each other?\n"
                  "Sentence 1: {s1}\nSentence 2: {s2}\n"
                  "Answer (Yes or No):")
    prompt = PROMPT_FMT.format(s1=row["sentence1"], s2=row["sentence2"])
    token_ids = tok(prompt).input_ids
    if truncate_to is not None and truncate_to < len(token_ids):
        token_ids = token_ids[:truncate_to]
    num_tokens = len(token_ids)
    P_local = num_tokens - 1
    print(f"  num_tokens={num_tokens}  label={row['label']}")
    print(f"  s1={row['sentence1']!r}")
    print(f"  s2={row['sentence2']!r}")

    # PyTorch reference (cached on disk)
    pytorch_ref, pytorch_pre_norm, yes_pt, no_pt = _cached_pytorch_ref(
        idx, truncate_to, token_ids)
    print(f"  PT  yes_logit={yes_pt:.4f}  no_logit={no_pt:.4f}")
    pt_pred = "Yes" if yes_pt > no_pt else "No"

    # RoPE tables for the prompt's positions
    cos_all_full = np.load(f"{PROBE_FULL}/rope_cos.npy").astype(np.float64)
    sin_all_full = np.load(f"{PROBE_FULL}/rope_sin.npy").astype(np.float64)

    yes_logit, no_logit = run_classifier_fhe(
        num_tokens, P_local, pytorch_ref, pytorch_pre_norm,
        cos_all_full, sin_all_full, label=f"mrpc_{idx}",
        debug_layer=DEBUG_LAYER, max_layer=MAX_LAYER, min_layer=MIN_LAYER)
    fhe_pred = "Yes" if yes_logit > no_logit else "No"
    print(f"\n=== Stage 3b-f-2 result ===")
    print(f"  FHE yes={yes_logit:.4f}  no={no_logit:.4f}  pred={fhe_pred}")
    print(f"  PT  yes={yes_pt:.4f}  no={no_pt:.4f}  pred={pt_pred}")
    print(f"  diff yes={abs(yes_logit-yes_pt):.3e}  no={abs(no_logit-no_pt):.3e}")
    print(f"  prediction agrees: {fhe_pred == pt_pred}")


def main_4tok():
    """Stage 3b-f-1 sanity: 4-token "[BOS] The quick brown" via the same
    pipeline. Loads the precomputed pytorch_ref / pre_norm from probe v2
    rather than re-running PyTorch."""
    print("=== main_4tok: 4-token sanity ===")
    cos_all_full = np.load(f"{PROBE}/rope_cos.npy").astype(np.float64)
    sin_all_full = np.load(f"{PROBE}/rope_sin.npy").astype(np.float64)
    pytorch_ref = np.load(f"{PROBE_FULL}/ref_acts/qbrown4_bos.npy").astype(np.float64)
    pytorch_pre_norm = np.load(f"{PROBE_FULL}/ref_acts/qbrown4_bos_prenorm.npy").astype(np.float64)
    yes_logit, no_logit = run_classifier_fhe(
        num_tokens=4, query_position=3,
        pytorch_ref=pytorch_ref, pytorch_pre_norm=pytorch_pre_norm,
        cos_all_full=cos_all_full, sin_all_full=sin_all_full,
        label="qbrown4")
    print(f"\n=== main_4tok result ===")
    print(f"  FHE yes={yes_logit:.4f}  no={no_logit:.4f}  "
          f"pred={'Yes' if yes_logit > no_logit else 'No'}")
    yes_pt_ref, no_pt_ref = 0.9551, 3.2324
    print(f"  PT  yes={yes_pt_ref:.4f}  no={no_pt_ref:.4f}  "
          f"pred={'Yes' if yes_pt_ref > no_pt_ref else 'No'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mrpc", choices=["mrpc", "qbrown4"])
    ap.add_argument("--idx", type=int, default=359,
                    help="MRPC dev example index (default: 359 — 44-token shortest)")
    ap.add_argument("--debug-layer", type=int, default=None,
                    help="Print stage probes for this layer index")
    ap.add_argument("--max-layer", type=int, default=None,
                    help="Stop after this layer index (early exit)")
    ap.add_argument("--min-layer", type=int, default=None,
                    help="Skip layers before this index (uses PT ref as input)")
    ap.add_argument("--truncate-to", type=int, default=None,
                    help="Truncate the MRPC prompt to first N tokens (for num_tokens sweep)")
    args = ap.parse_args()
    DEBUG_LAYER = args.debug_layer
    MAX_LAYER = args.max_layer
    MIN_LAYER = args.min_layer
    if args.mode == "qbrown4":
        main_4tok()
    else:
        run_mrpc_example(args.idx, truncate_to=args.truncate_to)
