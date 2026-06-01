"""FHE dense token-major attention block split out of llama3_mrpc.py.

Contains:
  - encrypt_layer_inputs_multi : Q/K/V build + RoPE + pack + encrypt
  - _LazyLayerWeights          : dict-like wrapper for per-layer weight subsets
  - fhe_attention_dense_full   : the 540-line end-to-end FHE attention block
                                 (QK^T -> softmax -> score·V -> IRP Wo)

Module-level state shared with llama3_mrpc.run_classifier_fhe (via re-export
in llama3_mrpc):
  - K_CACHE_SCALE              : pre-scaler on K cache magnitude
  - _DENSE_WQ_BABY_STEPS       : baby_steps for Wq BSGS (=64)
  - _LAZY_FULL_WEIGHT_CACHE    : module-level cache for lazy full-weight loads
  - _LAZY_FULL_WEIGHT_LOCK     : lock guarding the cache
"""
import math
import os
import sys
import threading

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from blocks.bootstrap import bootstrap
from blocks.kv_layout import pack_kv_blocks
from blocks import kv_layout_dense as _dense_oracle
from blocks import kv_layout_dense_fhe as _dense_fhe
from blocks import irp_cache as _irp_cache
from blocks.softmax import softmax_damping_schedule
from helpers.llama3 import (
    NUM_SLOTS, SCALE,
    D_MODEL, D_HEAD, N_HEADS, N_KV_HEADS, N_KV_GROUPS, D_TOTAL,
    T_MODEL,
    NUM_SQUARINGS, EXTRA_SCALE, ITERS, TARGET_MAG,
    rmsnorm_np, apply_rope_np,
    load_layer_weights,
)

from fhe.engine_setup import _real_nt


# K cache magnitude pre-scaler.
# design: doc/design/fhe_attention_dense.md#k-cache-scale
K_CACHE_SCALE = float(os.environ.get("K_CACHE_SCALE", "1.0"))


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

    # design: doc/design/fhe_attention_dense.md#c-per-head-real-keys
    real_nt = _real_nt(num_tokens, query_position)
    Q_np = (xn[query_position] @ Wq_baked.T).reshape(N_HEADS, D_HEAD)
    scores_np = (np.einsum('hd,thd->th', Q_np, K_full_h[:real_nt])
                 / math.sqrt(D_HEAD))
    c_per_head = scores_np.max(0) + 0.5

    # design: doc/design/fhe_attention_dense.md#pack-kv-real-tokens-only
    k_blocks_slots, v_blocks_slots = pack_kv_blocks(
        K_full_h, V_full_h, real_nt, T_MODEL, NUM_SLOTS, N_HEADS, D_HEAD,
        k_scale=K_CACHE_SCALE)
    k_cts = [sk.encrypt_symmetric(ctx,
        encoder.encode_double_vector(ctx, kb, SCALE, fresh_ci))
        for kb in k_blocks_slots]
    v_cts = [sk.encrypt_symmetric(ctx,
        encoder.encode_double_vector(ctx, vb, SCALE, fresh_ci))
        for vb in v_blocks_slots]

    x_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    x_slots[::T_MODEL][:D_MODEL] = x_btd[query_position]
    x_ct = sk.encrypt_symmetric(ctx,
        encoder.encode_double_vector(ctx, x_slots, SCALE, fresh_ci))

    return x_ct, k_cts, v_cts, c_per_head, Wq_baked


# ===========================================================================
# design: doc/design/fhe_attention_dense.md#dense-kernel-contract
# ===========================================================================

_DENSE_WQ_BABY_STEPS = 64  # bsgs_required_steps(64) ⊆ provisioned steps


def fhe_attention_dense_full(engine, ctx, encoder, sk, relin_key, galois_key,
                              xn_query, Wq_baked, K_full_h, V_full_h, Wo,
                              c_per_head, real_nt, chain_index,
                              layer_idx=None, P_local=None,
                              q_max_abs=None, o_max_abs=None, o_mean=None):
    """Stage 4 (dense-layout rewrite): close the dense attention block.

    design: doc/design/fhe_attention_dense.md#dense-full-pipeline
    """
    D = D_TOTAL
    H = D_HEAD
    nH = N_HEADS
    P = _dense_fhe.positions_per_ct(real_nt, NUM_SLOTS, D)
    n_shards = _dense_fhe.n_shards_for(real_nt, P)


    # ---- QK^T via IRP-Wq (Cachemir §4.1) + compute_qkt_irp (Cachemir §5.1).
    # design: doc/design/fhe_attention_dense.md#qkt-irp-wq
    from blocks import irp as _irp
    _BABY_STEPS_IRP_Q = 16  # M=16, G=32 for d=4096 K=512 (~sqrt(K))
    wq_irp = _irp_cache.wq_unfolded_plaintexts_cached(
        ctx, encoder,
        np.ascontiguousarray(np.asarray(Wq_baked, dtype=np.float64).T),
        N=NUM_SLOTS, d=D, scale=SCALE, baby_steps=_BABY_STEPS_IRP_Q,
        layer_idx=layer_idx, P_local=P_local)
    # Standard interleaved input: slot[i*t]=xn[i].
    x_irp_ct = _irp.encrypt_irp_input(
        ctx, encoder, sk, np.asarray(xn_query, dtype=np.float64),
        N=NUM_SLOTS, d=D, scale=SCALE, chain_index=chain_index)
    # Lazy-level: unfolded square matvec + mask + rescale = 2 levels.
    _wq_target_ci = engine.user_level_chain_index(11)
    if x_irp_ct.chain_index() < _wq_target_ci:
        x_irp_ct = phantom.mod_switch_to(ctx, x_irp_ct, _wq_target_ci)
    _mask_q = _irp.encode_irp_mask(
        ctx, encoder, NUM_SLOTS, D, SCALE, x_irp_ct.chain_index())
    q_ct = _irp.irp_matvec_host(
        ctx, encoder, galois_key, x_irp_ct, wq_irp,
        N=NUM_SLOTS, d=D, baby_steps=_BABY_STEPS_IRP_Q, mask_pt=_mask_q)
    q_ct = phantom.rescale_to_next(ctx, q_ct)
    q_ct.set_scale(SCALE)
    # design: doc/design/fhe_attention_dense.md#q-bootstrap-mean-center
    _q_oracle = (np.asarray(xn_query, np.float64)
                 @ np.asarray(Wq_baked, np.float64).T)
    _q_mean = float(_q_oracle.mean())
    _q_max_abs = q_max_abs if q_max_abs is not None else float(np.abs(_q_oracle).max()) * 1.5
    _t_q = NUM_SLOTS // D
    _q_mean_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    _q_mean_slots[::_t_q][:D] = _q_mean
    _q_mean_pt = encoder.encode_double_vector(
        ctx, _q_mean_slots, q_ct.scale(), q_ct.chain_index())
    q_ct = phantom.sub_plain(ctx, q_ct, _q_mean_pt)
    q_ct = bootstrap(engine, ctx, encoder, q_ct,
                     max_abs=_q_max_abs + abs(_q_mean),
                     slot_count=NUM_SLOTS)
    _q_mean_pt2 = encoder.encode_double_vector(
        ctx, _q_mean_slots, q_ct.scale(), q_ct.chain_index())
    q_ct = phantom.add_plain(ctx, q_ct, _q_mean_pt2)
    # design: doc/design/fhe_attention_dense.md#q-restore-user-level
    _q_target_ci = engine.user_level_chain_index(11)
    if q_ct.chain_index() < _q_target_ci:
        q_ct = phantom.mod_switch_to(ctx, q_ct, _q_target_ci)
        q_ct.set_scale(SCALE)
    # design: doc/design/fhe_attention_dense.md#qkt-irp-k-cache
    from blocks import attention as _attn
    t_k = NUM_SLOTS // D                      # 8 for LLaMA
    n_chunks_k = (real_nt + t_k - 1) // t_k   # ceil(real_nt / t_k)
    # Build §5.1 K cache cts (one ct per chunk of t_k tokens).
    k_cache_cts = []
    for c in range(n_chunks_k):
        k_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
        for h in range(nH):
            base_h = h * H * t_k
            for r in range(H):
                base_hr = base_h + r * t_k
                for p in range(t_k):
                    tok_abs = c * t_k + p
                    if tok_abs >= real_nt:
                        break
                    k_slots[base_hr + p] = K_full_h[tok_abs, h, r]
        k_pt = encoder.encode_double_vector(
            ctx, k_slots, SCALE, q_ct.chain_index())
        k_cache_cts.append(sk.encrypt_symmetric(ctx, k_pt))
    # ---- Path B: IRP-native attention chain.
    # design: doc/design/fhe_attention_dense.md#path-b-irp-native
    inv_sqrt_d = 1.0 / math.sqrt(float(H))
    nt_pad = 1
    while nt_pad < max(2, real_nt):
        nt_pad <<= 1

    # ---- Per-chunk compute_qkt_irp + per-chunk Stage A mask*scale.
    # design: doc/design/fhe_attention_dense.md#per-chunk-qkt-mask-scale
    score_cts_irp = []
    _ms_pt = None
    for c, k_ct in enumerate(k_cache_cts):
        if k_ct.chain_index() != q_ct.chain_index():
            phantom.mod_switch_to_inplace(ctx, k_ct, q_ct.chain_index())
        sc = _attn.compute_qkt_irp(
            ctx, encoder, relin_key, galois_key,
            q_ct, k_ct, d_head=H, d_total=D, t=t_k)
        if _ms_pt is None:
            _ms_pt = _attn.qkt_irp_mask_scale_plaintext(
                ctx, encoder, d_head=H, d_total=D, num_tokens=t_k, t=t_k,
                scale_value=inv_sqrt_d, chain_index=sc.chain_index(),
                encode_scale=SCALE)
        nominal = sc.scale()
        sc = phantom.multiply_plain(ctx, sc, _ms_pt)
        sc = phantom.rescale_to_next(ctx, sc)
        sc.set_scale(nominal)
        score_cts_irp.append(sc)

    # ---- Tree-aggregate the n_chunks_k per-chunk score cts into one global ct.
    # design: doc/design/fhe_attention_dense.md#tree-aggregate-scores
    n_chunks_pow2 = nt_pad // t_k
    # design: doc/design/fhe_attention_dense.md#pad-score-cts
    if len(score_cts_irp) < n_chunks_pow2:
        _zero_pt = encoder.encode_double_vector(
            ctx, np.zeros(NUM_SLOTS, dtype=np.float64),
            score_cts_irp[0].scale(), score_cts_irp[0].chain_index())
        _zero_ct = sk.encrypt_symmetric(ctx, _zero_pt)
        while len(score_cts_irp) < n_chunks_pow2:
            score_cts_irp.append(_zero_ct)
    cur = score_cts_irp
    _level = 0
    while len(cur) > 1:
        rot_step = -(t_k << _level)   # -8, -16, -32, ... (-t * 2^l)
        nxt = []
        for k in range(len(cur) // 2):
            left = cur[2 * k]
            right = phantom.rotate(ctx, cur[2 * k + 1], int(rot_step), galois_key)
            nxt.append(phantom.add(ctx, left, right))
        cur = nxt
        _level += 1
    S_global = cur[0]

    # ---- Global per-head sub(c_per_head).
    # design: doc/design/fhe_attention_dense.md#global-per-head-sub
    _sub_pt = _attn.qkt_irp_per_head_sub_plaintext(
        ctx, encoder, d_head=H, d_total=D, num_tokens=real_nt, t=t_k,
        c_per_head=c_per_head, chain_index=S_global.chain_index(),
        encode_scale=S_global.scale())
    S_global = phantom.sub_plain(ctx, S_global, _sub_pt)

    # ---- Bootstrap-1 (post-Stage-A).
    # design: doc/design/fhe_attention_dense.md#scores-calib-bound
    _SCORES_CALIB = 45.10
    S_global = bootstrap(
        engine, ctx, encoder, S_global,
        max_abs=_SCORES_CALIB, slot_count=NUM_SLOTS)

    # ---- Safety scale + global pre-bootstrap mean.
    # design: doc/design/fhe_attention_dense.md#safety-scale
    _SOFTMAX_TARGET = 1.5
    _Qd_s = (np.asarray(xn_query, np.float64)
             @ np.asarray(Wq_baked, np.float64).T).reshape(nH, H)
    _qs_s = _dense_oracle.pack_q_dense(_Qd_s, P)
    _ks_s, _ = _dense_oracle.pack_kv_dense_shards(
        np.asarray(K_full_h, np.float64),
        np.asarray(K_full_h, np.float64), real_nt, P, nH)
    _osc_s = _dense_oracle.dense_qkt(
        [_qs_s] * n_shards, _ks_s, nH, H, real_nt, P, inv_sqrt_d)
    _scc_s = _osc_s - np.asarray(c_per_head, np.float64)[None, :]
    _EC_S = [1.0000000000000002, 0.9999999011179665, 0.49999999014536933,
             0.16666798420023443, 0.04166679798739991, 0.008328598903862764,
             0.001388416857145537, 0.00020469833492755798,
             2.542872206845459e-05]
    _se_s = 2.0 ** NUM_SQUARINGS
    _lead_s = EXTRA_SCALE * (float(real_nt) ** (-1.0 / _se_s))
    _cf_s = [_lead_s * _EC_S[i] * ((1.0 / _se_s) ** i)
             for i in range(len(_EC_S))]
    _pe_s = np.zeros_like(_scc_s)
    for i, c in enumerate(_cf_s):
        _pe_s = _pe_s + c * np.power(_scc_s, i)
    _dmp_s = softmax_damping_schedule(
        NUM_SQUARINGS, real_nt, EXTRA_SCALE, TARGET_MAG)
    for d in _dmp_s:
        _pe_s = _pe_s * _pe_s
        if abs(d - 1.0) > 1e-12:
            _pe_s = _pe_s * d
    _max_head_denom = float(_pe_s.sum(axis=0).max())
    safety_scale = (_SOFTMAX_TARGET / _max_head_denom
                    if _max_head_denom > _SOFTMAX_TARGET else 1.0)

    # design: doc/design/fhe_attention_dense.md#global-pre-bootstrap-mean
    _v0 = 0.0
    for _c in reversed(_cf_s):
        _v0 = _v0 * 0.0 + _c
    for _d in _dmp_s:
        _v0 = _v0 * _v0
        if abs(_d - 1.0) > 1e-12:
            _v0 = _v0 * _d
    _v0 = float(_v0)
    _pop_sum_global = float(_pe_s[0:real_nt, :].sum())
    _n_pop_global = real_nt * nH
    _n_junk_global = NUM_SLOTS - _n_pop_global
    _global_pre_mean = (_pop_sum_global + _n_junk_global * _v0) / NUM_SLOTS

    # ---- Stage B IRP (single ct).
    # design: doc/design/fhe_attention_dense.md#stage-b-irp
    damps = softmax_damping_schedule(
        NUM_SQUARINGS, real_nt, EXTRA_SCALE, TARGET_MAG)
    e_ct = phantom.ps_exp_init(
        ctx, encoder, relin_key, S_global,
        real_nt, NUM_SQUARINGS, EXTRA_SCALE)
    phantom.square_iterations_damped_inplace(
        ctx, encoder, relin_key, e_ct, damps)
    _mean_pt = encoder.encode_double_vector(
        ctx, np.full(NUM_SLOTS, _global_pre_mean, dtype=np.float64),
        e_ct.scale(), e_ct.chain_index())
    e_ct = phantom.sub_plain(ctx, e_ct, _mean_pt)
    # Bootstrap-2 (mean-centered).
    e_ct = bootstrap(
        engine, ctx, encoder, e_ct,
        max_abs=TARGET_MAG, slot_count=NUM_SLOTS)
    _mean_pt2 = encoder.encode_double_vector(
        ctx, np.full(NUM_SLOTS, _global_pre_mean, dtype=np.float64),
        e_ct.scale(), e_ct.chain_index())
    e_ct = phantom.add_plain(ctx, e_ct, _mean_pt2)

    # ---- Stage B mask.
    # design: doc/design/fhe_attention_dense.md#stage-b-mask
    mask_slots = _attn._qkt_irp_head_mask_slots(
        NUM_SLOTS, H, D, t_k, real_nt, value=safety_scale)
    e_nominal = e_ct.scale()
    mask_pt = encoder.encode_double_vector(
        ctx, mask_slots, SCALE, e_ct.chain_index())
    e_ct = phantom.multiply_plain(ctx, e_ct, mask_pt)
    e_ct = phantom.rescale_to_next(ctx, e_ct)
    e_ct.set_scale(e_nominal)

    # ---- Stage C IRP.
    # design: doc/design/fhe_attention_dense.md#stage-c-irp
    weights_ct = _attn.finalize_softmax_irp_t(
        ctx, encoder, relin_key, galois_key,
        e_ct, num_tokens=nt_pad, iters=ITERS)

    # design: doc/design/fhe_attention_dense.md#former-bootstrap-3-removed

    # ---- Tree-distribute global weights → n_chunks_pow2 per-chunk cts.
    # design: doc/design/fhe_attention_dense.md#tree-distribute-weights
    weights_blocks = [weights_ct]
    _W = nt_pad
    while len(weights_blocks) < n_chunks_pow2:
        _half = _W // 2
        _lo_mask_slots = _attn._qkt_irp_head_mask_slots(
            NUM_SLOTS, H, D, t_k, _half, value=1.0)
        _lo_mask_pt = encoder.encode_double_vector(
            ctx, _lo_mask_slots, SCALE, weights_blocks[0].chain_index())
        _new_blocks = []
        for _wb in weights_blocks:
            _nom = _wb.scale()
            # Lower half:
            _lo = phantom.multiply_plain(ctx, _wb, _lo_mask_pt)
            _lo = phantom.rescale_to_next(ctx, _lo)
            _lo.set_scale(_nom)
            # Upper half.
            _up_rot = phantom.rotate(ctx, _wb, int(_half), galois_key)
            _up = phantom.multiply_plain(ctx, _up_rot, _lo_mask_pt)
            _up = phantom.rescale_to_next(ctx, _up)
            _up.set_scale(_nom)
            _new_blocks.append(_lo)
            _new_blocks.append(_up)
        weights_blocks = _new_blocks
        _W = _half

    # design: doc/design/fhe_attention_dense.md#truncate-weights-blocks
    weights_blocks = weights_blocks[:n_chunks_k]

    # ---- V cache: build n_chunks_k IRP-layout V cts (same layout as K cache).
    v_cache_cts = []
    for c in range(n_chunks_k):
        v_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
        for h in range(nH):
            base_h = h * H * t_k
            for r in range(H):
                base_hr = base_h + r * t_k
                for p in range(t_k):
                    tok_abs = c * t_k + p
                    if tok_abs >= real_nt:
                        break
                    v_slots[base_hr + p] = V_full_h[tok_abs, h, r]
        v_pt = encoder.encode_double_vector(
            ctx, v_slots, SCALE, weights_blocks[0].chain_index())
        v_cache_cts.append(sk.encrypt_symmetric(ctx, v_pt))

    # ---- IRP-native score_times_v_irp_multi (per-chunk; sums across chunks).
    # design: doc/design/fhe_attention_dense.md#score-times-v-irp
    _output_mask_pt = _attn.score_v_irp_output_mask_plaintext(
        ctx, encoder, d_head=H, d_total=D, t=t_k,
        chain_index=weights_blocks[0].chain_index() + 1,
        encode_scale=SCALE)
    # Align v_cache_cts chain to weights chain.
    _w_chain = weights_blocks[0].chain_index()
    for _i, _v in enumerate(v_cache_cts):
        if _v.chain_index() != _w_chain:
            phantom.mod_switch_to_inplace(ctx, _v, _w_chain)
    attn_h = _attn.score_times_v_irp_multi(
        ctx, encoder, relin_key, galois_key,
        weights_blocks, v_cache_cts,
        d_head=H, d_total=D, t=t_k,
        output_mask_pt=_output_mask_pt,
        num_tokens_per_block=t_k)

    # design: doc/design/fhe_attention_dense.md#decrypt-pre-wo-diagnostic
    _av_irp = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, attn_h)),
                       dtype=np.float64)
    fhe_attn_o = np.zeros((nH, H), dtype=np.float64)
    for h in range(nH):
        for j in range(H):
            fhe_attn_o[h, j] = _av_irp[(h * H + j) * t_k]

    # design: doc/design/fhe_attention_dense.md#bridge-2-wo
    from blocks import irp as _irp
    _BABY_STEPS_IRP_WO = 16
    t_wo = NUM_SLOTS // D   # 8 = T_MODEL
    _wo_irp = _irp_cache.wo_unfolded_plaintexts_cached(
        ctx, encoder,
        np.ascontiguousarray(np.asarray(Wo, dtype=np.float64).T),
        N=NUM_SLOTS, d=D, scale=SCALE, baby_steps=_BABY_STEPS_IRP_WO,
        layer_idx=layer_idx)
    # design: doc/design/fhe_attention_dense.md#lazy-level-wo
    _wo_target_ci = engine.user_level_chain_index(12)
    _attn_h_in = attn_h
    if _attn_h_in.chain_index() > _wo_target_ci:
        _ah_mean = float(_av_irp.mean())
        _ah_max_abs = float(np.abs(_av_irp - _ah_mean).max()) * 1.2
        _ah_mean_pt = encoder.encode_double_vector(
            ctx, [_ah_mean] * NUM_SLOTS, _attn_h_in.scale(), _attn_h_in.chain_index())
        _attn_h_in = phantom.sub_plain(ctx, _attn_h_in, _ah_mean_pt)
        _attn_h_in = bootstrap(engine, ctx, encoder, _attn_h_in,
                               max_abs=_ah_max_abs, slot_count=NUM_SLOTS)
        _ah_mean_pt2 = encoder.encode_double_vector(
            ctx, [_ah_mean] * NUM_SLOTS, _attn_h_in.scale(), _attn_h_in.chain_index())
        _attn_h_in = phantom.add_plain(ctx, _attn_h_in, _ah_mean_pt2)
    _attn_h_deep = _attn_h_in
    if _attn_h_deep.chain_index() < _wo_target_ci:
        _attn_h_deep = phantom.mod_switch_to(ctx, _attn_h_deep, _wo_target_ci)
    _mask_wo = _irp.encode_irp_mask(
        ctx, encoder, NUM_SLOTS, D, SCALE, _attn_h_deep.chain_index())
    o_ct = _irp.irp_matvec_host(
        ctx, encoder, galois_key, _attn_h_deep, _wo_irp,
        N=NUM_SLOTS, d=D, baby_steps=_BABY_STEPS_IRP_WO, mask_pt=_mask_wo)
    o_ct = phantom.rescale_to_next(ctx, o_ct)
    o_ct.set_scale(SCALE)
    # design: doc/design/fhe_attention_dense.md#o-bootstrap-bridge
    _o_max_abs = o_max_abs if o_max_abs is not None else 1.0
    _o_mean_v = o_mean if o_mean is not None else 0.0
    _o_mean_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    _o_mean_slots[::t_wo][:D] = _o_mean_v
    _o_mean_pt = encoder.encode_double_vector(
        ctx, _o_mean_slots, o_ct.scale(), o_ct.chain_index())
    o_ct = phantom.sub_plain(ctx, o_ct, _o_mean_pt)
    o_ct = bootstrap(engine, ctx, encoder, o_ct,
                     max_abs=_o_max_abs + abs(_o_mean_v),
                     slot_count=NUM_SLOTS)
    _o_mean_pt2 = encoder.encode_double_vector(
        ctx, _o_mean_slots, o_ct.scale(), o_ct.chain_index())
    o_ct = phantom.add_plain(ctx, o_ct, _o_mean_pt2)
    # Diagnostic decode (stride-t_wo natural order).
    _ov = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, o_ct)),
                   dtype=np.float64)
    fhe_out = _ov[::t_wo][:D_MODEL].copy()

    # ---- Oracle.
    # design: doc/design/fhe_attention_dense.md#oracle-spec
    Q_hd = (np.asarray(xn_query, dtype=np.float64)
            @ np.asarray(Wq_baked, dtype=np.float64).T).reshape(nH, H)
    q_slots = _dense_oracle.pack_q_dense(Q_hd, P)
    q_per_shard = [q_slots for _ in range(n_shards)]
    k_shards_oracle, v_shards_oracle = _dense_oracle.pack_kv_dense_shards(
        np.asarray(K_full_h, dtype=np.float64),
        np.asarray(V_full_h, dtype=np.float64),
        real_nt, P, nH)
    oracle_scores = _dense_oracle.dense_qkt(
        q_per_shard, k_shards_oracle, nH, H, real_nt, P, inv_sqrt_d)
    _os = oracle_scores - oracle_scores.max(axis=0, keepdims=True)
    _oe = np.exp(_os)
    oracle_weights = _oe / _oe.sum(axis=0, keepdims=True)  # (real_nt, nH)
    score_shards_oracle = [
        _dense_oracle.pack_scores_shard(
            oracle_weights, b * P, P, nH, H)
        for b in range(n_shards)
    ]
    oracle_attn_o = _dense_oracle.dense_score_v(
        score_shards_oracle, v_shards_oracle, nH, H, P)  # (nH, H)
    oracle_out = (np.asarray(Wo, dtype=np.float64)
                  @ oracle_attn_o.reshape(-1))[:D_MODEL]


    return {
        "o_ct": o_ct,
        "fhe_attn_o": fhe_attn_o,
        "oracle_attn_o": oracle_attn_o,
        "fhe_out": fhe_out,
        "oracle_out": oracle_out,
        "P": P,
        "n_shards": n_shards,
    }


# design: doc/design/fhe_attention_dense.md#lazy-full-weight-cache
_LAZY_FULL_WEIGHT_CACHE = {}
_LAZY_FULL_WEIGHT_LOCK = threading.Lock()


class _LazyLayerWeights:
    """Dict-like wrapper around a pre-loaded per-layer weight subset.

    design: doc/design/fhe_attention_dense.md#lazy-layer-weights-class
    """

    __slots__ = ("_layer_idx", "_subset", "_full_cache", "_lock")

    def __init__(self, layer_idx, subset, full_cache, lock):
        self._layer_idx = layer_idx
        self._subset = subset
        self._full_cache = full_cache
        self._lock = lock

    def _full(self):
        cached = self._full_cache.get(self._layer_idx)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._full_cache.get(self._layer_idx)
            if cached is None:
                cached = load_layer_weights(self._layer_idx)
                self._full_cache[self._layer_idx] = cached
        return cached

    def __getitem__(self, key):
        v = self._subset.get(key)
        if v is not None:
            return v
        return self._full()[key]

    def __contains__(self, key):
        if key in self._subset:
            return True
        # design: doc/design/fhe_attention_dense.md#contains-full-keyset
        return key in ("Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown", "g1", "g2")

    def __iter__(self):
        return iter(("Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown", "g1", "g2"))

    def get(self, key, default=None):
        if key in self._subset:
            return self._subset[key]
        return self._full().get(key, default)
