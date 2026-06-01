"""Single decoder-layer FHE forward + per-call shared-state dataclass.

design: doc/design/decoder_layer.md#module-overview
"""
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from blocks.bootstrap import bootstrap, merge_bootstrap
from blocks.residual import residual
from blocks.rmsnorm import rmsnorm_forward_stride_t, setup_rmsnorm_weights
from blocks.silu import silu, fit_silu_coeffs, fit_silu_chebyshev_basis
from blocks import irp_cache as _irp_cache
from blocks import calib_cache as _calib_cache
from llama3 import (
    NUM_SLOTS, SCALE,
    D_MODEL, D_HEAD, N_HEADS, N_KV_HEADS, N_KV_GROUPS, D_TOTAL,
    T_MODEL, D_HIDDEN,
    BOOT_CALIB_MARGIN,
    rmsnorm_np, apply_rope_np, silu_np, rms_z_window,
    load_layer_weights, load_layer_weights_subset,
)

from diagnostics import (
    _probe, _PROBE_DECRYPT_STAGES, _PROBE_DUMP_DIR, _PROBE_DUMP_LAYER,
)
from engine_setup import _real_nt, _make_rms_params_local, compute_layer_calib_n
from fhe_attention_dense import (
    encrypt_layer_inputs_multi, fhe_attention_dense_full,
)


@dataclass
class ClassifierCtx:
    """Per-call state shared across decoder layers + LM head.

    design: doc/design/decoder_layer.md#classifierctx-state
    """
    # --- call args ----------------------------------------------------------
    num_tokens: int
    P_local: int                  # query_position
    pytorch_ref: object
    pytorch_pre_norm: object
    cos_all: object               # cos_all_full[:num_tokens]
    sin_all: object               # sin_all_full[:num_tokens]
    R_P: object                   # rope_matrix_np at P_local
    debug_layer: object
    max_layer: object
    min_layer: object
    precomputed_calib: object
    # --- engine handles -----------------------------------------------------
    engine: object
    ctx: object
    encoder: object
    sk: object
    relin_key: object
    galois_key: object
    fresh_ci: int
    # --- plan + weights + flags --------------------------------------------
    boot_before: dict
    layer_weights: dict            # {layer_idx -> {Wq,Wk,Wv,g1,g2}}
    autonomous_fhe: bool
    # --- LM head data -------------------------------------------------------
    final_norm_g: object
    lm_head_yesno: object
    meta: dict


def run_decoder_layer(layer_idx, cctx, y_ct_carry, layer_times):
    """Run ONE decoder layer end-to-end.

    design: doc/design/decoder_layer.md#run-decoder-layer-contract
    """
    # Unpack the carry into the same locals the body used.
    num_tokens   = cctx.num_tokens
    P_local      = cctx.P_local
    pytorch_ref  = cctx.pytorch_ref
    pytorch_pre_norm = cctx.pytorch_pre_norm
    cos_all      = cctx.cos_all
    sin_all      = cctx.sin_all
    R_P          = cctx.R_P
    debug_layer  = cctx.debug_layer
    max_layer    = cctx.max_layer
    min_layer    = cctx.min_layer
    precomputed_calib = cctx.precomputed_calib
    engine       = cctx.engine
    ctx          = cctx.ctx
    encoder      = cctx.encoder
    sk           = cctx.sk
    relin_key    = cctx.relin_key
    galois_key   = cctx.galois_key
    fresh_ci     = cctx.fresh_ci
    boot_before  = cctx.boot_before
    layer_weights = cctx.layer_weights
    _autonomous_fhe = cctx.autonomous_fhe
    _y_ct_carry  = y_ct_carry
    NUM_DECODERS = 32

    t_layer_start = time.perf_counter()
    _t_wait = 0.0  # PERF_BREAKDOWN: default 0 when queue.get is not hit
    # ---- 1-layer-ahead prefetch (latency-only; never a correctness dep) --
    # design: doc/design/decoder_layer.md#prefetch-next-layer
    _next_li = layer_idx + 1
    _do_prefetch_next = (_next_li < NUM_DECODERS and
                         (max_layer is None or _next_li <= max_layer))
    _PF_BABY_ATTN = 16
    _PF_BABY_MLP = 16
    _PF_MLP_DIN = D_MODEL
    _PF_MLP_DOUT = 16384
    if _do_prefetch_next:
        _irp_cache.prefetch_layer(
            _next_li, P_local=P_local, d=D_TOTAL,
            mlp_d_in=_PF_MLP_DIN, mlp_d_out=_PF_MLP_DOUT,
            scale=SCALE, baby_steps_attn=_PF_BABY_ATTN,
            baby_steps_mlp=_PF_BABY_MLP)
        _irp_cache.prefetch_layer_weights(
            _next_li, ("Wo",), load_layer_weights_subset)
        # design: doc/design/decoder_layer.md#no-numpy-mlp-prefetch
    # Trim RAM entries for layers older than the current one.
    # design: doc/design/decoder_layer.md#evict-older-layers
    _irp_cache.evict_layers_before(
        layer_idx, P_local=P_local, d=D_TOTAL,
        mlp_d_in=_PF_MLP_DIN, mlp_d_out=_PF_MLP_DOUT,
        scale=SCALE, baby_steps_attn=_PF_BABY_ATTN,
        baby_steps_mlp=_PF_BABY_MLP)
    verbose = (debug_layer is not None and layer_idx == debug_layer)
    # design: doc/design/decoder_layer.md#probe-dump-layer-tag
    _PROBE_DUMP_LAYER[0] = layer_idx if (verbose and _PROBE_DECRYPT_STAGES) else None
    if _PROBE_DECRYPT_STAGES:
        try:
            import blocks.attention as _att_mod
            _att_mod._PROBE_DUMP_LAYER[0] = _PROBE_DUMP_LAYER[0]
        except Exception:
            pass
    x_btd = pytorch_ref[layer_idx]  # (NUM_TOKENS, D_MODEL) — input to layer L

    # Per-layer weights (the {Wq,Wk,Wv,g1,g2} subset preloaded above).
    # design: doc/design/decoder_layer.md#per-layer-weights-subset
    w = layer_weights[layer_idx]

    # Per-layer rmsnorm + bootstrap calibration (num_tokens-aware).
    # design: doc/design/decoder_layer.md#per-layer-calibration
    if precomputed_calib is not None:
        z1_l, z2_l, max_abs_calib = precomputed_calib[layer_idx]
    else:
        # design: doc/design/decoder_layer.md#calib-disk-cache
        def _compute_calib():
            _w_full = load_layer_weights(layer_idx)
            r = compute_layer_calib_n(
                x_btd, _w_full, cos_all, sin_all, num_tokens, P_local)
            del _w_full
            import gc as _gc; _gc.collect()
            return r
        z1_l, z2_l, max_abs_calib = _calib_cache.calib_cached(
            x_btd, layer_idx, num_tokens, P_local, _compute_calib)
    z1_min, z1_max = rms_z_window(z1_l)
    z2_min, z2_max = rms_z_window(z2_l)
    rms1_p = _make_rms_params_local(z1_min, z1_max)
    rms2_p = _make_rms_params_local(z2_min, z2_max)
    rms1_w = setup_rmsnorm_weights(ctx, encoder, rms1_p, w["g1"].tolist(), stride=T_MODEL)
    rms2_w = setup_rmsnorm_weights(ctx, encoder, rms2_p, w["g2"].tolist(), stride=T_MODEL)

    silu_max = max_abs_calib["gate"] / BOOT_CALIB_MARGIN
    # design: doc/design/decoder_layer.md#silu-domain-margin
    silu_domain = (-silu_max * 1.05, silu_max * 1.05)
    # design: doc/design/decoder_layer.md#silu-degree-search
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
    # design: doc/design/decoder_layer.md#silu-degree-search
    for _d in (10, 12, 16, 18, 20):
        _c = fit_silu_coeffs(silu_domain, deg=_d, normalized=True)
        _cq = [round(c * _SILU_ENC_SCALE) / _SILU_ENC_SCALE for c in _c]
        _err = float(np.abs(np.polyval(_cq[::-1], _silu_zs) - _silu_actual).max())
        if _err < _best_err:
            _best_err = _err
            silu_deg = _d
            silu_coeffs = _c
    # design: doc/design/decoder_layer.md#silu-clenshaw-dispatch
    _SILU_POLY_ERR_BUDGET = 5e-3
    # design: doc/design/decoder_layer.md#silu-clenshaw-dispatch
    if silu_deg <= 20 and _best_err <= _SILU_POLY_ERR_BUDGET and silu_max <= 6.0:
        silu_t_coeffs = None  # gates fhe_mlp_irp_bootstrap to eval_polynomial
        silu_D = None
        _silu_path = f"poly{silu_deg}"
    else:
        silu_D = silu_domain[1]
        # design: doc/design/decoder_layer.md#clenshaw-deg-32
        _clenshaw_deg = int(os.environ.get("SILU_CLENSHAW_DEG", "32"))
        silu_t_coeffs = fit_silu_chebyshev_basis(silu_domain, deg=_clenshaw_deg)
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

    # Encrypt inputs (multi-ct K, V).
    # design: doc/design/decoder_layer.md#encrypt-inputs-multi
    _t_prep_end = time.perf_counter()  # PERF_BREAKDOWN: end of host-prep phase
    t_encrypt0 = time.perf_counter()
    x_ct, k_cts, v_cts, c_per_head, _ = encrypt_layer_inputs_multi(
        ctx, encoder, sk, fresh_ci, x_btd, w, R_P,
        num_tokens, cos_all, sin_all, P_local)
    # design: doc/design/decoder_layer.md#probe-dump-calib
    if _PROBE_DECRYPT_STAGES and _PROBE_DUMP_LAYER[0] is not None:
        os.makedirs(_PROBE_DUMP_DIR, exist_ok=True)
        np.savez(f"{_PROBE_DUMP_DIR}/L{layer_idx}__calib.npz",
                 c_per_head=np.asarray(c_per_head, dtype=np.float64),
                 safety_scale=np.float64(
                     max_abs_calib.get("softmax_safety_scale", 1.0)),
                 scores_max=np.float64(max_abs_calib.get("scores", 0.0)),
                 q_max=np.float64(max_abs_calib.get("q", 0.0)),
                 num_tokens=np.int64(num_tokens),
                 query_position=np.int64(P_local))
    if _autonomous_fhe and layer_idx >= 1:
        # design: doc/design/decoder_layer.md#autonomous-fhe-carry
        assert _y_ct_carry is not None, "autonomous: missing y_ct carry"
        x_ct = bootstrap(engine, ctx, encoder, _y_ct_carry,
                               max_abs=max_abs_calib.get("x_in", 1.0),
                               slot_count=NUM_SLOTS)
    t_encrypt = time.perf_counter() - t_encrypt0

    # ---- FHE forward through one decoder layer ----
    if verbose:
        _probe("input x_ct", ctx, encoder, sk, x_ct)
        for kk, kct in enumerate(k_cts):
            _probe(f"input k_ct[{kk}]", ctx, encoder, sk, kct)
        for kk, vct in enumerate(v_cts):
            _probe(f"input v_ct[{kk}]", ctx, encoder, sk, vct)
    # rms1
    if boot_before.get("rms1", False):
        x_ct = bootstrap(engine, ctx, encoder, x_ct,
                                max_abs=max_abs_calib.get("x_in", 1.0),
                                slot_count=NUM_SLOTS)
    x_norm = rmsnorm_forward_stride_t(ctx, encoder, relin_key, galois_key,
                                        x_ct, rms1_w, rms1_p, t=T_MODEL)
    if verbose: _probe("post-rms1", ctx, encoder, sk, x_norm)
    # attention (multi-ct)
    if boot_before.get("attention", False):
        x_norm = bootstrap(engine, ctx, encoder, x_norm,
                                  max_abs=max_abs_calib.get("rms1_out", 1.0),
                                  slot_count=NUM_SLOTS)
    # design: doc/design/decoder_layer.md#dense-attention-block
    _real_nt_g = _real_nt(num_tokens, P_local)
    _g1 = w["g1"]; _Wq = w["Wq"]; _Wk = w["Wk"]; _Wv = w["Wv"]
    _Wq_baked = _Wq.copy()
    for _h in range(N_HEADS):
        _s, _e = _h * D_HEAD, (_h + 1) * D_HEAD
        _Wq_baked[_s:_e, :] = R_P @ _Wq[_s:_e, :]
    _xn = rmsnorm_np(x_btd, _g1)
    _K = (_xn @ _Wk.T).reshape(num_tokens, N_KV_HEADS, D_HEAD)
    _K = apply_rope_np(_K, cos_all, sin_all)
    _K_h = np.repeat(_K, N_KV_GROUPS, axis=1)[:_real_nt_g]
    _xn_q = _xn[P_local]
    _Wo = _irp_cache.get_layer_weights(
        layer_idx, ("Wo",), load_layer_weights_subset)["Wo"]
    _V = (_xn @ _Wv.T).reshape(num_tokens, N_KV_HEADS, D_HEAD)
    _V_h = np.repeat(_V, N_KV_GROUPS, axis=1)[:_real_nt_g]
    _fres = fhe_attention_dense_full(
        engine, ctx, encoder, sk, relin_key, galois_key,
        _xn_q, _Wq_baked, _K_h, _V_h, _Wo, c_per_head,
        _real_nt_g, fresh_ci, layer_idx=layer_idx, P_local=P_local,
        q_max_abs=max_abs_calib.get("q") if max_abs_calib else None,
        o_max_abs=max_abs_calib.get("o") if max_abs_calib else None,
        o_mean=max_abs_calib.get("o_mean") if max_abs_calib else None)
    # design: doc/design/decoder_layer.md#bridge2-bridgeless-attn
    attn_out = _fres["o_ct"]
    # residual1
    x_mid_ct = residual(ctx, x_ct, attn_out)
    if verbose: _probe("post-residual1", ctx, encoder, sk, x_mid_ct)
    # rms2
    if boot_before.get("rms2", False):
        x_mid_ct = bootstrap(engine, ctx, encoder, x_mid_ct,
                                    max_abs=max_abs_calib.get("x_mid", 1.0),
                                    slot_count=NUM_SLOTS)
    x_mid_norm = rmsnorm_forward_stride_t(ctx, encoder, relin_key, galois_key,
                                            x_mid_ct, rms2_w, rms2_p, t=T_MODEL)
    if verbose: _probe("post-rms2", ctx, encoder, sk, x_mid_norm)
    # mlp
    if boot_before.get("mlp", False):
        x_mid_norm = bootstrap(engine, ctx, encoder, x_mid_norm,
                                       max_abs=max_abs_calib.get("rms2_out", 1.0),
                                       slot_count=NUM_SLOTS)
    # ---- IRP-rect MLP (Cachemir §4.1 rect host).
    # design: doc/design/decoder_layer.md#irp-rect-mlp-overview
    from blocks import irp as _irp_mlp
    _BABY_STEPS_IRP_MLP_RECT = 16  # M=16, G=32 for K_sq=512 (~sqrt)
    _D_PAD_OUT_MLP = 16384  # D_HIDDEN=14336 padded to pow-2 multiple of D_MODEL (α=4)

    # design: doc/design/decoder_layer.md#mlp-weights-lazy
    _mlp_ws_cache = {}

    def _load_mlp_subset():
        # design: doc/design/decoder_layer.md#mlp-subset-cold-miss
        if not _mlp_ws_cache:
            _mlp_ws_cache.update(_irp_cache.get_layer_weights(
                layer_idx, ("Wgate", "Wup", "Wdown"),
                load_layer_weights_subset))
        return _mlp_ws_cache

    def _load_gate_padded():
        _Wgate = np.asarray(_load_mlp_subset()["Wgate"], dtype=np.float64)
        p = np.zeros((D_MODEL, _D_PAD_OUT_MLP), dtype=np.float64)
        p[:, :D_HIDDEN] = _Wgate.T
        return p

    def _load_up_padded():
        _Wup = np.asarray(_load_mlp_subset()["Wup"], dtype=np.float64)
        p = np.zeros((D_MODEL, _D_PAD_OUT_MLP), dtype=np.float64)
        p[:, :D_HIDDEN] = _Wup.T
        return p

    def _load_down_padded():
        _Wdown = np.asarray(_load_mlp_subset()["Wdown"], dtype=np.float64)
        p = np.zeros((_D_PAD_OUT_MLP, D_MODEL), dtype=np.float64)
        p[:D_HIDDEN, :] = _Wdown.T
        return p

    # design: doc/design/decoder_layer.md#encode-irp-rect-scps
    _D_OUT_FOLD_MLP = _D_PAD_OUT_MLP // 2  # 8192
    _gate_irp = _irp_cache.gate_plaintexts_cached(
        ctx, encoder, _load_gate_padded, N=NUM_SLOTS,
        d_in=D_MODEL, d_out=_D_PAD_OUT_MLP, scale=SCALE,
        baby_steps=_BABY_STEPS_IRP_MLP_RECT, layer_idx=layer_idx)
    _up_irp = _irp_cache.up_plaintexts_cached(
        ctx, encoder, _load_up_padded, N=NUM_SLOTS,
        d_in=D_MODEL, d_out=_D_PAD_OUT_MLP, scale=SCALE,
        baby_steps=_BABY_STEPS_IRP_MLP_RECT, layer_idx=layer_idx)
    # design: doc/design/decoder_layer.md#bridge3-bridgeless-wdown
    _down_irp = _irp_cache.down_unfolded_plaintexts_cached(
        ctx, encoder, _load_down_padded, N=NUM_SLOTS,
        d_in=_D_PAD_OUT_MLP, d_out=D_MODEL, scale=SCALE,
        baby_steps=_BABY_STEPS_IRP_MLP_RECT, layer_idx=layer_idx,
        gate_up_d_in=D_MODEL, gate_up_d_out=_D_PAD_OUT_MLP)

    # design: doc/design/decoder_layer.md#lazy-level-gate-up
    _mlp_target_ci = engine.user_level_chain_index(11)
    _x_mid_norm_deep = x_mid_norm
    if _x_mid_norm_deep.chain_index() < _mlp_target_ci:
        _x_mid_norm_deep = phantom.mod_switch_to(ctx, x_mid_norm, _mlp_target_ci)

    # design: doc/design/decoder_layer.md#rect-irp-mask-convention
    _sub_mask_gate_up = _irp_mlp.encode_irp_mask_rect(
        ctx, encoder, N=NUM_SLOTS, d_in=D_MODEL, d_out=_D_OUT_FOLD_MLP,
        scale=SCALE, chain_index=_x_mid_norm_deep.chain_index())

    def _folded_interleaved_matvec(_irp_pts):
        """Folded wide matvec → complex ct → split → interleave-recombine → real ct.

        design: doc/design/decoder_layer.md#folded-interleaved-matvec
        """
        _c = _irp_mlp.irp_matvec_rect_folded_host(
            ctx, encoder, galois_key, _x_mid_norm_deep, _irp_pts,
            N=NUM_SLOTS, d_in=D_MODEL, d_out=_D_PAD_OUT_MLP,
            baby_steps=_BABY_STEPS_IRP_MLP_RECT,
            sub_mask_pt=_sub_mask_gate_up, input_mask_pt=None)
        _c = phantom.rescale_to_next(ctx, _c)
        _c.set_scale(SCALE)
        _re, _im = _irp_mlp.extract_real_imag_pair(
            ctx, encoder, galois_key, _c, NUM_SLOTS, SCALE)
        return _irp_mlp.interleave_recombine(
            ctx, galois_key, _re, _im, NUM_SLOTS, _D_OUT_FOLD_MLP)

    # -- Wgate / Wup (folded wide rect IRP matvecs → interleaved real cts) --
    _gate_ct = _folded_interleaved_matvec(_gate_irp)
    _up_ct = _folded_interleaved_matvec(_up_irp)

    # -- silu(gate): slot-wise, layout-invariant.
    # design: doc/design/decoder_layer.md#silu-gate-layout-invariant
    # design: doc/design/decoder_layer.md#merge-bootstrap-gate-up
    _up_bound = (max_abs_calib.get("up", silu_max) / BOOT_CALIB_MARGIN
                 if max_abs_calib else silu_max)
    _merge_max_abs = max(silu_max, _up_bound)
    _gate_ct, _up_ct = merge_bootstrap(
        engine, ctx, encoder, _gate_ct, _up_ct,
        max_abs=_merge_max_abs, slot_count=NUM_SLOTS, galois_key=galois_key)
    if silu_t_coeffs is not None and silu_D is not None:
        from blocks.silu import silu_cheb_bsgs
        _silu_ct = silu_cheb_bsgs(
            engine, ctx, encoder, relin_key, _gate_ct,
            silu_D, silu_t_coeffs, NUM_SLOTS,
            galois_key=galois_key)
    else:
        _silu_ct = silu(ctx, encoder, relin_key, _gate_ct,
                        coeffs=silu_coeffs,
                        norm_factor=silu_norm_factor,
                        slot_count=NUM_SLOTS if silu_norm_factor is not None else None)

    # -- h = silu(gate) * up (chain alignment as before) --
    # design: doc/design/decoder_layer.md#h-mul-chain-align
    _s_ci = _silu_ct.chain_index()
    _u_ci = _up_ct.chain_index()
    if _u_ci < _s_ci:
        _up_ct = phantom.mod_switch_to(ctx, _up_ct, _s_ci)
    elif _u_ci > _s_ci:
        _silu_ct = phantom.mod_switch_to(ctx, _silu_ct, _u_ci)
    _silu_ct.set_scale(_up_ct.scale())
    _h_ct = phantom.multiply_and_relin(ctx, _silu_ct, _up_ct, relin_key)
    _h_ct = phantom.rescale_to_next(ctx, _h_ct)
    _h_ct.set_scale(SCALE)

    # design: doc/design/decoder_layer.md#h-boot-eliminated

    # -- Wdown (bridgeless unfolded tall rect IRP matvec, K=2048 SCPs) --
    # design: doc/design/decoder_layer.md#wdown-tall-matvec
    _wdown_target_ci = engine.user_level_chain_index(11)
    if _h_ct.chain_index() < _wdown_target_ci:
        _h_ct = phantom.mod_switch_to(ctx, _h_ct, _wdown_target_ci)
    _input_mask_down = _irp_mlp.encode_irp_mask(
        ctx, encoder, N=NUM_SLOTS, d=D_MODEL, scale=SCALE,
        chain_index=_h_ct.chain_index())
    _sub_mask_down = _irp_mlp.encode_irp_mask_rect(
        ctx, encoder, N=NUM_SLOTS, d_in=_D_PAD_OUT_MLP, d_out=D_MODEL,
        scale=SCALE, chain_index=_h_ct.chain_index() + 1)
    mlp_out = _irp_mlp.irp_matvec_rect_host(
        ctx, encoder, galois_key, _h_ct, _down_irp,
        N=NUM_SLOTS, d_in=_D_PAD_OUT_MLP, d_out=D_MODEL,
        baby_steps=_BABY_STEPS_IRP_MLP_RECT,
        sub_mask_pt=_sub_mask_down,
        input_mask_pt=_input_mask_down)
    mlp_out = phantom.rescale_to_next(ctx, mlp_out)
    mlp_out.set_scale(SCALE)
    # design: doc/design/decoder_layer.md#mlp-out-bootstrap-mean-center
    _mlp_max_abs = (max_abs_calib.get("mlp_out", 1.0)
                    if max_abs_calib else 1.0)
    _mlp_mean = (max_abs_calib.get("mlp_out_mean", 0.0)
                 if max_abs_calib else 0.0)
    _mlp_mean_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    _mlp_mean_slots[::T_MODEL][:D_MODEL] = _mlp_mean
    _mlp_mean_pt = encoder.encode_double_vector(
        ctx, _mlp_mean_slots, mlp_out.scale(), mlp_out.chain_index())
    mlp_out = phantom.sub_plain(ctx, mlp_out, _mlp_mean_pt)
    mlp_out = bootstrap(engine, ctx, encoder, mlp_out,
                        max_abs=_mlp_max_abs + abs(_mlp_mean),
                        slot_count=NUM_SLOTS)
    _mlp_mean_pt2 = encoder.encode_double_vector(
        ctx, _mlp_mean_slots, mlp_out.scale(), mlp_out.chain_index())
    mlp_out = phantom.add_plain(ctx, mlp_out, _mlp_mean_pt2)
    if verbose: _probe("post-mlp", ctx, encoder, sk, mlp_out)
    # design: doc/design/decoder_layer.md#residual2
    y_ct = residual(ctx, x_mid_ct, mlp_out)
    if verbose: _probe("post-residual2 y_ct", ctx, encoder, sk, y_ct)
    layer_ms = (time.perf_counter() - t_layer_start) * 1000
    layer_times.append(layer_ms)

    # design: doc/design/decoder_layer.md#decrypt-accuracy-check
    t_decrypt0 = time.perf_counter()
    y_full = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, y_ct)),
                       dtype=np.float64)
    t_decrypt = time.perf_counter() - t_decrypt0
    y_p = y_full[::T_MODEL][:D_MODEL]
    if layer_idx < NUM_DECODERS - 1:
        ref = pytorch_ref[layer_idx + 1, P_local]
    else:
        ref = pytorch_pre_norm[P_local]  # pre-final-norm for L=31
    max_err = float(np.abs(y_p - ref).max())
    rel_rms = float(np.linalg.norm(y_p - ref) / np.linalg.norm(ref))
    t_fhe_ms = layer_ms - (t_encrypt + t_decrypt) * 1000.0
    print(f"  Layer {layer_idx:2d}: ‖y_fhe‖={np.linalg.norm(y_p):.4f}  "
          f"‖y_ref‖={np.linalg.norm(ref):.4f}  max|err|={max_err:.3e}  "
          f"rel-RMS={rel_rms:.3e}  t={layer_ms:.0f}ms  "
          f"[encrypt={t_encrypt*1000:.0f}ms decrypt={t_decrypt*1000:.0f}ms "
          f"fhe={t_fhe_ms:.0f}ms]")
    if os.environ.get("PERF_BREAKDOWN") == "1":
        _qwait_ms    = _t_wait * 1000.0
        _prep_ms     = (_t_prep_end - t_layer_start) * 1000.0 - _qwait_ms
        _fhec_ms     = t_fhe_ms - _qwait_ms - _prep_ms
        print(f"    [pb L{layer_idx:02d}] qwait={_qwait_ms:.0f}ms "
              f"prep={_prep_ms:.0f}ms fhecompute={_fhec_ms:.0f}ms "
              f"(fhe_field={t_fhe_ms:.0f}ms total={layer_ms:.0f}ms)")
    y_p_fhe = y_p
    if _autonomous_fhe:
        # design: doc/design/decoder_layer.md#autonomous-fhe-carry-out
        _y_ct_carry = y_ct
    return y_p_fhe, _y_ct_carry
