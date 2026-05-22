"""End-to-end MRPC single-example FHE forward via multi-ct K/V cache.

Stage 3b-f-1: skeleton that runs at NUM_TOKENS up to T_MODEL=8 (single
block, n_blocks=1). Verifies the multi-ct attention path reduces to the
single-ct path on a known input. Same prompt as the existing llama3.py
sanity check: [BOS, "The", " quick", " brown"] (4 tokens).

3b-f-2 will scale to NUM_TOKENS=64 (n_blocks=8) on a real MRPC prompt.
"""
import ctypes
import json
import math
import os
import sys
import threading
import time

# libc.malloc_trim helper for the streaming-rp_indep path. Phantom's
# cudaMallocHost pages live OUTSIDE glibc, but per-layer numpy
# temporaries in _build_irp_slots and astype copies DO go through
# glibc, and on a 62 GB box they accumulate uncoalesced free chunks
# fast enough to push RSS past the ceiling between layers.
try:
    _LIBC = ctypes.CDLL("libc.so.6")
    _LIBC.malloc_trim.argtypes = [ctypes.c_size_t]
    _LIBC.malloc_trim.restype = ctypes.c_int
except Exception:
    _LIBC = None


def _malloc_trim():
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
# Dense pipeline rotation-step builders (IRP machinery deleted; the dense
# token-major path is the SOLE compute path). All four cover exactly the
# galois steps the dense kernels rotate:
#   - qkt_required_steps  -> phantom.compute_qkt inner-sum(d_head)
#   - score_v_required_steps -> phantom.score_times_v broadcast+accumulate
#   - broadcast_required_steps -> optional DENSE_SMX_BCAST per-head fold
#   - bsgs_required_steps(64) -> Wq/Wo/MLP BSGS (phantom C++ primitive)
from blocks.attention import (
    qkt_required_steps, score_v_required_steps, broadcast_required_steps,
)
from blocks.bootstrap import bootstrap_safe, merge_bootstrap
from blocks.bootstrap_placement import (
    build_layers_from_table, find_optimal_placement, render_plan_table,
)
from blocks.kv_layout import pack_kv_blocks
from blocks import kv_layout_dense as _dense_oracle
from blocks import kv_layout_dense_fhe as _dense_fhe
from blocks import dense_bsgs_cache as _dense_bsgs_cache
from blocks import irp_cache as _irp_cache
from blocks import calib_cache as _calib_cache
from blocks.lm_head import yes_no_logits_np
from blocks.residual import residual
from blocks.rmsnorm import (
    rmsnorm_forward_stride_t, rmsnorm_required_steps_stride_t,
    setup_rmsnorm_weights,
)
from blocks.silu import silu, fit_silu_coeffs, fit_silu_chebyshev_basis
from blocks.softmax import (
    softmax_damping_schedule, softmax_required_steps, sum_reduce_stride,
)
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
    load_layer_weights, load_layer_weights_subset,
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


# K cache magnitude pre-scaler. Reduces ||K_h||_2 entering the QKT ct·ct
# multiply — post-QKT err is Cauchy-Schwarz-bounded by err_Q · ||K_h||_2.
# Default 0.25 → 4× err reduction at zero level/runtime cost. Math is
# invariant: inv_sqrt_d divides by the same factor downstream.
K_CACHE_SCALE = float(os.environ.get("K_CACHE_SCALE", "1.0"))

# Fixed-nt causal-correctness invariant.
#
# FHE attention has no causal mask: correctness comes purely from the
# query only ever seeing real (non-padded) keys. In variable-nt mode the
# prompt is exactly real_nt tokens with the query at the last position
# (query_position == num_tokens - 1), so num_tokens IS the real token
# count and there is nothing to clip. In --fixed-nt N mode the prompt is
# padded with EOS keys AFTER the query (query_position = real_nt - 1,
# num_tokens = N > real_nt); those pad keys must be excluded from every
# score / softmax / calibration reduction or the query non-causally
# attends future EOS tokens and the layer output diverges (L0 ~1e45).
#
# `real_nt = query_position + 1` is the count of real tokens up to and
# including the query and is the single quantity every num_tokens-direct
# attention/softmax/calibration site must use. It is derivable at every
# call site from the already-threaded query_position, so the fix needs
# no new parameter or env flag and is the DEFAULT behavior. When the
# prompt is not padded (variable-nt), real_nt == num_tokens and every
# clip below is a no-op: the variable-nt path is byte-identical.
def _real_nt(num_tokens, query_position):
    """Real token count (incl. query). Pad slots [real_nt, num_tokens) are
    excluded from all reductions. No-op (== num_tokens) for variable-nt."""
    if query_position is None:
        return num_tokens
    return min(num_tokens, query_position + 1)


# Degree-8 polynomial coefficients for exp on [-2, 2], extracted from
# softmax.cu:21-31 (EXP_CHEB_COEFFS_DEG4_R2).  Used by
# _sim_pre_finsmx_mean below; stored at module level to avoid repeated
# array construction.
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

    Plaintext simulation (pure NumPy, no GPU).  Mirrors /tmp/sim_pre_finsmx_mean.py.

    Args:
        scores_post_C_ht: np.ndarray shape (N_HEADS, num_tokens) — real
            per-head post-sub_C scores from the current layer.
        num_tokens: int — number of populated tokens.
        real_nt: int or None — real (non-padded) token count. At fixed-nt
            the block_sizes clip masks pad-token score slots to 0 in the
            FHE pipeline, so columns [real_nt, num_tokens) behave like
            unpopulated (score=0) slots. None / == num_tokens → no-op
            (variable-nt: byte-identical to the original).
    """
    if real_nt is None:
        real_nt = num_tokens
    # ps_exp_init / damps use the SAME token count the FHE pipeline uses
    # (real_nt). The damping schedule's f_sq*d = target_mag cancellation
    # only holds when t_factor and damps share one token count, so these
    # MUST move together (see softmax_damping_schedule).
    # --- ps_exp_init on populated slots ---
    t_factor = float(real_nt) ** (-1.0 / float(2 ** NUM_SQUARINGS))
    lead = EXTRA_SCALE * t_factor
    inv_se = 1.0 / float(2 ** NUM_SQUARINGS)
    inv_pow = 1.0
    coeffs = np.empty(len(_EXP_CHEB_DEG4_R2))
    for i in range(len(_EXP_CHEB_DEG4_R2)):
        coeffs[i] = lead * _EXP_CHEB_DEG4_R2[i] * inv_pow
        inv_pow *= inv_se

    # Only the first real_nt score columns are populated; pad columns are
    # masked to 0 by the block_sizes clip and contribute v0 like every
    # other unpopulated slot.
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
      max_abs: dict of bootstrap_safe max_abs values for the in-block sites.
    """
    g1, g2 = w["g1"], w["g2"]
    Wq, Wk, Wv, Wo = w["Wq"], w["Wk"], w["Wv"], w["Wo"]
    Wgate, Wup, Wdown = w["Wgate"], w["Wup"], w["Wdown"]
    P_q = query_position
    # Real (non-padded) token count. The FHE pipeline's block_sizes clip
    # masks pad-token K/V slots [real_nt, num_tokens) to 0 at stage-A, so
    # the calibration MUST compute scores / c_per_head / softmax_safety /
    # pre_finsmx_mean over the SAME first real_nt keys — otherwise the
    # padded-key scores (huge at L0: residual magnitude largest there)
    # pollute c_per_head, ps_exp saturates, and the layer blows up to
    # ~1e45. Variable-nt: real_nt == num_tokens → identical slicing,
    # byte-identical result.
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
    # Exclude EOS-pad keys/values: the query (at P_q = real_nt - 1) only
    # ever attends real tokens [0, real_nt). No-op for variable-nt.
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
    # Softmax safety scale: post-damped per-head sum is approximately
    # TARGET_MAG (0.45) * sum_t exp(score_post_C[h, t]). Goldschmidt
    # `softmax_correct` converges for a∈(0, 2) but its CKKS noise floor
    # scales roughly as |a|^iters per iteration multiplication — at a≈1.5
    # the floor is ~570× larger than at a≈0.6. Block-0 (attention sink)
    # carries the largest weights → inherits this amplified Goldschmidt
    # residual, causing the observed 7.3× per-block noise asymmetry at L=10.
    # weights=e/a is scale-invariant, so we aggressively scale to land at
    # ~0.6 instead of ~1.5.
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
        "pre_finsmx_mean": _sim_pre_finsmx_mean(
            scores_post_C, real_nt, real_nt=real_nt),
    }
    return z1, z2, max_abs


def build_user_steps_mrpc():
    """Galois rotation steps for the dense token-major MRPC pipeline.

    The IRP machinery is gone; the dense path is the SOLE compute path.
    Every step below is a rotation a dense FHE kernel actually performs
    (verified: the dense pipeline rotates exactly this set, a subset of
    the rotations the deleted IRP step builders used to provision, so the
    galois-key set is unchanged in coverage):

      rms_steps      rmsnorm_required_steps_stride_t  -> stride-t rmsnorm
                       {8,16,...,16384}
      bsgs_steps     phantom.bsgs_required_steps(64)  -> Wq/Wo/MLP BSGS
                       (bsgs_matmul_preencoded, d_pad in {D_TOTAL,D_PAD_MLP},
                        baby_steps in {64,128}; required steps {1,2,...,64})
      qkt_steps      qkt_required_steps(D_HEAD)       -> phantom.compute_qkt
                       inner-sum over d_head {1,2,...,64}
      smx_steps      softmax_required_steps(P,D_TOTAL)-> dense softmax
                       sum_reduce_stride(stride=D_TOTAL,count=P) {4096,8192,16384}
      score_v_steps  score_v_required_steps(D_HEAD,   -> phantom.score_times_v
                       D_TOTAL,P)  {-1,..,-64, 4096,8192,16384}
      bcast_steps    broadcast_required_steps(N_HEADS)-> optional per-head
                       softmax fold (env DENSE_SMX_BCAST) {-1,..,-16}

    Returns (user_steps, step_categories); step_categories buckets steps
    by the chain depth they fire at for setup_engine's per-step galois
    target-chain assignment.
    """
    # Full positions per ciphertext frame (dense token-major layout):
    # NUM_SLOTS / D_TOTAL frames; the dense softmax sum_reduce_stride folds
    # these P frames at stride D_TOTAL.
    P_frames = NUM_SLOTS // D_TOTAL

    rms_steps      = list(rmsnorm_required_steps_stride_t(D_MODEL, T_MODEL))
    bsgs_steps     = [int(s) for s in phantom.bsgs_required_steps(64)]
    qkt_steps      = list(qkt_required_steps(D_HEAD))
    smx_steps      = list(softmax_required_steps(P_frames, D_TOTAL))
    score_v_steps  = list(score_v_required_steps(D_HEAD, D_TOTAL, P_frames))
    # The DENSE_SMX_BCAST per-head fold is OFF by default; provision its
    # keys unconditionally so toggling the env var never needs an engine
    # rebuild (5 negative single-step keys, already a subset of score_v's
    # negative broadcast set — zero extra distinct galois elements).
    bcast_steps    = list(broadcast_required_steps(N_HEADS))

    # IRP §4.1 rotations (used by Wq + Wo) + §5.1 compute_qkt_irp rotations.
    # Wq and Wo share the same IRP rotation pattern at d=D_TOTAL=4096
    # baby_steps=16 → set-union collapses them. sdpa_irp_required_steps is a
    # safe superset covering QK^T + softmax + score_v rotations for future
    # downstream IRP extensions.
    from blocks import irp as _irp
    from blocks import attention as _attn_steps
    _BABY_STEPS_IRP = 16
    _t_k = NUM_SLOTS // D_TOTAL
    irp_steps = list(_irp.irp_required_steps(
        NUM_SLOTS, D_TOTAL, baby_steps=_BABY_STEPS_IRP))
    sdpa_steps = list(_attn_steps.sdpa_irp_required_steps(
        D_HEAD, D_TOTAL, 512, _t_k))

    # IRP-rect MLP rotations (Cachemir §4.1 rect). MLP is now IRP-rect:
    # Wgate/Wup wide (d_in=D_MODEL=4096, d_out=D_PAD_OUT=16384) and Wdown
    # tall (d_in=D_PAD_OUT=16384, d_out=D_MODEL=4096). baby_steps=16 shared
    # with the attention IRP, so the square sub-IRP rotations set-union
    # collapse; only the α-stride rect rotations (q*t_prime / (N-q*t_prime))
    # are new. Provisioned unconditionally — IRP MLP is the sole compute path.
    _BABY_STEPS_IRP_MLP_RECT = 16
    _D_PAD_OUT_MLP = 16384
    _D_OUT_FOLD_MLP = _D_PAD_OUT_MLP // 2  # 8192
    irp_rect_wide_steps = list(_irp.irp_required_steps_rect(
        NUM_SLOTS, D_MODEL, _D_PAD_OUT_MLP, baby_steps=_BABY_STEPS_IRP_MLP_RECT))
    irp_rect_tall_steps = list(_irp.irp_required_steps_rect(
        NUM_SLOTS, _D_PAD_OUT_MLP, D_MODEL, baby_steps=_BABY_STEPS_IRP_MLP_RECT))
    # MLP gate/up are now complex output-FOLDED at d_out_fold = D_PAD_OUT/2.
    # The folded matvec runs at the folded wide dims (d_in=D_MODEL,
    # d_out=8192), then extract_real_imag_pair (conj) + interleave_recombine
    # (right-rotate by t_fold/2). All of these collapse into the existing
    # wide/tall sets (verified empty diff), but provision them explicitly so
    # the keys survive a future dim change. NOTE: the conjugation step (0) is
    # auto-generated by phantom across all chains (merge_bootstrap + bootstrap
    # rely on it); it MUST NOT be added to user_rotation_steps or it gets a
    # single shallow target chain and breaks bootstrap's conjugation.
    irp_rect_fold_steps = list(_irp.irp_required_steps_rect(
        NUM_SLOTS, D_MODEL, _D_OUT_FOLD_MLP, baby_steps=_BABY_STEPS_IRP_MLP_RECT))
    _t_fold = NUM_SLOTS // _D_OUT_FOLD_MLP
    irp_fold_recombine_steps = [
        (NUM_SLOTS - _t_fold // 2) % NUM_SLOTS,  # interleave right-rotate
    ]
    # MLP Wdown is now complex output-FOLDED too (d_out 4096 → d_out_fold 2048,
    # the biggest remaining tall fold). The folded TALL matvec runs the rect
    # machinery at d_out_fold = D_MODEL/2 (alpha doubles 4→8), needing finer
    # input-alignment / reduce rotations than the unfolded (D_PAD_OUT, D_MODEL)
    # tall path. The output is split by extract_real_imag_pair (conj step 0,
    # auto-generated — NOT added) and bridged out (decrypt + numpy recombine +
    # re-encrypt), so no interleave-recombine rotation here.
    _D_OUT_FOLD_DOWN = D_MODEL // 2  # 2048
    irp_rect_tall_fold_steps = list(_irp.irp_required_steps_rect(
        NUM_SLOTS, _D_PAD_OUT_MLP, _D_OUT_FOLD_DOWN,
        baby_steps=_BABY_STEPS_IRP_MLP_RECT))
    # Wq + Wo are now complex output-FOLDED SQUARE matvecs (d=D_TOTAL,
    # d_out_fold = D_TOTAL/2, K=512→256 SCPs each). The folded square matvec
    # runs the TALL-rect machinery at (d_in=D_TOTAL, d_out=D_TOTAL/2, alpha=2),
    # then extract_real_imag_pair (conj step 0, auto-generated — NOT added) and
    # an output SK bridge (decrypt + numpy recombine + re-encrypt). The rect
    # steps here are already a subset of the square IRP set (verified empty
    # diff), but provision them explicitly so the keys survive a dim change.
    irp_sq_fold_steps = list(_irp.irp_required_steps_rect(
        NUM_SLOTS, D_TOTAL, D_TOTAL // 2, baby_steps=_BABY_STEPS_IRP))

    user_steps = sorted(set(
        rms_steps + bsgs_steps + qkt_steps + smx_steps
        + score_v_steps + bcast_steps + irp_steps + sdpa_steps
        + irp_rect_wide_steps + irp_rect_tall_steps
        + irp_rect_fold_steps + irp_fold_recombine_steps
        + irp_rect_tall_fold_steps + irp_sq_fold_steps
    ))
    # Chain-depth buckets (dense pipeline trace, restarts at 16 each
    # bootstrap): BSGS Wq + compute_qkt inner-sum fire post-bootstrap at
    # chain 16 (qkt-class); the dense-softmax sum_reduce_stride fires
    # post-mask at chain 17 (finalize-class); score_times_v's broadcast
    # fires post-softmax at chain 23 (score_v-class). rms steps fire at 16.
    qkt_bucket = set(bsgs_steps) | set(qkt_steps)
    smx_bucket = set(smx_steps)
    # score_v's positive accumulate steps coincide with smx sum_reduce
    # ({4096,8192,16384}); the broadcast (negative) steps are score_v-only.
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
        # FRESHEST_CHAIN=16 is invariant for both legacy (NSL=14) and use17
        # (NSL=16) under evalmod_r=3. Verified post-engine-construction
        # via engine.freshest_chain_index().
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
        # Packed-score softmax: same chain depths as the per-block
        # equivalents. sum_reduce + broadcast fire post-stage-A bootstrap
        # (chain 17 = TARGET_FINALIZE); unpack rotates the post-Goldschmidt
        # weights (chain 23 = TARGET_SCORE_V). Empty sets when packed
        # softmax is disabled.
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
    # Opt-in to the BootstrapTo17Levels chain layout (Lapis-shape, the_lib's
    # prime *counts* but Lapis prime *sizes*). Gives max_user_level = NSL-1
    # = 16 at NSL=17; useful for NUM_SQUARINGS=6 if the resulting bootstrap
    # working memory fits the GPU.
    cfg.use_bootstrap_to_17_levels = os.environ.get("USE_BOOTSTRAP_17") == "1"
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

    # c_per_head is the per-head softmax shift the ciphertext actually
    # receives (via qkt_irp_per_head_sub_plaintext at stage-A). It MUST
    # be computed over the same real keys [0, real_nt) the FHE pipeline
    # reduces over (block_sizes clip masks pad keys to 0). Using the
    # padded keys here is the primary L0/L1 blow-up: at L0 the EOS-pad
    # raw scores are astronomically large, c_per_head is wildly off, and
    # ps_exp saturates -> ~1e45. Variable-nt: real_nt == num_tokens
    # -> identical, byte-for-byte.
    real_nt = _real_nt(num_tokens, query_position)
    Q_np = (xn[query_position] @ Wq_baked.T).reshape(N_HEADS, D_HEAD)
    scores_np = (np.einsum('hd,thd->th', Q_np, K_full_h[:real_nt])
                 / math.sqrt(D_HEAD))
    c_per_head = scores_np.max(0) + 0.5

    # Pack/encrypt K/V for the REAL tokens [0, real_nt) only. The query
    # only ever attends real keys; the EOS-pad keys [real_nt, num_tokens)
    # must never enter the encrypted KV cache. Packing the padded
    # num_tokens at fixed-nt would encode/encrypt ceil(num_tokens/T)
    # ciphertexts (64 at nt=512) instead of ceil(real_nt/T) (8 for
    # real_nt=60). Beyond the obvious slot waste, feeding those extra
    # pad ciphertexts through the engine corrupts accuracy at scale
    # (uniform ~0.3 rel-RMS + L7/L8/L31 ~1e3 blow-ups at fixed-512;
    # survivable but still wrong at fixed-128). pack_kv_blocks already
    # zeros slots past each block_size and fills only block_size real
    # tokens, so packing real_nt yields EXACTLY the variable-nt-real_nt
    # blocks: structurally and numerically identical to a real
    # nt=real_nt prompt. BSGS / rotation / diagonal decomposition is
    # unchanged — only the (real-only) block count differs.
    # Variable-nt: real_nt == num_tokens -> identical, byte-for-byte.
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
# FHE dense token-major attention — THE compute path (IRP machinery deleted).
#
# Kernel contract (same primitives attention_forward_llama uses for Stage A):
#   phantom.bsgs_matmul_preencoded (Wq, d_pad == D_TOTAL, replicated-block I/O)
#   -> phantom.compute_qkt(q, k_shard_list, d_head)  (the multi-shard loop)
#   -> fused multiply_plain (1/sqrt(d_head) * per-head mask * pad-token-zero)
#   -> per-head sub_plain (c_per_head centering).
#
# baby_steps=64 for the d_pad=4096 Wq BSGS: bsgs_required_steps(64) ==
# {1,2,4,8,16,32,64} == inner_sum(D_HEAD=128) steps, all provisioned by
# build_user_steps_mrpc (now built directly from the dense step builders).
#
# Slot geometry is byte-identical to the verified numpy oracle
# blocks.kv_layout_dense (commit 744e61f); the caller validates each layer
# against kv_layout_dense.dense_qkt on the same Q/K.
# ===========================================================================

_DENSE_WQ_BABY_STEPS = 64  # bsgs_required_steps(64) ⊆ provisioned steps


def fhe_attention_dense_scores(ctx, encoder, sk, relin_key, galois_key,
                                xn_query, Wq_baked, K_full_h, c_per_head,
                                real_nt, chain_index):
    """Run the FHE dense token-major QK^T -> scaled/masked/centered scores.

    Stops at the scores ciphertext list (one per shard), post scale*mask
    and post per-head sub(C). Self-contained: encrypts its own dense inputs
    from the teacher-forced numpy Q/K (same as encrypt_layer_inputs_multi /
    the calibration), runs the real kernel, returns the decrypted scores
    plus the oracle scores on the identical Q/K for the validation gate.

    Args:
      xn_query: (D_MODEL,) rmsnormed hidden at the query position.
      Wq_baked: (D_TOTAL, D_MODEL) Wq with R_P (rope@query) pre-applied.
      K_full_h: (real_nt, N_HEADS, D_HEAD) rope-applied + GQA-expanded K.
      c_per_head: (N_HEADS,) per-head softmax shift (real-key max + 0.5).
      real_nt: real token count (== num_tokens for variable-nt).
      chain_index: fresh chain to encode/encrypt the dense inputs at.

    Returns dict:
      'fhe_scores'   : (real_nt, N_HEADS) decrypted, post scale*mask*sub(C)
      'oracle_scores': (real_nt, N_HEADS) kv_layout_dense.dense_qkt - C
      'P', 'n_shards'
    """
    D = D_TOTAL
    H = D_HEAD
    nH = N_HEADS
    P = _dense_fhe.positions_per_ct(real_nt, NUM_SLOTS, D)
    n_shards = _dense_fhe.n_shards_for(real_nt, P)

    # ---- BSGS Wq (d_pad == D_TOTAL); replicated-block x -> dense Q.
    wq_diags = _dense_fhe.bsgs_wq_diags_dense(
        ctx, encoder, Wq_baked, D, _DENSE_WQ_BABY_STEPS, SCALE)
    x_ct = _dense_fhe.encrypt_x_replicated_block(
        ctx, encoder, sk, xn_query, D, NUM_SLOTS, SCALE, chain_index)
    q_ct = phantom.bsgs_matmul_preencoded(ctx, galois_key, x_ct, wq_diags)
    # NO rescale here — mirror attention_forward_llama's canonical BSGS
    # pattern exactly: compute_qkt reads nominal = q.scale() and snaps its
    # product back to it. Rescaling/re-snapping q_ct (the IRP-Wq contract)
    # would mis-set that nominal and corrupt the score magnitude.
    # BSGS output is replicated-block period D_TOTAL == dense pre-broadcast Q
    # (q_slot[tok*D + h*H + j] = Q[h,j] for every token frame). == pack_q_dense.

    # ---- Dense token-major K shards (oracle-exact slot geometry).
    k_cts = _dense_fhe.encrypt_k_dense_shards(
        ctx, encoder, sk, K_full_h, real_nt, P, nH, H,
        NUM_SLOTS, SCALE, chain_index)
    for kc in k_cts:
        phantom.mod_switch_to_inplace(ctx, kc, q_ct.chain_index())

    # ---- The real kernel: elementwise q*k + inner_sum(d_head), per shard.
    raw_score_cts = phantom.compute_qkt(
        ctx, relin_key, galois_key, q_ct, k_cts, H)

    inv_sqrt_d = 1.0 / math.sqrt(float(H))
    fhe_scores = np.zeros((real_nt, nH), dtype=np.float64)
    for b, sc in enumerate(raw_score_cts):
        tok_start = b * P
        # Fused scale * per-head mask * pad-token-zero (one multiply_plain).
        nominal = sc.scale()
        ms_slots = _dense_fhe.dense_scale_mask_slots(
            NUM_SLOTS, H, D, tok_start, P, real_nt, inv_sqrt_d)
        ms_pt = encoder.encode_double_vector(
            ctx, ms_slots, SCALE, sc.chain_index())
        sc = phantom.multiply_plain(ctx, sc, ms_pt)
        sc = phantom.rescale_to_next(ctx, sc)
        sc.set_scale(nominal)
        # Per-head centering sub(C) (mirror IRP stage-A).
        sub_slots = _dense_fhe.dense_per_head_sub_slots(
            NUM_SLOTS, H, D, tok_start, P, real_nt, c_per_head)
        sub_pt = encoder.encode_double_vector(
            ctx, sub_slots, sc.scale(), sc.chain_index())
        sc = phantom.sub_plain(ctx, sc, sub_pt)
        # Decrypt & read base slots tok_local*D + h*H.
        sv = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, sc)),
                      dtype=np.float64)
        for tok_local in range(P):
            tok_abs = tok_start + tok_local
            if tok_abs >= real_nt:
                break
            for h in range(nH):
                fhe_scores[tok_abs, h] = sv[tok_local * D + h * H]

    # ---- Oracle on the IDENTICAL Q/K (the trusted Stage-1 spec).
    # Q[h,j] is exactly xn_query @ Wq_baked.T reshaped (== BSGS output target).
    Q_hd = (np.asarray(xn_query, dtype=np.float64)
            @ np.asarray(Wq_baked, dtype=np.float64).T).reshape(nH, H)
    q_slots = _dense_oracle.pack_q_dense(Q_hd, P)
    q_per_shard = [q_slots for _ in range(n_shards)]
    # K shards: oracle packer with n_kv_heads == nH (K already GQA-expanded).
    k_shards_oracle, _ = _dense_oracle.pack_kv_dense_shards(
        np.asarray(K_full_h, dtype=np.float64),
        np.asarray(K_full_h, dtype=np.float64),  # V unused for QK^T gate
        real_nt, P, nH)
    oracle_scores = _dense_oracle.dense_qkt(
        q_per_shard, k_shards_oracle, nH, H, real_nt, P, inv_sqrt_d)
    oracle_scores = oracle_scores - np.asarray(c_per_head,
                                               dtype=np.float64)[None, :]

    return {
        "fhe_scores": fhe_scores,
        "oracle_scores": oracle_scores,
        "P": P,
        "n_shards": n_shards,
    }


def fhe_attention_dense_softmax(engine, ctx, encoder, sk, relin_key, galois_key,
                                 xn_query, Wq_baked, K_full_h, c_per_head,
                                 real_nt, chain_index):
    """Stage 3 (dense-layout rewrite): FHE dense token-major softmax.

    Extends fhe_attention_dense_scores' QK^T -> scaled/masked/centered scores
    ciphertexts (kept encrypted, NOT decrypted) with the softmax compute:

      ps_exp_init -> damped squarings -> STRICT 0/1 base-slot re-mask
        -> per-shard sum_reduce (stride=D, count=P) -> cross-shard ADD
        -> Goldschmidt softmax_correct  (== exp / per-head Σexp).

    The strict 0/1 re-mask after ps_exp_init is THE poly(0) trap fix
    (kv_layout_dense_fhe.dense_real_base_mask_slots): ps_exp(0)=poly(0)!=0
    (~0.449), so without it the per-head sum-reduce over the P token frames
    would add (nt_pad-real_nt) bogus poly(0) terms and pollute every per-head
    denominator. Mirrors the IRP path's stage-C re-mask exactly.

    Per-shard partial-sum then cross-shard ADD then reciprocal exactly
    mirrors the verified oracle (test_kv_layout_dense.
    test_softmax_denom_cross_shard_sum) AND the IRP multi_ct_softmax_finalize
    (cross-block add -> sum_reduce -> per-block softmax_correct).

    Reuses the SAME softmax primitive chain / constants the IRP path uses:
      damps = softmax_damping_schedule(NUM_SQUARINGS, real_nt, EXTRA_SCALE,
                                       TARGET_MAG)
      e = phantom.ps_exp_init(.., real_nt, NUM_SQUARINGS, EXTRA_SCALE)
      phantom.square_iterations_damped_inplace(.., e, damps)
      w = phantom.softmax_correct(.., e_shard, a_total, ITERS)

    Args / signature identical to fhe_attention_dense_scores.

    Returns dict:
      'fhe_weights'   : (real_nt, N_HEADS) decrypted softmax weights
      'oracle_weights': (real_nt, N_HEADS) numpy oracle softmax (kv_layout_
                        dense.dense_qkt on the SAME Q/K, then stable softmax
                        over the token axis)
      'head_sums'     : (N_HEADS,) per-head Σ_tok fhe_weights (must be ~1.0)
      'P', 'n_shards'
    """
    D = D_TOTAL
    H = D_HEAD
    nH = N_HEADS
    P = _dense_fhe.positions_per_ct(real_nt, NUM_SLOTS, D)
    n_shards = _dense_fhe.n_shards_for(real_nt, P)

    # ---- QK^T (identical to fhe_attention_dense_scores up to raw scores). ----
    wq_diags = _dense_fhe.bsgs_wq_diags_dense(
        ctx, encoder, Wq_baked, D, _DENSE_WQ_BABY_STEPS, SCALE)
    x_ct = _dense_fhe.encrypt_x_replicated_block(
        ctx, encoder, sk, xn_query, D, NUM_SLOTS, SCALE, chain_index)
    q_ct = phantom.bsgs_matmul_preencoded(ctx, galois_key, x_ct, wq_diags)
    k_cts = _dense_fhe.encrypt_k_dense_shards(
        ctx, encoder, sk, K_full_h, real_nt, P, nH, H,
        NUM_SLOTS, SCALE, chain_index)
    for kc in k_cts:
        phantom.mod_switch_to_inplace(ctx, kc, q_ct.chain_index())
    raw_score_cts = phantom.compute_qkt(
        ctx, relin_key, galois_key, q_ct, k_cts, H)

    inv_sqrt_d = 1.0 / math.sqrt(float(H))

    # ---- Stage A: fused scale*mask + per-head sub(C), per shard. ----
    score_cts = []
    for b, sc in enumerate(raw_score_cts):
        tok_start = b * P
        nominal = sc.scale()
        ms_slots = _dense_fhe.dense_scale_mask_slots(
            NUM_SLOTS, H, D, tok_start, P, real_nt, inv_sqrt_d)
        ms_pt = encoder.encode_double_vector(
            ctx, ms_slots, SCALE, sc.chain_index())
        sc = phantom.multiply_plain(ctx, sc, ms_pt)
        sc = phantom.rescale_to_next(ctx, sc)
        sc.set_scale(nominal)
        sub_slots = _dense_fhe.dense_per_head_sub_slots(
            NUM_SLOTS, H, D, tok_start, P, real_nt, c_per_head)
        sub_pt = encoder.encode_double_vector(
            ctx, sub_slots, sc.scale(), sc.chain_index())
        sc = phantom.sub_plain(ctx, sc, sub_pt)
        score_cts.append(sc)

    # ---- Bootstrap after stage A (before ps_exp), mirroring the IRP path's
    # post-stage-A bootstrap (fhe_attention_multi_ct: bootstrap_safe at
    # max_abs=_calib["scores"]). Without it the ps_exp(3) + squarings(NSQ) +
    # mask(1) + softmax_correct(2*ITERS) depth overruns a single fresh chain
    # ("end of modulus switching chain reached"). The dense gate is a
    # self-contained validation harness with `engine` in scope at the call
    # site, so it bootstraps exactly like the live IRP softmax. ----
    _SCORES_CALIB = 45.10  # same static post-stage-A bound the IRP path uses
    score_cts = [
        bootstrap_safe(engine, ctx, encoder, sc,
                       max_abs=_SCORES_CALIB, slot_count=NUM_SLOTS)
        for sc in score_cts
    ]

    # ---- Safety scale (mirrors the IRP path's softmax_safety_scale). ----
    # Goldschmidt softmax_correct's denominator update a ← a(2-a) converges
    # ONLY for a ∈ (0,2). A peaky head's true per-head Σ_tok pipeline(score)
    # can legitimately exceed 2 (observed L0: max per-head denom ≈ 2.06 > 2),
    # which makes softmax_correct DIVERGE for that one head (a₁ = 2.06·(2-2.06)
    # < 0, oscillates) -> garbage weights, per-head-sum |1-Σ| ≈ 55. The IRP
    # path's _stage_c_post folds a `safety_scale` into its stage-C mask so the
    # LARGEST per-head sum lands at SOFTMAX_TARGET (< 2). Softmax weights are
    # scale-invariant ((s·e)/Σ(s·e) == e/Σe) so this is exact. Computed here
    # from the SAME teacher-forced numpy oracle compute_layer_calib_n uses
    # (NOT from any decrypted FHE value) — self-contained for the gate.
    _SOFTMAX_TARGET = 1.5  # == compute_layer_calib_n's SOFTMAX_TARGET
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

    # ---- Dense-layout-correct per-shard pre-bootstrap mean (H2 fix). ----
    # bootstrap_safe assumes a ~mean-zero input; the mean-sub-before /
    # mean-add-after pair must use the shard ciphertext's TRUE mean. The
    # old hardcoded 0.4487 is the IRP head-major layout's empirical mean;
    # the dense token-major shard layout has a different populated/junk
    # fill (per shard: nH base slots per REAL token-frame hold the real
    # pipeline-exp value `_pe_s[tok,h]`; ALL other NUM_SLOTS slots — j>0
    # within real frames AND every slot of padded frames — hold the
    # off-support constant v0 = pipeline(score=0)). Decrypt-probe showed
    # the true per-shard mean is ~0.4466 vs the hardcoded 0.4487 (a
    # consistent -0.0021 mis-centering that biased every post-bootstrap
    # `e` value -> the dominant ~2.6e-3 dense-softmax excess over the IRP
    # floor). Compute it analytically with the SAME ps_exp Chebyshev +
    # damping primitives already used for `_pe_s` (v0 = `_pe_s` evaluated
    # at score 0). ADDITIVE; DENSE_PRE_MEAN_FIX=0 restores the 0.4487
    # functional baseline.
    _pre_mean_fix = os.environ.get("DENSE_PRE_MEAN_FIX", "1") == "1"
    _v0 = 0.0
    for _c in reversed(_cf_s):
        _v0 = _v0 * 0.0 + _c
    for _d in _dmp_s:
        _v0 = _v0 * _v0
        if abs(_d - 1.0) > 1e-12:
            _v0 = _v0 * _d
    _v0 = float(_v0)
    _dense_pre_mean = np.full(n_shards, 0.4487, dtype=np.float64)
    if _pre_mean_fix:
        for _b in range(n_shards):
            _t0 = _b * P
            _t1 = min(_t0 + P, real_nt)
            _n_real_frames = max(_t1 - _t0, 0)
            _pop_sum = float(_pe_s[_t0:_t1, :].sum())
            _n_pop = _n_real_frames * nH
            _n_junk = NUM_SLOTS - _n_pop
            _dense_pre_mean[_b] = (_pop_sum + _n_junk * _v0) / NUM_SLOTS
    if os.environ.get("PROBE_DENSE_SMX") == "1":
        print(f"  [PROBE-SMX] v0(pipeline@0)={_v0:.6f}  "
              f"dense per-shard pre-mean (fix={_pre_mean_fix}): "
              f"{np.array2string(_dense_pre_mean, precision=6)}")

    # DIAGNOSTIC ONLY (PROBE_DENSE_SMX=1): compute the IRP path's safety_scale
    # estimator (compute_layer_calib_n L274-280: exp(scores).sum().max() *
    # TARGET_MAG) on the IDENTICAL scores and report it next to the dense
    # estimator (cheb-poly + damped squarings sum). Pure numpy, zero
    # ciphertext effect. When unset this block is not entered.
    if os.environ.get("PROBE_DENSE_SMX") == "1":
        _irp_sum_t_exp = np.exp(_scc_s).sum(axis=0)            # per-head Σ exp
        _irp_expected_max = float(_irp_sum_t_exp.max() * TARGET_MAG)
        _irp_safety = (_SOFTMAX_TARGET / _irp_expected_max
                       if _irp_expected_max > _SOFTMAX_TARGET else 1.0)
        # Post-pipeline per-head denom magnitude the Goldschmidt `a` SEES,
        # i.e. after folding each estimator's safety_scale (a == that estimate
        # * safety, the value softmax_correct must keep inside (0,2)).
        _dense_a_max = _max_head_denom * safety_scale
        _irp_a_max = _irp_expected_max * _irp_safety
        # Per-head dense `a` distribution (numpy proxy = the simulated denom):
        _ph = _pe_s.sum(axis=0) * safety_scale
        print(f"  [PROBE-SMX] real_nt={real_nt} P={P} n_shards={n_shards}")
        print(f"  [PROBE-SMX] DENSE estimator: max_head_denom(cheb+damp)="
              f"{_max_head_denom:.6f}  safety_scale={safety_scale:.6f}")
        print(f"  [PROBE-SMX] IRP   estimator: expected_max(exp*TM)="
              f"{_irp_expected_max:.6f}  safety_scale={_irp_safety:.6f}")
        print(f"  [PROBE-SMX] safety_scale RATIO dense/irp="
              f"{(safety_scale / _irp_safety if _irp_safety else 0.0):.6f}")
        print(f"  [PROBE-SMX] post-fold a_max  DENSE={_dense_a_max:.6f}  "
              f"IRP={_irp_a_max:.6f}  (Goldschmidt window (0,2), target 1.5)")
        print(f"  [PROBE-SMX] dense per-head a (post-fold) numpy proxy: "
              f"min={_ph.min():.5f} max={_ph.max():.5f} "
              f"mean={_ph.mean():.5f} >1.9:{int((_ph>1.9).sum())} "
              f">1.99:{int((_ph>1.99).sum())} <0.05:{int((_ph<0.05).sum())}")

    # ---- Stage B: ps_exp_init + damped squarings, per shard. ----
    # real_nt (NOT nt_pad): paired with the damps; matches IRP _stage_b_ps_exp
    # + softmax_damping_schedule (the t^(-1) baked in must be the real summed
    # key count). A per-shard constant -> identical across shards.
    damps = softmax_damping_schedule(
        NUM_SQUARINGS, real_nt, EXTRA_SCALE, TARGET_MAG)
    e_cts = []
    for b, sc in enumerate(score_cts):
        e_ct = phantom.ps_exp_init(
            ctx, encoder, relin_key, sc,
            real_nt, NUM_SQUARINGS, EXTRA_SCALE)
        phantom.square_iterations_damped_inplace(
            ctx, encoder, relin_key, e_ct, damps)
        # Mean-subtract BEFORE the bootstrap, mean-add AFTER — EXACTLY the IRP
        # path's _stage_b_ps_exp (sub) / _stage_c_post (add). bootstrap_safe
        # assumes an approximately mean-zero input (its docstring); but after
        # ps_exp + damped squarings the off-support slots all sit at
        # pipeline(0) ≈ TARGET_MAG ≈ 0.45, so the ciphertext mean is ≈ 0.45.
        # Bootstrapping that ~0.45-mean signal directly corrupts it (the
        # bootstrap mod-reduction polynomial is centred at 0) — observed:
        # post-bootstrap exp values came out ≈ -0.39 (negative! exp must be
        # > 0), a_total ≈ -22.7, Goldschmidt blow-up to ~1e88. Subtracting
        # _PRE_FINSMX_MEAN first centres the signal for the bootstrap, then
        # adding it back restores the true pipeline values. _PRE_FINSMX_MEAN
        # is the SAME functional-baseline constant the IRP path uses.
        # H2 FIX: use the dense-layout-correct per-shard mean (computed
        # above from the same ps_exp+damping primitives) instead of the
        # IRP-tuned 0.4487 so the bootstrap input is actually mean-zero.
        _PRE_FINSMX_MEAN = float(_dense_pre_mean[b])
        # DIAGNOSTIC ONLY (PROBE_DENSE_SMX=1): decrypt this dense shard ct's
        # TRUE mean right before the mean-sub. bootstrap_safe assumes an
        # ~mean-zero input; `_PRE_FINSMX_MEAN` should equal this true mean.
        # Any gap is added uniformly to every slot by the mean-add-back and
        # de-centers the bootstrap input -> value-independent absolute `e`
        # error (H2). Decrypt only; zero ciphertext effect.
        if os.environ.get("PROBE_DENSE_SMX") == "1":
            _pre = np.array(
                encoder.decode_double_vector(ctx, sk.decrypt(ctx, e_ct)),
                dtype=np.float64)
            print(f"  [PROBE-SMX] shard {b} e PRE-meansub: true ct "
                  f"mean={_pre.mean():.6f}  hardcoded={_PRE_FINSMX_MEAN:.6f}"
                  f"  gap(true-hc)={_pre.mean() - _PRE_FINSMX_MEAN:+.6f}  "
                  f"max|.|={np.abs(_pre).max():.4e}")
        _mean_pt = encoder.encode_double_vector(
            ctx, np.full(NUM_SLOTS, _PRE_FINSMX_MEAN, dtype=np.float64),
            e_ct.scale(), e_ct.chain_index())
        e_ct = phantom.sub_plain(ctx, e_ct, _mean_pt)
        e_ct = bootstrap_safe(engine, ctx, encoder, e_ct,
                              max_abs=TARGET_MAG, slot_count=NUM_SLOTS)
        _mean_pt2 = encoder.encode_double_vector(
            ctx, np.full(NUM_SLOTS, _PRE_FINSMX_MEAN, dtype=np.float64),
            e_ct.scale(), e_ct.chain_index())
        e_ct = phantom.add_plain(ctx, e_ct, _mean_pt2)
        # THE poly(0) trap fix: strict 0/1 base-slot mask applied AFTER the
        # bootstrap (== the IRP path's stage-C re-mask order). ps_exp(0) =
        # poly(0) ≈ 0.45 != 0, so EVERY padded / mid-block (j>0) slot holds
        # ~0.45 after exp+squarings; the bootstrap then ALSO injects ~1e-4
        # noise at those slots. Masking AFTER the bootstrap with a clean
        # encoded-zero plaintext drives those slots to EXACT 0 (a
        # mask-before-bootstrap leaves ~6.7e-5 bootstrap residue, which makes
        # Goldschmidt x ← x(2-a) explode where a≈0 — observed: CUDA error /
        # 4.2x weight inflation). With e EXACTLY 0 off-support, softmax_correct
        # keeps x = 0 there (no explosion) and the per-head sum-reduce sees
        # ONLY the real_nt legitimate exp values. `safety_scale` is folded
        # into this same multiply_plain (mirrors IRP stage-C's mask*safety),
        # keeping the peaky-head Goldschmidt denominator inside (0,2).
        tok_start = b * P
        mask_slots = _dense_fhe.dense_real_base_mask_slots(
            NUM_SLOTS, H, D, tok_start, P, real_nt, keep_value=safety_scale)
        e_nominal = e_ct.scale()
        mask_pt = encoder.encode_double_vector(
            ctx, mask_slots, SCALE, e_ct.chain_index())
        e_ct = phantom.multiply_plain(ctx, e_ct, mask_pt)
        e_ct = phantom.rescale_to_next(ctx, e_ct)
        e_ct.set_scale(e_nominal)
        # DIAGNOSTIC ONLY (PROBE_DENSE_SMX=1): decrypt this shard's post-
        # bootstrap-masked `e` (the softmax NUMERATOR Goldschmidt consumes)
        # and compare to the analytic target TARGET_MAG*exp(scores_post_C)*
        # safety_scale on the SAME shard's tokens. Localizes whether the
        # softmax error is in `e` (ps_exp/bootstrap/mask) or in the e/a
        # division. Decrypt only; zero ciphertext effect.
        if os.environ.get("PROBE_DENSE_SMX") == "1":
            _ev = np.array(
                encoder.decode_double_vector(ctx, sk.decrypt(ctx, e_ct)),
                dtype=np.float64)
            _e_fhe, _e_ana = [], []
            for _tl in range(P):
                _ta = tok_start + _tl
                if _ta >= real_nt:
                    break
                for _h in range(nH):
                    _e_fhe.append(_ev[_tl * D + _h * H])
                    _e_ana.append(
                        TARGET_MAG
                        * math.exp(float(_scc_s[_ta, _h]))
                        * safety_scale)
            if _e_fhe:
                _ef = np.asarray(_e_fhe); _ea = np.asarray(_e_ana)
                _den = np.linalg.norm(_ea)
                _rr = (float(np.linalg.norm(_ef - _ea) / _den)
                       if _den > 0 else 0.0)
                print(f"  [PROBE-SMX] shard {b} e (FHE post-mask) vs "
                      f"analytic TM*exp*ss: rel-RMS={_rr:.4e} "
                      f"‖efhe‖={np.linalg.norm(_ef):.5f} "
                      f"‖eana‖={_den:.5f} "
                      f"max_abs_err={float(np.abs(_ef-_ea).max()):.3e}")
        e_cts.append(e_ct)

    # ---- Stage C: per-shard partial sum -> cross-shard ADD -> reciprocal. ----
    # Per shard: sum_reduce_stride(stride=D, count=P) folds the P token frames
    # within that shard cyclically (slot len == P*D), landing the per-shard
    # per-head partial Σ_{tok in shard} e at EVERY base slot tok*D+h*H.
    # Cross-shard ADD the partials -> the true per-head total Σ_all_tok e at
    # every base slot. THEN Goldschmidt softmax_correct (== oracle
    # test_softmax_denom_cross_shard_sum + IRP multi_ct_softmax_finalize).
    a_total = None
    for e_ct in e_cts:
        partial = sum_reduce_stride(ctx, galois_key, e_ct, D, P)
        if a_total is None:
            a_total = partial
        else:
            a_total = phantom.add(ctx, a_total, partial)

    # Mask + rescale + set_scale(user_scale) on the accumulated denominator,
    # EXACTLY the IRP path's multi_ct_softmax_finalize step 3 (a_masked =
    # multiply_plain(a, head_first_slot_mask); rescale; set_scale(user_scale)).
    # a_total has accumulated noise from log2(P) sum_reduce rotate-adds +
    # (n_shards-1) cross-shard adds — fed straight into Goldschmidt (which
    # squares `a` 2*ITERS times) that noise is amplified. The IRP path resets
    # it with a mask*rescale before the broadcast; omitting that reset is the
    # ~2.7e-3 extra error the dense path showed over the IRP fidelity floor.
    # The mask (keep_value=1.0 at base slots) also re-zeros any sum_reduce
    # residue at j>0 so the broadcast stays clean. 1 level (chain budget OK:
    # post-bootstrap fresh chain has ~16 levels; e-mask 1 + a-mask 1 +
    # softmax_correct 2*ITERS=12 = 14 <= 16).
    _a_reset_slots = _dense_fhe.dense_real_base_mask_slots(
        NUM_SLOTS, H, D, 0, P, P * n_shards, keep_value=1.0)
    _a_nominal = a_total.scale()
    _a_mask_pt = encoder.encode_double_vector(
        ctx, _a_reset_slots, SCALE, a_total.chain_index())
    a_total = phantom.multiply_plain(ctx, a_total, _a_mask_pt)
    a_total = phantom.rescale_to_next(ctx, a_total)
    a_total.set_scale(engine.user_scale())

    # DIAGNOSTIC ONLY (PROBE_DENSE_SMX=1): decrypt the ACTUAL FHE per-head
    # denominator `a_total` the dense Goldschmidt softmax_correct will consume,
    # read at the base slots (tok*D + h*H) for the populated tokens. This is
    # the real `a` whose magnitude vs the (0,2)/1.5 window determines the
    # Goldschmidt noise floor. Decrypt only — zero ciphertext effect.
    if os.environ.get("PROBE_DENSE_SMX") == "1":
        _av = np.array(
            encoder.decode_double_vector(ctx, sk.decrypt(ctx, a_total)),
            dtype=np.float64)
        _a_base = []
        for _b in range(n_shards):
            _ts = _b * P
            for _tl in range(P):
                _ta = _ts + _tl
                if _ta >= real_nt:
                    break
                for _h in range(nH):
                    _a_base.append(_av[_tl * D + _h * H])
        _a_base = np.asarray(_a_base, dtype=np.float64)
        # Per-head Σ over tokens of the decrypted base-slot `a` (each base
        # slot ALREADY holds the per-head total after sum_reduce+cross-add,
        # so the per-head a == the value at any one base slot of that head;
        # collapse over tokens via the first token's row).
        _a_head = _av[np.array([0 * D + _h * H for _h in range(nH)])]
        print(f"  [PROBE-SMX] FHE a_total (decrypted, base slots, real "
              f"tokens): min={_a_base.min():.6f} max={_a_base.max():.6f} "
              f"mean={_a_base.mean():.6f}")
        print(f"  [PROBE-SMX] FHE per-head a (tok0 row, nH={nH}): "
              f"min={_a_head.min():.6f} max={_a_head.max():.6f} "
              f">1.9:{int((_a_head>1.9).sum())} >1.99:{int((_a_head>1.99).sum())} "
              f">=2.0:{int((_a_head>=2.0).sum())} <0.05:{int((_a_head<0.05).sum())}")

    # Intra-head broadcast of the per-head sum across the d_head block
    # (mirrors the verified oracle kv_layout_dense._broadcast_within_heads
    # and the IRP path's a_bc broadcast). After sum_reduce_stride the per-head
    # denominator sits ONLY at the j=0 base slot tok*D+h*H; j>0 slots hold the
    # sum of ~0 (strict-masked) residues ≈ 0. Goldschmidt softmax_correct
    # updates `a` GLOBALLY: a slot with a≈0 makes x ← x(2-a) ≈ 2x DOUBLE every
    # iteration, and 2*ITERS doublings across ~30k near-zero slots blow up the
    # ciphertext's global scale/noise (CKKS rescale/relin is shared) — that
    # is why softmax_correct returned 4.2x-inflated weights even though
    # a_total at the base slots was numerically correct (FHE/numpy ratio
    # 1.007). Broadcasting the per-head sum to ALL d_head slots keeps
    # a ∈ (0,2) near the per-head sum everywhere -> Goldschmidt converges
    # globally; j>0 slots have e≈0 so their weight≈0 (harmless), base slots
    # get the correct e/Σe. Right-rotation broadcast == oracle's
    # rotate_right(bstride) for bstride=d_head//2..1. Steps {-64..-1} are all
    # pre-provisioned (sdpa/irp) -> ZERO new galois keys.
    # Doubling broadcast, EXACTLY the IRP path's proven a_bc pattern
    # (multi_ct_softmax_finalize step 3): s = 1,2,4,...,H/2 with
    # rotate(-s) then add. Each step doubles the filled span so the j=0
    # per-head value spreads to all H slots of the head block.
    #
    # NOTE: the per-token weight readout reads ONLY the j=0 base slot
    # tok*D+h*H, where sum_reduce already placed the correct per-head sum.
    # Now that the post-bootstrap strict mask makes e EXACTLY 0 at j>0
    # (≈2e-7), Goldschmidt x=0 stays 0 there regardless of a — so the
    # broadcast is no longer required for correctness and its log2(H)=7
    # rotate-adds only inject extra noise into a_total. Default: SKIP it.
    # DENSE_SMX_BCAST=1 restores the broadcast (kept for the original
    # explosion-avoidance rationale / debugging).
    if os.environ.get("DENSE_SMX_BCAST") == "1":
        a_bc = a_total
        s = 1
        while s < H:
            rot = phantom.rotate(ctx, a_bc, -int(s), galois_key)
            a_bc = phantom.add(ctx, a_bc, rot)
            s <<= 1
        a_total = a_bc

    a_chain = a_total.chain_index()
    fhe_weights = np.zeros((real_nt, nH), dtype=np.float64)
    _probe_smx = os.environ.get("PROBE_DENSE_SMX") == "1"
    _w_recon = np.zeros((real_nt, nH), dtype=np.float64) if _probe_smx else None
    if _probe_smx:
        _a_dec = np.array(
            encoder.decode_double_vector(ctx, sk.decrypt(ctx, a_total)),
            dtype=np.float64)
    for b, e_ct in enumerate(e_cts):
        if e_ct.chain_index() != a_chain:
            e_ct = phantom.mod_switch_to(ctx, e_ct, a_chain)
        # DIAGNOSTIC ONLY: decrypt the EXACT `e` fed to Goldschmidt (post
        # mod-switch) and reconstruct w = e/a in numpy at the base slots.
        # If w_recon ≈ oracle but the FHE softmax_correct output diverges,
        # the residual is the FHE Goldschmidt division; if w_recon already
        # diverges, the residual is in `e`/`a` (ps_exp domain). Decrypt
        # only; the ciphertext path is unchanged.
        if _probe_smx:
            _e_dec = np.array(
                encoder.decode_double_vector(ctx, sk.decrypt(ctx, e_ct)),
                dtype=np.float64)
            _ts = b * P
            for _tl in range(P):
                _ta = _ts + _tl
                if _ta >= real_nt:
                    break
                for _h in range(nH):
                    _sl = _tl * D + _h * H
                    _ad = _a_dec[_sl]
                    _w_recon[_ta, _h] = (_e_dec[_sl] / _ad
                                         if abs(_ad) > 1e-30 else 0.0)
        wb = phantom.softmax_correct(
            ctx, encoder, relin_key, e_ct, a_total, ITERS)
        wv = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, wb)),
                      dtype=np.float64)
        tok_start = b * P
        for tok_local in range(P):
            tok_abs = tok_start + tok_local
            if tok_abs >= real_nt:
                break
            for h in range(nH):
                fhe_weights[tok_abs, h] = wv[tok_local * D + h * H]

    head_sums = fhe_weights.sum(axis=0)  # (nH,) — must be ~1.0

    # ---- Oracle softmax on the IDENTICAL Q/K (the trusted Stage-1 spec). ----
    Q_hd = (np.asarray(xn_query, dtype=np.float64)
            @ np.asarray(Wq_baked, dtype=np.float64).T).reshape(nH, H)
    q_slots = _dense_oracle.pack_q_dense(Q_hd, P)
    q_per_shard = [q_slots for _ in range(n_shards)]
    k_shards_oracle, _ = _dense_oracle.pack_kv_dense_shards(
        np.asarray(K_full_h, dtype=np.float64),
        np.asarray(K_full_h, dtype=np.float64),
        real_nt, P, nH)
    oracle_scores = _dense_oracle.dense_qkt(
        q_per_shard, k_shards_oracle, nH, H, real_nt, P, inv_sqrt_d)
    # Stable softmax over the token axis (axis=0) — shift-invariant, so the
    # c_per_head centering the FHE path applies is irrelevant to the weights.
    _os = oracle_scores - oracle_scores.max(axis=0, keepdims=True)
    _oe = np.exp(_os)
    oracle_weights = _oe / _oe.sum(axis=0, keepdims=True)

    # DIAGNOSTIC ONLY (PROBE_DENSE_SMX=1): w_recon = (decrypted e)/(decrypted
    # a) vs oracle and vs the FHE softmax_correct output. Splits the dense
    # softmax error into (i) e/a-domain (ps_exp+bootstrap+mask) and (ii) the
    # FHE Goldschmidt division residual.
    if os.environ.get("PROBE_DENSE_SMX") == "1" and _w_recon is not None:
        def _rr(a, b):
            a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
            n = np.linalg.norm(b)
            return float(np.linalg.norm(a - b) / n) if n > 0 else 0.0
        print(f"  [PROBE-SMX] w_recon(e_dec/a_dec) vs oracle  "
              f"rel-RMS={_rr(_w_recon, oracle_weights):.4e}")
        print(f"  [PROBE-SMX] w_recon(e_dec/a_dec) vs FHE_w   "
              f"rel-RMS={_rr(_w_recon, fhe_weights):.4e}  "
              f"(FHE Goldschmidt-division residual)")
        print(f"  [PROBE-SMX] FHE_w vs oracle (gate softmax (a)) "
              f"rel-RMS={_rr(fhe_weights, oracle_weights):.4e}")

    return {
        "fhe_weights": fhe_weights,
        "oracle_weights": oracle_weights,
        "head_sums": head_sums,
        "P": P,
        "n_shards": n_shards,
    }


def fhe_attention_dense_full(engine, ctx, encoder, sk, relin_key, galois_key,
                              xn_query, Wq_baked, K_full_h, V_full_h, Wo,
                              c_per_head, real_nt, chain_index,
                              layer_idx=None, P_local=None,
                              q_max_abs=None):
    """Stage 4 (dense-layout rewrite): close the dense attention block.

    Runs the FULL dense token-major attention pipeline end-to-end, ENTIRELY
    in FHE, returning the post-Wo attention-output ciphertext (replicated-
    block period D_TOTAL) so the caller can wire it into the residual stream:

      QK^T (compute_qkt)  ->  scale*mask + per-head sub(C)  ->  bootstrap
        ->  ps_exp_init + damped squarings  ->  bootstrap (mean-centered)
        ->  STRICT 0/1 base-slot re-mask (the poly(0) trap fix)
        ->  per-shard sum_reduce(stride=D, count=P) -> cross-shard ADD
        ->  a-reset mask + Goldschmidt softmax_correct  (== exp / Σexp)
            [softmax weights kept ENCRYPTED per shard — token-major]
        ->  score_times_v (src/attention.cu kernel, AS-IS) over the
            softmax-weight cts + token-major V shards: mask base ->
            negative-stride d_head broadcast -> ×V -> +d_total accumulate
            over P -> cross-shard ADD.  Because P*D == NUM_SLOTS exactly
            (8*4096 == 32768), the kernel's step-4 accumulate doubling is a
            full cyclic sum-reduce that REPLICATES the per-head attention
            output into ALL NUM_SLOTS/D_TOTAL periods -> slot[k*D + h*H + j]
            = Σ_tok w[tok,h]·V[tok,h,j] for every period k.  That IS the
            replicated-block period-D_pad layout BSGS Wo consumes (identical
            to encrypt_x_replicated_block's Wq input) — no phantom.replicate
            needed (its -4096/-8192 galois steps are NOT provisioned).
        ->  BSGS Wo (bsgs_matmul_preencoded, d_pad == D_TOTAL,
            baby_steps == _DENSE_WQ_BABY_STEPS == 64; bsgs_required_steps(64)
            = {1,2,4,8,16,32,64} ALL already provisioned -> ZERO new keys).
            Output: replicated-block period D_TOTAL, o_ct[k*D + i] = O[i].

    The QK^T -> softmax stages are byte-identical to fhe_attention_dense_
    softmax (same constants, same bootstraps, same poly(0)-trap re-mask,
    same safety_scale, same a-reset); the ONLY difference is the softmax
    weights are NOT decrypted — they stay as per-shard token-major
    ciphertexts fed straight into the score_times_v kernel.

    Args:
      xn_query : (D_MODEL,) rmsnormed hidden at the query position.
      Wq_baked : (D_TOTAL, D_MODEL) Wq with R_P (rope@query) pre-applied.
      K_full_h : (real_nt, N_HEADS, D_HEAD) rope-applied + GQA-expanded K.
      V_full_h : (real_nt, N_HEADS, D_HEAD) GQA-expanded V (NO rope).
      Wo       : (D_MODEL, D_TOTAL) output projection (NO R_P).
      c_per_head : (N_HEADS,) per-head softmax shift (real-key max + 0.5).
      real_nt  : real token count (== num_tokens for variable-nt).
      chain_index : fresh chain to encode/encrypt the dense inputs at.

    Returns dict:
      'o_ct'        : post-Wo attention-output ciphertext (replicated-block
                      period D_TOTAL); o_ct decoded[i] == attn_out[i] for
                      i in [0, D_MODEL).  THIS is wired into the residual.
      'fhe_attn_o'  : (N_HEADS, D_HEAD) decrypted score·V output (pre-Wo)
      'oracle_attn_o': (N_HEADS, D_HEAD) kv_layout_dense.dense_score_v on
                       the IDENTICAL Q/K/V softmax weights (trusted spec)
      'fhe_out'     : (D_MODEL,) decrypted post-Wo attention output
      'oracle_out'  : (D_MODEL,) numpy Wo @ (flattened oracle score·V)
      'P', 'n_shards'
    """
    D = D_TOTAL
    H = D_HEAD
    nH = N_HEADS
    P = _dense_fhe.positions_per_ct(real_nt, NUM_SLOTS, D)
    n_shards = _dense_fhe.n_shards_for(real_nt, P)


    # ---- QK^T via IRP-Wq (Cachemir §4.1) + compute_qkt_irp (Cachemir §5.1).
    # Wq: K=d²/N=512 SCPs (8× fewer than dense BSGS's 4096); irp_matvec_host
    # computes y = x @ M so we pass Wq_baked.T to get q = Wq_baked @ xn_query.
    # Post-matvec scale snap SCALE^2 → SCALE, then bootstrap_safe refreshes
    # chain so compute_qkt_irp sees q at canonical scale. q stays in IRP
    # stride-t layout (slot[i*t]=q[i]) — directly consumed by compute_qkt_irp.
    from blocks import irp as _irp
    _BABY_STEPS_IRP_Q = 16  # M=16, G=32 for d=4096 K=512 (~sqrt(K))
    # COMPLEX OUTPUT-FOLDED Wq (K=512→256 SCPs, the cleanest square fold).
    # encode_irp_diagonals_folded_host folds the output columns into the imag
    # part (d×d → d×(d/2) tall rect, alpha=2), so the folded matvec runs the
    # tall-rect machinery at d_out_fold = D/2 and consumes its input in the
    # TALL layout for (d_in=D, d_out=D/2). The result is a complex ct split by
    # extract_real_imag_pair, then SK-bridged: decrypt both halves, recombine
    # to natural length-D q in numpy, re-encrypt FRESH in the stride-t_k
    # layout compute_qkt_irp consumes (slot[i*t_k]=q[i]). The bridge is
    # near-lossless and refreshes the chain (q enters compute_qkt_irp fresh,
    # giving the Stage-A bootstrap maximum headroom — strictly safer than the
    # old lazy-leveled q).
    _D_OUT_FOLD_Q = D // 2  # 2048
    wq_irp = _irp_cache.wq_plaintexts_cached(
        ctx, encoder,
        np.ascontiguousarray(np.asarray(Wq_baked, dtype=np.float64).T),
        N=NUM_SLOTS, d=D, scale=SCALE, baby_steps=_BABY_STEPS_IRP_Q,
        layer_idx=layer_idx, P_local=P_local)
    # Folded input: permuted TALL layout for (d_in=D, d_out=D/2).
    x_irp_ct = _irp.encrypt_irp_input_rect(
        ctx, encoder, sk, np.asarray(xn_query, dtype=np.float64),
        N=NUM_SLOTS, d_in=D, d_out=_D_OUT_FOLD_Q, scale=SCALE,
        chain_index=chain_index)
    # Lazy-level: drop the folded-IRP input to a deep chain so the rotation-
    # heavy matvec runs at few RNS limbs (cheap). The folded matvec + conj-
    # split consumes ~1 extra level vs the unfolded path, but the output is
    # immediately SK-bridged (decrypt + re-encrypt fresh below), so there is
    # no downstream bootstrap-with-scaling constraint on these cts — only the
    # matvec's own ~2-3 levels matter. Target input user_level 9 leaves the
    # pre-decrypt cts at user_level ~12-13 (< max). mod_switch_to drops limbs
    # cleanly (no added noise). Tall masks at the FOLDED dim d_out=D/2:
    # input_mask (square at d=D/2, at input chain) + sub_mask (rect at chain+1).
    _wq_target_ci = engine.user_level_chain_index(12)
    if x_irp_ct.chain_index() < _wq_target_ci:
        x_irp_ct = phantom.mod_switch_to(ctx, x_irp_ct, _wq_target_ci)
    _input_mask_q = _irp.encode_irp_mask(
        ctx, encoder, NUM_SLOTS, _D_OUT_FOLD_Q, SCALE, x_irp_ct.chain_index())
    _sub_mask_q = _irp.encode_irp_mask_rect(
        ctx, encoder, NUM_SLOTS, D, _D_OUT_FOLD_Q, SCALE,
        x_irp_ct.chain_index() + 1)
    _q_complex = _irp.irp_matvec_folded_host(
        ctx, encoder, galois_key, x_irp_ct, wq_irp,
        N=NUM_SLOTS, d=D, baby_steps=_BABY_STEPS_IRP_Q,
        sub_mask_pt=_sub_mask_q, input_mask_pt=_input_mask_q)
    # Snap SCALE^2 → SCALE before the conj-split (pipeline convention).
    _q_complex = phantom.rescale_to_next(ctx, _q_complex)
    _q_complex.set_scale(SCALE)
    _q_re, _q_im = _irp.extract_real_imag_pair(
        ctx, encoder, galois_key, _q_complex, NUM_SLOTS, SCALE)
    # OUTPUT SK BRIDGE: decrypt both halves, recombine to natural length-D q
    # in numpy (TALL fold decode: real=q[:D/2], imag=q[D/2:]), then re-encrypt
    # in the stride-t_k layout compute_qkt_irp consumes (slot[i*t_k]=q[i]).
    # CRITICAL: re-encrypt at the SAME chain the OLD unfolded path produced
    # (user_level 11 = input ul9 + matvec mask rescale +1 + the external
    # SCALE^2→SCALE rescale +1) so the entire downstream §5.1 / Stage-A /
    # softmax / score_v chain budget — and the finalize(17)/score_v(23) galois
    # target chains tuned for it — are unchanged. (Re-encrypting fresh would
    # shift every downstream op ~11 levels shallower than its galois key's
    # target chain → illegal memory access.) The K cache below encodes at
    # q_ct.chain_index(), so it tracks this chain automatically.
    _q_target_ci = engine.user_level_chain_index(11)
    _t_fold_q = NUM_SLOTS // _D_OUT_FOLD_Q   # 16
    _q_dec_re = np.asarray(encoder.decode_double_vector(
        ctx, sk.decrypt(ctx, _q_re)), dtype=np.float64)
    _q_dec_im = np.asarray(encoder.decode_double_vector(
        ctx, sk.decrypt(ctx, _q_im)), dtype=np.float64)
    _q_lo = _q_dec_re[::_t_fold_q][:_D_OUT_FOLD_Q]
    _q_hi = _q_dec_im[::_t_fold_q][:_D_OUT_FOLD_Q]
    _q_natural = np.concatenate([_q_lo, _q_hi])   # natural order, length D
    q_ct = _irp.encrypt_irp_input(
        ctx, encoder, sk, _q_natural,
        N=NUM_SLOTS, d=D, scale=SCALE, chain_index=_q_target_ci)
    # Cachemir §5.1 compute_qkt_irp on IRP-Wq output. K cache packs t_k =
    # NUM_SLOTS//D tokens per ct in interleaved layout
    # (slot[h*d_head*t + r*t + p] = K[c*t + p, h, r]); compute_qkt_irp on
    # (q_ct, k_chunk_ct) yields scores at slot[h*d_head*t + tok_local].
    # SK bridge per chunk: decrypt §5.1 scores, repack into dense Stage-A
    # base-slot layout (slot[tok_local*D + h*H]) per kv_layout_dense_fhe.py:
    # 158 so downstream Stage A/softmax/score_v consumes unchanged.
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
    # ---- Path B: IRP-native attention chain. Eliminates dense Stage A/B/C +
    # C++ score_times_v + IRP-Wo input SK bridge. Single-ct IRP layout across
    # the chain: slot[h*d_head*t + tok] = m[tok, h].
    inv_sqrt_d = 1.0 / math.sqrt(float(H))
    # Pad real_nt to next pow2 for finalize_softmax_irp_t (which asserts pow2>=2).
    # For real_nt=60 -> 64; for real_nt=512 -> 512.
    nt_pad = 1
    while nt_pad < max(2, real_nt):
        nt_pad <<= 1

    # ---- Per-chunk compute_qkt_irp + per-chunk Stage A mask*scale (Section 1
    # partial-junk fix: mask each chunk BEFORE tree-agg so the partial-junk
    # in slots [h*1024+t..h*1024+1023] doesn't pollute valid token slots in
    # the global ct).
    score_cts_irp = []
    # Per-chunk mask*scale plaintext shared across chunks (every chunk has
    # t=8 valid slots per head at offsets [0,t)). Encoded LAZILY at the
    # post-compute_qkt_irp chain (q.chain + 1 after the rescale inside
    # compute_qkt_irp).
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
        # Per-chunk mask*scale: keep slot[h*1024+p] = inv_sqrt_d * m[c*t+p, h]
        # for p in [0, t); zero junk slots so tree-agg is collision-safe.
        nominal = sc.scale()
        sc = phantom.multiply_plain(ctx, sc, _ms_pt)
        sc = phantom.rescale_to_next(ctx, sc)
        sc.set_scale(nominal)
        score_cts_irp.append(sc)

    # ---- Tree-aggregate the n_chunks_k per-chunk score cts into one global
    # ct with slot[h*1024 + tok] = (m[tok, h] - 0) / sqrt(d_head) for h<nH,
    # tok<real_nt; zero elsewhere within each head's first-nt_pad slots.
    # Pre-condition: n_chunks_k must be a power of 2 (= nt_pad / t_k).
    # nt_pad/t_k = 64/8 = 8 chunks (3 levels of tree).
    n_chunks_pow2 = nt_pad // t_k
    # Pad score_cts_irp to n_chunks_pow2 with zero ciphertexts (mask*0 trick:
    # encode a zero ct at the chunk-mask chain so the tree-add is a no-op).
    if len(score_cts_irp) < n_chunks_pow2:
        # Create a zero ct at the same chain/scale as the masked score cts.
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

    # ---- Global per-head sub(c_per_head). Keeps slot[h*1024 + tok] valid
    # for tok in [0, real_nt) — the helper only writes the first real_nt
    # slots per head, so slots [real_nt, nt_pad) (the pad-to-pow2 buffer)
    # are untouched and remain zero from the per-chunk mask above.
    _sub_pt = _attn.qkt_irp_per_head_sub_plaintext(
        ctx, encoder, d_head=H, d_total=D, num_tokens=real_nt, t=t_k,
        c_per_head=c_per_head, chain_index=S_global.chain_index(),
        encode_scale=S_global.scale())
    S_global = phantom.sub_plain(ctx, S_global, _sub_pt)

    # ---- Bootstrap-1 (post-Stage-A). Single ct vs dense's per-shard 4×.
    _SCORES_CALIB = 45.10
    S_global = bootstrap_safe(
        engine, ctx, encoder, S_global,
        max_abs=_SCORES_CALIB, slot_count=NUM_SLOTS)

    # ---- Safety scale + global pre-bootstrap mean (numpy oracle; layout-
    # agnostic). Identical derivation to the dense path; only the mean
    # reduction shape changes (global, not per-shard).
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

    # Global pre-bootstrap mean: numpy poly applied to padded (real_nt, nH)
    # = (60, 32) scores; the rest of the slots are zero (mask*0 → poly(0)
    # = _v0). The IRP layout populates real_nt*nH slots; the rest are
    # _v0-valued junk after ps_exp+squarings (poly evaluated at 0 elementwise).
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

    # ---- Stage B IRP (single ct): ps_exp_init + damped squarings +
    # mean-centered bootstrap. Layout-agnostic (elementwise polynomial).
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
    e_ct = bootstrap_safe(
        engine, ctx, encoder, e_ct,
        max_abs=TARGET_MAG, slot_count=NUM_SLOTS)
    _mean_pt2 = encoder.encode_double_vector(
        ctx, np.full(NUM_SLOTS, _global_pre_mean, dtype=np.float64),
        e_ct.scale(), e_ct.chain_index())
    e_ct = phantom.add_plain(ctx, e_ct, _mean_pt2)

    # ---- Stage B mask: keep slot[h*1024 + tok] for h<nH, tok<real_nt with
    # value safety_scale; zero elsewhere (especially slots [real_nt, nt_pad)
    # within each head's first-nt_pad block — required for finalize_softmax_
    # irp_t's cyclic-replica precondition at num_tokens=nt_pad).
    mask_slots = _attn._qkt_irp_head_mask_slots(
        NUM_SLOTS, H, D, t_k, real_nt, value=safety_scale)
    e_nominal = e_ct.scale()
    mask_pt = encoder.encode_double_vector(
        ctx, mask_slots, SCALE, e_ct.chain_index())
    e_ct = phantom.multiply_plain(ctx, e_ct, mask_pt)
    e_ct = phantom.rescale_to_next(ctx, e_ct)
    e_ct.set_scale(e_nominal)

    # ---- Stage C IRP: single finalize_softmax_irp_t call. Cyclic-replica
    # at -nt_pad (= -64 for real_nt=60). All rotations in provisioned set.
    weights_ct = _attn.finalize_softmax_irp_t(
        ctx, encoder, relin_key, galois_key,
        e_ct, num_tokens=nt_pad, iters=ITERS)

    # ---- (Former Bootstrap-3 removed.) finalize_softmax_irp_t outputs
    # weights_ct at user_level 7; the only downstream consumers are
    # tree-distribute (3 levels for nt_pad=64 / 8 chunks), score_times_v_irp_
    # multi (2 levels), and IRP-Wo (lazy-leveled via mod_switch to user_level
    # 13, independent of the input level). So the real chain budget is
    # 7 + 3 + 2 = 12 at the score_v output, well under max_user_level 15
    # (3 levels of headroom). Wo's lazy mod_switch only ever drops deeper
    # (guarded `if chain_index < target`), so entering distribute at level 7
    # is safe. The earlier audit's "12 levels" referred to softmax_correct
    # inside finalize_softmax_irp_t, which runs UPSTREAM of this point — not a
    # downstream constraint. Dropping this bootstrap removes its ~170ms and its
    # injected noise.

    # ---- Tree-distribute global weights → n_chunks_pow2 per-chunk cts.
    # Inverse of tree-agg: at each level, split one ct into "lower" and
    # "upper" via mask+rotate. Only positive power-of-2 rotations needed
    # (all in provisioned set). For 8 chunks: 3 levels (8 -> 4 -> 2 -> 1
    # reversed = 1 -> 2 -> 4 -> 8). Each level uses one shared mask plaintext
    # at this level's "lower-half" pattern.
    weights_blocks = [weights_ct]
    # Process levels from coarse to fine: at level L (W = nt_pad >> L), we
    # have 2^L blocks each holding W consecutive tokens. To split one block
    # of W tokens into two blocks of W/2 tokens:
    #   lower = mask_low(block)              # keeps slots [h*1024+0 .. +W/2-1]
    #   upper = mask_high(block) rotate +W/2 # shifts slots [h*1024+W/2 ..] -> [h*1024+0 ..]
    # Equivalently: lower = block * lo_mask; upper = (rotate(block, +W/2)) * lo_mask.
    # Use a single shared lo_mask per level.
    _W = nt_pad
    while len(weights_blocks) < n_chunks_pow2:
        _half = _W // 2
        # lo_mask: 1.0 at slot[h*1024+0..half-1] for h<nH; zero elsewhere.
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
            # Upper half: rotate left by +_half (source slot h*1024+half lands
            # at h*1024+0), then mask.
            _up_rot = phantom.rotate(ctx, _wb, int(_half), galois_key)
            _up = phantom.multiply_plain(ctx, _up_rot, _lo_mask_pt)
            _up = phantom.rescale_to_next(ctx, _up)
            _up.set_scale(_nom)
            _new_blocks.append(_lo)
            _new_blocks.append(_up)
        weights_blocks = _new_blocks
        _W = _half

    # Truncate weights_blocks to actual n_chunks_k (drop the pad-to-pow2
    # trailing blocks; they'll be ignored in the sum_v anyway since V cache
    # only has n_chunks_k chunks).
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
    # output_mask is applied AFTER the ct·ct multiply+rescale in
    # score_times_v_irp, so encode at weights_chain + 1.
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

    # Decrypt the pre-Wo score·V output for the per-stage diagnostic. The
    # IRP layout is slot[(h*d_head + j)*t] = attn[h, j] (stride-t).
    _av_irp = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, attn_h)),
                       dtype=np.float64)
    fhe_attn_o = np.zeros((nH, H), dtype=np.float64)
    for h in range(nH):
        for j in range(H):
            fhe_attn_o[h, j] = _av_irp[(h * H + j) * t_k]

    # ---- IRP-Wo (Cachemir §4.1) — COMPLEX OUTPUT-FOLDED (K=512→256 SCPs).
    # encode_irp_diagonals_folded_host folds Wo.T's output columns into the
    # imag part (d×d → d×(d/2) tall rect, alpha=2), so the folded matvec runs
    # the TALL-rect machinery at d_out_fold = D/2 and consumes its input in the
    # TALL layout for (d_in=D, d_out=D/2). Path B: the score·V output (attn_h)
    # is already decrypted above (_av_irp, for the diagnostic), so the Wo INPUT
    # bridge is free — re-encrypt the natural length-D attn vector in the
    # folded tall-rect layout. The matvec emits a complex ct split by
    # extract_real_imag_pair; the output is ALREADY SK-bridged (decrypt +
    # re-encrypt to replicated-block for residual + rms2), so the recombine is
    # absorbed into that existing bridge at zero extra cost.
    from blocks import irp as _irp
    _BABY_STEPS_IRP_WO = 16  # M=16, G=32 for d=4096 K=512 (~sqrt(K))
    t_wo = NUM_SLOTS // D                          # = 8 for D=4096
    _D_OUT_FOLD_WO = D // 2  # 2048
    # Folded Wo SCPs (pass Wo.T to match irp_matvec's y = x @ M convention;
    # gives o = Wo @ attn). The fold halves the SCP count + disk cache.
    _wo_irp = _irp_cache.wo_plaintexts_cached(
        ctx, encoder,
        np.ascontiguousarray(np.asarray(Wo, dtype=np.float64).T),
        N=NUM_SLOTS, d=D, scale=SCALE, baby_steps=_BABY_STEPS_IRP_WO,
        layer_idx=layer_idx)
    # INPUT bridge: rebuild the natural length-D attn vector from the already-
    # decrypted _av_irp (IRP stride-t_wo: slot[i*t_wo] = attn[i]), then encrypt
    # in the FOLDED tall-rect layout for (d_in=D, d_out=D/2). Lazy-level: drop
    # to user_level 12 so the rotation-heavy folded matvec runs cheap. Headroom:
    # the folded matvec + conj-split consumes ~3 levels, landing the pre-decrypt
    # cts at user_level ~15 (≤ max) before the output bridge discards the chain.
    _attn_natural = _av_irp[::t_wo][:D]
    _wo_input_ci = engine.user_level_chain_index(12)
    av_folded = _irp.encrypt_irp_input_rect(
        ctx, encoder, sk, np.asarray(_attn_natural, dtype=np.float64),
        N=NUM_SLOTS, d_in=D, d_out=_D_OUT_FOLD_WO, scale=SCALE,
        chain_index=_wo_input_ci)
    # Tall masks at the FOLDED dim d_out=D/2: input_mask (square at d=D/2, at
    # input chain) + sub_mask (rect at chain+1).
    _input_mask_wo = _irp.encode_irp_mask(
        ctx, encoder, NUM_SLOTS, _D_OUT_FOLD_WO, SCALE, av_folded.chain_index())
    _sub_mask_wo = _irp.encode_irp_mask_rect(
        ctx, encoder, NUM_SLOTS, D, _D_OUT_FOLD_WO, SCALE,
        av_folded.chain_index() + 1)
    _o_complex = _irp.irp_matvec_folded_host(
        ctx, encoder, galois_key, av_folded, _wo_irp,
        N=NUM_SLOTS, d=D, baby_steps=_BABY_STEPS_IRP_WO,
        sub_mask_pt=_sub_mask_wo, input_mask_pt=_input_mask_wo)
    # Snap SCALE^2 → SCALE before the conj-split (pipeline convention).
    _o_complex = phantom.rescale_to_next(ctx, _o_complex)
    _o_complex.set_scale(SCALE)
    _o_re, _o_im = _irp.extract_real_imag_pair(
        ctx, encoder, galois_key, _o_complex, NUM_SLOTS, SCALE)
    # OUTPUT SK BRIDGE (already required): decrypt both halves, recombine to
    # natural length-D o in numpy (TALL fold decode: real=o[:D/2], imag=o[D/2:]),
    # then re-encrypt to replicated-block slot[k*D+j]=o[j] for residual + rms2.
    _t_fold_wo = NUM_SLOTS // _D_OUT_FOLD_WO   # 16
    _o_dec_re = np.asarray(
        encoder.decode_double_vector(ctx, sk.decrypt(ctx, _o_re)),
        dtype=np.float64)
    _o_dec_im = np.asarray(
        encoder.decode_double_vector(ctx, sk.decrypt(ctx, _o_im)),
        dtype=np.float64)
    _o_lo = _o_dec_re[::_t_fold_wo][:_D_OUT_FOLD_WO]
    _o_hi = _o_dec_im[::_t_fold_wo][:_D_OUT_FOLD_WO]
    _o_vec = np.concatenate([_o_lo, _o_hi])   # natural order, length D
    _o_slots_rep = np.zeros(NUM_SLOTS, dtype=np.float64)
    for _k in range(NUM_SLOTS // D):
        _o_slots_rep[_k * D:(_k + 1) * D] = _o_vec
    o_ct = sk.encrypt_symmetric(
        ctx, encoder.encode_double_vector(
            ctx, _o_slots_rep, SCALE, attn_h.chain_index()))

    _ov = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, o_ct)),
                   dtype=np.float64)
    fhe_out = _ov[:D_MODEL].copy()

    # ---- Oracle: softmax weights -> dense_score_v -> Wo, on the IDENTICAL
    # teacher-forced Q/K/V (the trusted Stage-1 spec). ----
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
    # Wo @ flattened attn (attn_flat[h*H+j] == oracle_attn_o[h,j]).
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


_PROBE_DECRYPT_STAGES = os.environ.get("PROBE_DECRYPT_STAGES") == "1"
_PROBE_DUMP_DIR = os.environ.get("PROBE_DUMP_DIR", "/tmp/probe_stage_dump")
_PROBE_DUMP_LAYER = [None]  # set per-layer by run_classifier_fhe when verbose


def _probe(tag, ctx, encoder, sk, ct):
    v = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                 dtype=np.float64)
    print(f"    [probe] {tag:30s} chain={ct.chain_index():2d} "
          f"max|.|={np.abs(v).max():.4e} mean|.|={np.abs(v).mean():.4e}")
    # DIAGNOSTIC ONLY (opt-in via PROBE_DECRYPT_STAGES=1). Dumps the full
    # decrypted slot vector to disk so an offline harness can compute the
    # rel-RMS vs plain-math per stage. When the flag is unset this block is
    # not entered: byte-identical to the original.
    if _PROBE_DECRYPT_STAGES and _PROBE_DUMP_LAYER[0] is not None:
        os.makedirs(_PROBE_DUMP_DIR, exist_ok=True)
        safe = tag.replace("/", "_").replace(" ", "_").replace("[", "").replace("]", "")
        np.save(f"{_PROBE_DUMP_DIR}/L{_PROBE_DUMP_LAYER[0]}__{safe}.npy", v)


# Module-level full-weight cache + lock used as a defensive fallback by
# _LazyLayerWeights. As of the "preload all 9 weights" fix, the parallel
# sweep pre-loads every key per layer up front, so the lazy fallback path
# is normally never taken — `w[k]` always hits the subset dict and returns
# without touching this lock. We keep the machinery in place purely as a
# safety net: if a future caller passes a partial `preloaded_weights` dict
# (only some keys), the lazy path will still satisfy the missing accesses
# correctly (at the cost of the global-lock serialization that motivated
# the preload-all fix). Cost when unused: zero.
_LAZY_FULL_WEIGHT_CACHE = {}
_LAZY_FULL_WEIGHT_LOCK = threading.Lock()


class _LazyLayerWeights:
    """Dict-like wrapper around a pre-loaded per-layer weight subset.

    DEFENSIVE FALLBACK ONLY. The parallel sweep now pre-loads all 9
    weights per layer, so the subset is the full set and every
    `__getitem__` returns from `self._subset` without entering `_full()`.
    If a caller ever passes a partial subset, missed keys trigger a
    one-shot `load_layer_weights(layer_idx)` cached in `full_cache`
    under `lock` — note this serializes ALL worker threads on the lock,
    which is why we now avoid it via the preload-all default.

    Returns values directly from the subset when present; on a miss
    (Wo/Wgate/Wup/Wdown for the per-example hot path), falls back to a
    one-shot full `load_layer_weights(layer_idx)` cached in `full_cache`
    under `lock`. Subsequent misses for the same layer hit the cache; a
    miss in one worker thread populates the cache for all workers.

    Supports `__getitem__`, `__contains__`, `__iter__`, and `get()` so it
    is a drop-in stand-in for the subset dict at every call site that
    treats it as read-only.
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
        # Treat the full on-disk weight set as the source of truth so callers
        # using `if k in w` (e.g. encode_layer_irps' subset check) see all
        # 9 keys without forcing a disk load.
        return key in ("Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown", "g1", "g2")

    def __iter__(self):
        return iter(("Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown", "g1", "g2"))

    def get(self, key, default=None):
        if key in self._subset:
            return self._subset[key]
        return self._full().get(key, default)


def run_classifier_fhe(num_tokens, query_position, pytorch_ref, pytorch_pre_norm,
                         cos_all_full, sin_all_full, label="prompt",
                         debug_layer=None, max_layer=None, min_layer=None,
                         rp_indep_cache=None, engine=None,
                         shared_wq_cache=None, shared_wq_cache_events=None,
                         shared_wq_cache_lock=None,
                         preloaded_weights=None,
                         precomputed_calib=None,
                         rp_indep_disk_root=None):
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

    # Per-layer weight accessor. The parallel sweep pre-loads the per-example
    # subset (Wq/Wk/Wv/g1/g2) ONCE on the main thread and passes the dict
    # here via preloaded_weights; serial / legacy callers leave it None and
    # fall back to the original per-call np.load. py-spy showed concurrent
    # workers stuck on disk I/O + glibc malloc contention inside
    # load_layer_weights (~128 MB allocations × 9 keys × 4 threads); the
    # pre-load eliminates that contention entirely.
    #
    # The preloaded subset is missing the R_P-independent keys
    # (Wo/Wgate/Wup/Wdown). Most consumers (encode_layer_irps, attention/MLP
    # blocks) serve those from the shared rp_indep_cache and never touch
    # `w[...]` directly, but a few call sites (e.g. compute_layer_calib_n in
    # this module) do read them. We wrap the subset in _LazyLayerWeights so
    # any missed key triggers a one-shot full load_layer_weights() on first
    # access. The full-weight cache is module-level so the cost is paid ONCE
    # per layer across the entire sweep (all examples, all workers).
    def _get_layer_w(layer_idx):
        if preloaded_weights is not None:
            return _LazyLayerWeights(
                layer_idx, preloaded_weights[layer_idx],
                _LAZY_FULL_WEIGHT_CACHE, _LAZY_FULL_WEIGHT_LOCK)
        return load_layer_weights(layer_idx)

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
    fresh_ci = engine.freshest_chain_index()
    # Galois-key target chains were computed against FRESHEST_CHAIN=16; fail
    # fast if the actual freshest chain has moved (e.g. evalmod_r=4).
    assert fresh_ci == 16, (
        f"engine.freshest_chain_index()={fresh_ci} != 16 — "
        "build_user_steps_mrpc targets need to be updated.")

    # (IRP layer-independent masks removed — the dense token-major path
    # builds its plaintext masks per-shard inside the dense kernels.)

    # ---- Bootstrap placement (same as llama3.py)
    NSL_MAX = NUM_SCALE_LEVELS - 1
    T_BOOT_MS = 182.0
    OUTPUT_LEVEL_AFTER_IRP = USER_LEVEL_IRP_ATTN + 2
    # output_level here is the planner's REMAINING-BUDGET level (NSL_MAX = fresh,
    # 0 = exhausted), the inverse of the runtime's consumed-level view.
    # attention output is SK-bridged (decrypted + re-encrypted at fresh_ci, see
    # the attn_out re-encrypt below), so it emerges FRESH — NOT at the IRP output
    # level. Modeling it as fresh stops the planner scheduling a redundant
    # bootstrap_before rms2 (which was firing on a user_level-0 ct). NOTE: revert
    # to OUTPUT_LEVEL_AFTER_IRP if the SK output bridge is ever removed (bridgeless
    # autonomous path), where attention really does emerge deep.
    placement_table = [
        ("rms1",      7,  29.4, True,  None,    True),
        ("attention", 0, 521.0, True,  NSL_MAX, False),
        ("residual1", 0,   1.0, True,  None,    False),
        ("rms2",      7,  27.4, True,  None,    True),
        ("mlp",       0, 624.1, True,  OUTPUT_LEVEL_AFTER_IRP, False),
        ("residual2", 0,   1.0, True,  None,    False),
    ]
    layers_for_dag = build_layers_from_table(placement_table)
    plan = find_optimal_placement(layers_for_dag, NSL_MAX, T_BOOT_MS)
    boot_before = {plan.layers[s.layer_idx].name: s.bootstrap_before for s in plan.steps}

    # ---- Per-layer FHE forward
    NUM_DECODERS = 32
    # Opt-in autonomous residual stream (mirrors reverted commit 625ea9c
    # "llama3: bootstrap y_ct forward as next-layer x_ct"). When
    # AUTONOMOUS_FHE=1, layer >= 1 feeds the PREVIOUS layer's output
    # ciphertext y_ct forward (bootstrapped to the same fresh chain /
    # scale / stride-T_MODEL layout that encrypt_layer_inputs_multi
    # produces for x_ct) instead of re-encrypting pytorch_ref[layer_idx].
    # K/V/c_per_head still come from the clean numpy ref (x_btd) exactly
    # as in 625ea9c, so the per-layer decrypt/log now measures the TRUE
    # drift of the carried encrypted state vs pytorch_ref[layer_idx+1].
    # Default (unset) leaves the guided path byte-for-byte unchanged.
    _autonomous_fhe = os.environ.get("AUTONOMOUS_FHE") == "1"
    _y_ct_carry = None  # bootstrapped y_ct from previous layer -> next x_ct
    if _autonomous_fhe:
        print("  [AUTONOMOUS_FHE] carrying y_ct forward (K/V from ref)")
    # Per-layer weight-subset preload. The dense token-major pipeline
    # rebuilds Q/K/V/Wo/Wgate/Wup/Wdown directly from the numpy weights
    # each layer (no pre-encoded IRP plaintexts), so all that is needed
    # up front is the small per-example-hot subset (Wq/Wk/Wv + g1/g2,
    # used by encrypt_layer_inputs_multi & rmsnorm). The big R_P-indep
    # matrices (Wo/Wgate/Wup/Wdown) are loaded per-layer via
    # load_layer_weights_subset inside the dense attention / MLP blocks.
    #
    # The shared_wq_cache* / rp_indep_cache / rp_indep_disk_root params
    # are retained for call-signature compatibility with the sweep
    # drivers but are inert now that the IRP machinery is gone (the
    # pre-encoded-IRP cache they fed no longer exists).
    t_wq_encode0 = time.perf_counter()
    layer_weights = {}  # layer_idx -> {Wq,Wk,Wv,g1,g2} subset
    for _li in range(NUM_DECODERS):
        if min_layer is not None and _li < min_layer:
            continue
        if max_layer is not None and _li > max_layer:
            break
        _w = _get_layer_w(_li)
        layer_weights[_li] = {
            k: _w[k] for k in ("Wq", "Wk", "Wv", "g1", "g2") if k in _w
        }
        del _w
    t_wq_encode = time.perf_counter() - t_wq_encode0
    print(f"[weight-subset preload: {t_wq_encode:.1f}s]")

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
        _t_wait = 0.0  # PERF_BREAKDOWN: default 0 when queue.get is not hit
        # ---- 1-layer-ahead prefetch (latency-only; never a correctness dep) --
        # The 32-layer warm run is I/O-bound: each layer cold-reads ~1.6 GB of
        # IRP SCP blobs + ~1.4 GB of fp64 weights from a 57 GB cache (> RAM →
        # page-cache thrash), adding ~4.6 s/layer of disk read on top of ~5.9 s
        # compute. Background-LOAD the NEXT layer's 5 IRP blobs + numpy weights
        # (encoder-FREE: mmap + scp_from_bytes + np.load only — the encoder's
        # expand happens later, per-matvec, ON THIS MAIN THREAD) so the next
        # layer's read overlaps this layer's GPU compute. The wrappers
        # (*_plaintexts_cached / get_layer_weights) await the pending future or
        # fall back to a synchronous load on a RAM miss, so a skipped/failed
        # prefetch only costs latency, never correctness. Cold MISS (no blob)
        # is left to the synchronous encode path — never threaded.
        _next_li = layer_idx + 1
        _do_prefetch_next = (_next_li < NUM_DECODERS and
                             (max_layer is None or _next_li <= max_layer))
        # Prefetch constants mirror the actual call sites exactly:
        #   Wq/Wo: d=D_TOTAL(=4096), baby_steps=16 (fhe_attention_dense_full)
        #   MLP  : d_in=D_MODEL, d_out=16384, baby_steps=_BABY_STEPS_IRP_MLP_RECT=16
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
            # NOTE: no numpy prefetch for (Wgate, Wup, Wdown) — those are now
            # LAZY (passed as 0-arg loaders to the IRP cache wrappers). On a
            # warm run the SCP blobs hit and the loaders never fire, so a
            # background numpy prefetch would only re-introduce the ~1.4 GB/
            # layer of WASTED I/O this change exists to eliminate. The
            # IRP-SCP prefetch above (prefetch_layer) is what overlaps the
            # real reads with compute.
        # Trim RAM entries for layers older than the current one (the LRU bound
        # already caps memory; this keeps only current + next layer resident).
        _irp_cache.evict_layers_before(
            layer_idx, P_local=P_local, d=D_TOTAL,
            mlp_d_in=_PF_MLP_DIN, mlp_d_out=_PF_MLP_DOUT,
            scale=SCALE, baby_steps_attn=_PF_BABY_ATTN,
            baby_steps_mlp=_PF_BABY_MLP)
        verbose = (debug_layer is not None and layer_idx == debug_layer)
        # DIAGNOSTIC ONLY: tag stage dumps with the current layer so the
        # offline rel-RMS harness can pick the right file. No-op when
        # PROBE_DECRYPT_STAGES is unset (verbose is also False here normally).
        _PROBE_DUMP_LAYER[0] = layer_idx if (verbose and _PROBE_DECRYPT_STAGES) else None
        if _PROBE_DECRYPT_STAGES:
            try:
                import blocks.attention as _att_mod
                _att_mod._PROBE_DUMP_LAYER[0] = _PROBE_DUMP_LAYER[0]
            except Exception:
                pass
        x_btd = pytorch_ref[layer_idx]  # (NUM_TOKENS, D_MODEL) — input to layer L

        # Per-layer weights (the {Wq,Wk,Wv,g1,g2} subset preloaded above).
        # The dense path reloads the big R_P-indep matrices
        # (Wo/Wgate/Wup/Wdown) per-layer inside the attention / MLP blocks.
        w = layer_weights[layer_idx]

        # Per-layer rmsnorm + bootstrap_safe calibration (num_tokens-aware).
        # When `precomputed_calib` is supplied (parallel sweep), skip the
        # per-example shadow forward pass entirely — calib was precomputed
        # once at startup using a representative example, which also lets
        # the worker preload drop the big Wo/Wgate/Wup/Wdown matrices
        # (~45 GB across 32 layers) since the per-example hot path only
        # touches Wq/Wk/Wv/g1/g2 directly.
        if precomputed_calib is not None:
            z1_l, z2_l, max_abs_calib = precomputed_calib[layer_idx]
        else:
            # Disk-cached calibration: load_layer_weights() pulls the full
            # ~1.4 GB weight dict (Wo/Wgate/Wup/Wdown) purely to run the numpy
            # shadow forward, then discards it. The (z1,z2,max_abs) output is
            # deterministic in (x_btd, layer, num_tokens, query_position), so
            # cache it — warm runs skip both the weight load and the forward.
            def _compute_calib():
                # layer_weights[layer_idx] is the subset; reload the full dict
                # per-layer and drop after calib so the heap doesn't grow to
                # 60 GB across the 32-layer loop.
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
        # Tightened silu_domain margin 1.2 → 1.05 to narrow the Chebyshev fit
        # range. Narrower domain → smaller polynomial coefficients → less
        # CKKS noise amplification through Clenshaw recurrence. Safe because
        # max_abs_calib already includes BOOT_CALIB_MARGIN=1.5× over actual
        # numpy-predicted max; the additional 1.05× covers FHE noise on gate.
        silu_domain = (-silu_max * 1.05, silu_max * 1.05)
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
        # Force Clenshaw at high-magnitude layers (L=30/31 in LLaMA-3.1-8B,
        # silu_max ≥ 6 there). With NSQ=6, accumulated softmax-path drift
        # can push some gate slot past the ±1.2·silu_max cushion at those
        # layers — deg-20 monomial extrapolation past domain is catastrophic
        # (silu(1.2D=9.5)≈-60 vs true 9.5), causing the L=30 cascade blowup
        # to 150k+ observed in NSQ=6 sweeps. Clenshaw with deg-32 Chebyshev
        # basis bounds intermediates by max|t_k| and stays bounded outside
        # the fit domain. Cost: +2 bootstraps + ~840ms on the dispatched
        # layers (only 2 of 32) → negligible at the layer-sweep scale.
        if silu_deg <= 20 and _best_err <= _SILU_POLY_ERR_BUDGET and silu_max <= 6.0:
            silu_t_coeffs = None  # gates fhe_mlp_irp_bootstrap to eval_polynomial
            silu_D = None
            _silu_path = f"poly{silu_deg}"
        else:
            silu_D = silu_domain[1]
            # Clenshaw deg=32 is sufficient; deg=48 gives no measurable
            # accuracy gain (verified idx=6: identical max|err| at L=30/31,
            # confirming silu fit error ≪ accumulated CKKS noise floor).
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

        # Encrypt inputs (multi-ct K, V). K/V/c_per_head always derive
        # from the clean numpy x_btd (= pytorch_ref[layer_idx]); only the
        # encrypted query residual x_ct is carried in autonomous mode.
        _t_prep_end = time.perf_counter()  # PERF_BREAKDOWN: end of host-prep phase
        t_encrypt0 = time.perf_counter()
        x_ct, k_cts, v_cts, c_per_head, _ = encrypt_layer_inputs_multi(
            ctx, encoder, sk, fresh_ci, x_btd, w, R_P,
            num_tokens, cos_all, sin_all, P_local)
        # DIAGNOSTIC ONLY (PROBE_DECRYPT_STAGES=1): dump the EXACT live
        # c_per_head + safety_scale the FHE pipeline uses, so the offline
        # harness compares decrypted intermediates against the SAME centering
        # the ciphertext actually got. No-op when the flag is unset.
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
            # Mirror reverted 625ea9c: discard the freshly-encrypted x_ct
            # (from pytorch_ref[layer_idx]) and feed the previous layer's
            # output ciphertext forward instead. 625ea9c called
            # engine.bootstrap_inplace(y_ct) directly to refresh it to a
            # fresh level; the SK-free equivalent here is bootstrap_safe
            # with the same x_in calibration the pipeline already trusts
            # for x_ct (line below mirrors the boot_before["rms1"] site).
            # This restores y_ct to the freshest chain index / SCALE that
            # encrypt_layer_inputs_multi produces; the stride-T_MODEL slot
            # layout is already preserved by the decoder pipeline (same
            # layout that y_full[::T_MODEL][:D_MODEL] decodes).
            assert _y_ct_carry is not None, "autonomous: missing y_ct carry"
            x_ct = bootstrap_safe(engine, ctx, encoder, _y_ct_carry,
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
        # Dense token-major attention (THE compute path — IRP deleted).
        # QK^T + softmax + score·V + BSGS Wo, all FHE.
        # fhe_attention_dense_full runs the complete dense block
        # internally; its _f_out is the layer's attention output feeding
        # residual1. Teacher-forced Q/K/V mirror encrypt_layer_inputs_multi
        # exactly (same numpy x_btd / weights / rope). Wo / (Wv via w) are
        # R_P-independent; Wo is NOT in the per-example hot subset and the
        # shared cache may hold only the 5-key subset (w["Wo"] -> KeyError),
        # so load Wo directly off disk via the subset loader (one
        # (4096,4096) fp64 array). Layout bridge: dense BSGS Wo output is
        # replicated-block period-D_TOTAL; the residual stream / x_ct is
        # stride-T_MODEL (slot[i*T_MODEL]==x[i], cf.
        # encrypt_layer_inputs_multi). Re-encode into stride-T_MODEL and
        # re-encrypt at fresh_ci — the SAME teacher-forcing layout bridge
        # the per-layer pipeline already applies to its inputs each layer.
        # The `Layer {L}` rel-RMS vs pytorch_ref[L+1] below is the
        # validation metric.
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
            q_max_abs=max_abs_calib.get("q") if max_abs_calib else None)
        _f_out = _fres["fhe_out"]            # (D_MODEL,)
        _attn_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
        _attn_slots[::T_MODEL][:D_MODEL] = _f_out
        attn_out = sk.encrypt_symmetric(
            ctx, encoder.encode_double_vector(
                ctx, _attn_slots, SCALE, fresh_ci))
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
        # ---- IRP-rect MLP (Cachemir §4.1 rect host).
        # K_sq = d²/N = 512 per square sub-IRP, K_total = K_sq*α = 2048
        # SCPs per matmul — 8× fewer than dense BSGS's 16384. NO bridges:
        # stride-T_MODEL == IRP layout at d=D_MODEL (both slot[i*8]=x[i]),
        # so x_mid_norm enters and mlp_out exits in the same shape the
        # surrounding rmsnorm+residual already use.
        from blocks import irp as _irp_mlp
        _BABY_STEPS_IRP_MLP_RECT = 16  # M=16, G=32 for K_sq=512 (~sqrt)
        _D_PAD_OUT_MLP = 16384  # D_HIDDEN=14336 padded to pow-2 multiple of D_MODEL (α=4)

        # Wgate/Wup/Wdown are R_P-independent and may not be in w's hot
        # subset (same situation as Wo). They are LAZY: the ~1.4 GB fp64 load +
        # zero-pad below is wrapped in 0-arg callables and passed straight to
        # the IRP cache wrappers. On a WARM run the SCP blobs hit on disk, the
        # wrappers return the cached SCPs WITHOUT calling the loader, and the
        # big numpy load/pad never fires — dropping ~1.4 GB/layer of wasted I/O
        # so the prefetch can finally hide the IRP-only reads. On a COLD MISS
        # the three loaders share ONE memoized subset load (so a triple-miss
        # reads the weights off disk once, not 3×).
        #
        # Convention: irp_matvec_rect_host computes y = x @ M.
        # For Wgate/Wup we want gate = x @ Wgate.T → M = Wgate.T padded to
        # (D_MODEL, D_PAD_OUT) with trailing columns zero. Wdown.T padded to
        # (D_PAD_OUT, D_MODEL) with trailing rows zero.
        _mlp_ws_cache = {}

        def _load_mlp_subset():
            # COLD-MISS ONLY: load (Wgate, Wup, Wdown) once, memoized so the
            # three lazy loaders below share a single ~1.4 GB disk read.
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

        # Encode IRP-rect SCPs. Wgate/Wup are WIDE (d_in=D_MODEL < d_out=D_PAD_OUT).
        # They are complex output-FOLDED (K/2 SCPs each): the folded matvec
        # emits a complex ct (real=out[:d/2], imag=out[d/2:]) at the folded dim
        # d_out_fold = D_PAD_OUT/2, split + interleave-recombined back to a real
        # interleaved-layout ct below. Wdown is TALL (d_in=D_PAD_OUT >
        # d_out=D_MODEL), UNFOLDED, with rows permuted to absorb the gate/up
        # interleave layout (interleave_output_order, applied in the cache
        # wrapper) → mlp_out comes out NATURAL order, no un-permute.
        _D_OUT_FOLD_MLP = _D_PAD_OUT_MLP // 2  # 8192
        _gate_irp = _irp_cache.gate_plaintexts_cached(
            ctx, encoder, _load_gate_padded, N=NUM_SLOTS,
            d_in=D_MODEL, d_out=_D_PAD_OUT_MLP, scale=SCALE,
            baby_steps=_BABY_STEPS_IRP_MLP_RECT, layer_idx=layer_idx)
        _up_irp = _irp_cache.up_plaintexts_cached(
            ctx, encoder, _load_up_padded, N=NUM_SLOTS,
            d_in=D_MODEL, d_out=_D_PAD_OUT_MLP, scale=SCALE,
            baby_steps=_BABY_STEPS_IRP_MLP_RECT, layer_idx=layer_idx)
        _down_irp = _irp_cache.down_plaintexts_cached(
            ctx, encoder, _load_down_padded, N=NUM_SLOTS,
            d_in=_D_PAD_OUT_MLP, d_out=D_MODEL, scale=SCALE,
            baby_steps=_BABY_STEPS_IRP_MLP_RECT, layer_idx=layer_idx,
            gate_up_d_in=D_MODEL, gate_up_d_out=_D_PAD_OUT_MLP)

        # Lazy-level: drop the IRP-Wgate/Wup input to a deep chain so these two
        # rotation-heavy wide rect matvecs (D_MODEL×D_HIDDEN, K=2048 SCPs each —
        # the biggest matvecs in the model) run at few RNS limbs (cheap).
        # Headroom audit: gate AND up are refreshed together by ONE
        # merge_bootstrap right before silu (both land at user_level ~1, fresh),
        # so neither output has a deep downstream constraint. The matvec
        # consumes 2 levels (sub_mask + rescale), so targeting input user_level
        # 11 leaves both outputs at user_level 13 going into the merge. After
        # the merge silu consumes ~7 levels (bootstrap pre-scale + Clenshaw) and
        # up stays fresh, so the post-silu multiply aligns to silu (~ul 7→8)
        # leaving h_ct at user_level ~9 — shallow enough that Wdown's lazy-level
        # mod_switch handles the rest without a dedicated h-boot. Both matvecs
        # consume the same input, so mod_switch a single shared deep copy.
        _mlp_target_ci = engine.user_level_chain_index(11)
        _x_mid_norm_deep = x_mid_norm
        if _x_mid_norm_deep.chain_index() < _mlp_target_ci:
            _x_mid_norm_deep = phantom.mod_switch_to(ctx, x_mid_norm, _mlp_target_ci)

        # Per K-2 test (blocks/kv_cache_test.py:115-127): wide path needs
        # only sub_mask_pt; tall path needs BOTH sub_mask_pt (at chain+1)
        # AND input_mask_pt (at chain, encoded at d=d_out as square mask).
        # FOLDED wide rect path: the fold halves d_out (16384 → 8192) but the
        # path stays wide (d_in=D_MODEL=4096 < d_out_fold=8192). ct_in goes
        # directly into the matvec with mask_pt=sub_mask; the mask op fires
        # INSIDE the matvec at ct_in's chain (the lazy-leveled
        # _x_mid_norm_deep). The fold input layout == the unfolded wide input
        # layout (slot[i*t]=x[i], t=N/D_MODEL), so _x_mid_norm_deep enters
        # unchanged.
        _sub_mask_gate_up = _irp_mlp.encode_irp_mask_rect(
            ctx, encoder, N=NUM_SLOTS, d_in=D_MODEL, d_out=_D_OUT_FOLD_MLP,
            scale=SCALE, chain_index=_x_mid_norm_deep.chain_index())

        def _folded_interleaved_matvec(_irp_pts):
            """Folded wide matvec → complex ct → split → interleave-recombine
            → real ct in interleaved (stride t_fold/2) layout. The fold +
            extract adds ~+1 level vs the unfolded matvec; interleave_recombine
            is 0 levels (1 rot + 1 add)."""
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
        #    At high-magnitude layers (L30/31, silu_max>6) the harness
        #    sets silu_t_coeffs/silu_D and the BOUNDED Chebyshev-basis
        #    Clenshaw path is taken. The deg<=20 monomial silu_coeffs
        #    catastrophically extrapolates past the ±1.2·silu_max fit
        #    domain (silu(9.5)≈-60 vs ≈9.5), so this path MUST honor
        #    silu_t_coeffs/silu_D too. --
        # Merge gate+up refresh into ONE bootstrap (same cost as one). This
        # makes _up_ct fresh too, so the post-silu multiply lands h_ct shallow
        # enough to skip the h-boot. max_abs must bound BOTH gate and up; both
        # bounds are taken raw (no double BOOT_CALIB_MARGIN) to match the
        # previous gate-boot which used max_abs=silu_max (raw gate).
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

        # h-boot eliminated: merge_bootstrap above made _up_ct fresh, so the
        # post-silu multiply (silu @ ~7 * up @ fresh) aligns to ~7, leaving
        # _h_ct shallow enough that Wdown's lazy-level mod_switch handles the
        # rest without a dedicated bootstrap.

        # -- Wdown (tall rect IRP matvec — COMPLEX OUTPUT-FOLDED, K=2048→1024
        #    SCPs, the biggest remaining tall fold) --
        # The folded matvec runs the rect machinery at d_out_fold = D_MODEL/2,
        # so it consumes its input in the TALL layout for (D_PAD_OUT, D_MODEL/2)
        # — the row-permutation absorbing the gate/up interleave is computed at
        # this folded dim in the cache wrapper (down_plaintexts_cached). Masks
        # are therefore encoded at the FOLDED dims (d_in=D_PAD_OUT,
        # d_out=D_MODEL/2): tall path needs sub_mask_pt (at chain+1) AND
        # input_mask_pt (square mask at d=d_out_fold).
        # Lazy-level: drop the IRP-Wdown input to a deep chain so the
        # rotation-heavy tall rect matvec runs at few RNS limbs (cheap).
        # Headroom audit: the folded matvec consumes ~2 levels (input_mask +
        # sub_mask), a rescale (+1), then extract_real_imag_pair (+1 for the
        # conj-split multiply) — one more level than the old unfolded path. The
        # split halves are then DECRYPTED by the output SK bridge (the result
        # is decrypted for residual2 anyway), recombined to natural order in
        # numpy, and RE-ENCRYPTED fresh, so downstream is unaffected. Targeting
        # input user_level 10 leaves the pre-decrypt cts at user_level ~12-13
        # (< max 15). mod_switch_to drops limbs cleanly (no added noise).
        _D_OUT_FOLD_DOWN = D_MODEL // 2  # 2048
        _wdown_target_ci = engine.user_level_chain_index(11)
        if _h_ct.chain_index() < _wdown_target_ci:
            _h_ct = phantom.mod_switch_to(ctx, _h_ct, _wdown_target_ci)
        _input_mask_down = _irp_mlp.encode_irp_mask(
            ctx, encoder, N=NUM_SLOTS, d=_D_OUT_FOLD_DOWN, scale=SCALE,
            chain_index=_h_ct.chain_index())
        _sub_mask_down = _irp_mlp.encode_irp_mask_rect(
            ctx, encoder, N=NUM_SLOTS, d_in=_D_PAD_OUT_MLP, d_out=_D_OUT_FOLD_DOWN,
            scale=SCALE, chain_index=_h_ct.chain_index() + 1)
        _mlp_complex = _irp_mlp.irp_matvec_rect_folded_host(
            ctx, encoder, galois_key, _h_ct, _down_irp,
            N=NUM_SLOTS, d_in=_D_PAD_OUT_MLP, d_out=D_MODEL,
            baby_steps=_BABY_STEPS_IRP_MLP_RECT,
            sub_mask_pt=_sub_mask_down,
            input_mask_pt=_input_mask_down)
        # Snap SCALE^2 → SCALE before the conj-split (pipeline convention).
        _mlp_complex = phantom.rescale_to_next(ctx, _mlp_complex)
        _mlp_complex.set_scale(SCALE)
        _mlp_re, _mlp_im = _irp_mlp.extract_real_imag_pair(
            ctx, encoder, galois_key, _mlp_complex, NUM_SLOTS, SCALE)
        # -- OUTPUT SK BRIDGE: decrypt both halves, recombine to natural order
        #    in numpy (free + legitimate — mlp_out is decrypted anyway), then
        #    re-encrypt fresh in the stride-T_MODEL layout residual2's other
        #    operand (x_mid_ct) uses. This near-lossless bridge also refreshes
        #    the chain. The TALL fold decode (real=out[:d/2], imag=out[d/2:])
        #    mirrors /tmp/irp_rect_fold_fhe_test.py's _folded reader. --
        _dec_re = np.asarray(encoder.decode_double_vector(
            ctx, sk.decrypt(ctx, _mlp_re)), dtype=np.float64)
        _dec_im = np.asarray(encoder.decode_double_vector(
            ctx, sk.decrypt(ctx, _mlp_im)), dtype=np.float64)
        _y_lo = _irp_mlp.decode_irp_output_rect(
            _dec_re, N=NUM_SLOTS, d_in=_D_PAD_OUT_MLP, d_out=_D_OUT_FOLD_DOWN)
        _y_hi = _irp_mlp.decode_irp_output_rect(
            _dec_im, N=NUM_SLOTS, d_in=_D_PAD_OUT_MLP, d_out=_D_OUT_FOLD_DOWN)
        _mlp_natural = np.concatenate([_y_lo, _y_hi])  # natural order, length D_MODEL
        _mlp_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
        _mlp_slots[::T_MODEL][:D_MODEL] = _mlp_natural
        mlp_out = sk.encrypt_symmetric(
            ctx, encoder.encode_double_vector(ctx, _mlp_slots, SCALE, fresh_ci))
        if verbose: _probe("post-mlp", ctx, encoder, sk, mlp_out)
        # residual2 (both operands now natural stride-T_MODEL at fresh_ci)
        y_ct = residual(ctx, x_mid_ct, mlp_out)
        if verbose: _probe("post-residual2 y_ct", ctx, encoder, sk, y_ct)
        layer_ms = (time.perf_counter() - t_layer_start) * 1000
        layer_times.append(layer_ms)

        # Decrypt for accuracy check (vs pre-norm reference for L=31, vs pytorch_ref[L+1] for others)
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
            # Carry the post-residual2 output ciphertext into the next
            # layer's x_ct (bootstrapped at the encrypt site above). The
            # decrypt above is logging-only here (true drift measurement),
            # exactly as in reverted 625ea9c.
            _y_ct_carry = y_ct

    # ---- LM head (host-side)
    yes_logit, no_logit = yes_no_logits_np(y_p_fhe, final_norm_g, lm_head_yesno,
                                              eps=meta["rms_norm_eps"])
    print(f"\n--- LM head: FHE yes_logit={yes_logit:.4f}  no_logit={no_logit:.4f} ---")
    print(f"--- Total layer time: {sum(layer_times)/1000:.1f}s "
          f"(avg {sum(layer_times)/len(layer_times):.0f}ms/layer) ---")
    return yes_logit, no_logit


def capture_pytorch_ref_with_model(model, tok, token_ids):
    """Run a forward pass on a pre-loaded model and return the same data as
    capture_pytorch_ref. The caller is responsible for loading and deleting
    the model; this function does NOT load or free it.

    Args:
      model: pre-loaded AutoModelForCausalLM on cuda:0 (fp16, eval mode).
      tok:   unused; kept for call-site symmetry with capture_pytorch_ref.
      token_ids: list[int] token ids for the prompt.

    Returns:
      pytorch_ref:      (n_layers+1, num_tokens, D_MODEL) ndarray float64
      pytorch_pre_norm: (num_tokens, D_MODEL) ndarray float64
      yes_pt, no_pt:    float logits at the last token position
    """
    import torch
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
    return pytorch_ref, pytorch_pre_norm, yes_pt, no_pt


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
    ref, prenorm, yes_pt, no_pt = capture_pytorch_ref_with_model(model, None, token_ids)
    del model
    torch.cuda.empty_cache()
    return ref, prenorm, yes_pt, no_pt


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
