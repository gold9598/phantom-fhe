"""
LLaMA-3.1-8B layer-0 single decoder. SK-free decoder body: rmsnorm, the
residual stream, and SDPA all run in stride-T_MODEL / interleaved-replicated
layout end-to-end, matching the IRP module's native input/output convention.
The only sk-touching sites are at boundaries — client-side initial
encryption of x/K/V (`sk.encrypt_symmetric` on the input slot vectors)
and the test-harness decrypt of the final output.

Cachemir IRP plaintext-encoding swap with per-step galois target chain
indices to fit on a 32 GB GPU. The pre-IRP attention/MLP plaintext bulk
(~30 GiB, with BSGS Wq/Wo + complex BSGS Wgate/Wup/Wdown) collapses to
~3 GiB via host-stored IRP plaintexts. The per-step galois bundle uses
`CKKSEngineConfig.user_rotation_target_chain_indices` to assign each user
rotation step the smallest galois key (deepest chain target) compatible
with its actual call depth, shrinking the engine's static GPU footprint
by another several GiB.

Pipeline:
  rms1 -> bootstrap -> Wq IRP -> compute_qkt_irp -> mask*scale -> sub(C[h])
       -> bootstrap -> ps_exp + damped sq -> bootstrap -> mask
       -> finalize_softmax_irp_t -> score_times_v_irp -> Wo IRP
       -> +x_ct (residual1) -> bootstrap -> rms2 -> bootstrap
       -> Wgate IRP (wide) -> silu -> Wup IRP (wide) -> ct*ct
       -> [refresh] -> Wdown IRP (tall) -> +x_mid (residual2) -> decrypt y_ct
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from blocks.irp import (
    encode_irp_diagonals_host, irp_matvec_host,
    encode_irp_mask, irp_required_steps,
    encode_irp_diagonals_rect_host, irp_matvec_rect_host,
    encode_irp_mask_rect, irp_required_steps_rect,
    decode_irp_output_rect,
)
from blocks.attention import (
    compute_qkt_irp, score_times_v_irp, finalize_softmax_irp_t,
    qkt_irp_mask_scale_plaintext, qkt_irp_per_head_sub_plaintext,
    score_v_irp_output_mask_plaintext,
    sdpa_irp_required_steps,
)
from blocks.softmax import softmax_damping_schedule
from blocks.silu import silu, fit_silu_coeffs
from blocks.bootstrap import bootstrap_safe
from blocks.bootstrap_placement import (
    build_layers_from_table, find_optimal_placement, render_plan_table,
)
from blocks.residual import residual
from blocks.rmsnorm import (
    rmsnorm_forward, rmsnorm_forward_stride_t,
    rmsnorm_required_steps, rmsnorm_required_steps_stride_t,
    setup_rmsnorm_weights,
)


# ============================ Constants ============================
LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2
SCALE = 2.0 ** 40
SPARSE_HW = 128

D_MODEL = 4096
D_HEAD = 128
N_HEADS = 32
N_KV_HEADS = 8
N_KV_GROUPS = N_HEADS // N_KV_HEADS
D_TOTAL = N_HEADS * D_HEAD
NUM_TOKENS = 4

# Stride-t residual stream layout (IRP-native): rmsnorm and the residual
# stream operate on stride-t-packed ciphertexts (data at slots 0, t, 2t, ...
# and zeros elsewhere) so the IRP input/output sites no longer require sk
# round-trips. T_MODEL = NUM_SLOTS // D_MODEL = 32768 // 4096 = 8.
T_MODEL = (1 << (LOG_N - 1)) // D_MODEL  # NUM_SLOTS // D_MODEL = 8

# IRP setup: square IRP for Wq, Wo (d = D_TOTAL = D_MODEL = 4096 here).
BABY_STEPS_IRP_SQUARE = 16    # K = 512; M*G = 512 -> M=16, G=32

D_HIDDEN = 14336              # actual SwiGLU hidden dim
D_PAD_MLP = 16384             # padded power-of-two (alpha=4 over D_MODEL)
ALPHA_MLP = D_PAD_MLP // D_MODEL
BABY_STEPS_IRP_MLP = 16

EPSILON = 1e-5
P = 3                          # query position (last token attends to all 4)

# NUM_SQUARINGS=4 (not 5) to fit in NSL=14's 13-level budget:
# ps_exp_init=4 levels + 4 damped squarings×2 levels = 12 levels ≤ max_user_level=13.
# Softmax approximation quality: NUM_SQUARINGS=4 covers scores in [-8,0] well;
# our per-head scores after C_per_head subtraction are near zero, so this suffices.
NUM_SQUARINGS = 4
EXTRA_SCALE = 0.5
ITERS = 6
TARGET_MAG = 0.45

RMS_POLY_DEG = 4
RMS_Z_MARGIN = 0.30  # ±30% multiplicative window for per-layer z calibration

# CKKSEngine layout.
# num_scale_levels=14 → size_Q = 1+14+3+9+3 = 30, 30/6=5 chunks ✓
# max_user_level = 13; pre_boot chain = 16+13 = 29.
# Level budget per sub-stage (between bootstraps):
#   rms1:        ~5 levels  (sum_of_sq ct*ct + invsqrt deg-4 poly + gamma mul)
#   Wq IRP:      1 level    (mask * rescale)
#   SDPA A:      2 levels   (compute_qkt ct*ct+rescale, mask*scale+rescale)
#   SDPA B:      ~8 levels  (ps_exp_init deg-4 + 5 damped squarings)  → bootstrap between A and B
#   SDPA C:      ~4 levels  (finalize_softmax, score_v ct*ct+rescale, mask+replicate)  → bootstrap between B and C
#   Wo IRP:      1 level
#   rms2:        ~5 levels  → bootstrap before rms2
#   MLP gate/up: 1 level each IRP → bootstrap_safe refresh (fresh chain ~13 ul above msg)
#   silu:        ~4 levels  (deg-8 poly)  ← fits within freshened budget
#   swiglu:      1 level    (ct*ct)       → bootstrap_safe refresh before Wdown
#   Wdown IRP:   1 level
# Total per sub-stage ≤ 10; NSL=14 (13 usable) gives comfortable headroom.
NUM_SCALE_LEVELS = 14
NUM_SPECIAL_PRIMES = 6

# User-level shorthand. freshest_chain_index = 16 (fixed by bootstrap pipeline).
# Key size scales as (size_Q - user_level) primes.
# Assignment strategy:
#   - rms steps {1,2,...,2048}: MUST be fresh (rmsnorm called at level 0 after bootstrap).
#   - Everything else (sdpa-only, replicate, irp-only): assigned to pre_boot = level 13
#     so they use the smallest possible keys.  These steps fire only after a
#     decrypt+re-encrypt that can place the ciphertext at any desired chain.
USER_LEVEL_FRESH = 0
USER_LEVEL_DEEP   = NUM_SCALE_LEVELS - 1   # = 13 (pre_boot)
# IRP re-encrypt depth: user_level 10 → chain 16+10=26.
# target_chain=26: size_Ql=30-(26-1)=5, beta_k=ceil(5/6)=1 partition per key.
# 44 IRP keys × 1 partition × 37.7 MB = 1,659 MB vs 3,322 MB at user_level 8.
# Saves ~1.7 GB on steady state AND eliminates the transient allocation spike.
USER_LEVEL_IRP_ATTN = 10      # Wq, Wo IRP decrypt+re-encrypt depth
USER_LEVEL_IRP_MLP  = 10      # Wgate, Wup, Wdown IRP depth
USER_LEVEL_SILU_REFRESH = 8   # gate/up refreshed before silu (deg-8: 4 levels; 8+4=12 ≤ 13)

PROBE = "/tmp/llama_probe"
PROBE_FULL = "/tmp/llama_probe_full"


def load_layer_weights(layer_idx):
    """Load real per-layer LLaMA-3.1-8B weights from probe v2."""
    ld = f"{PROBE_FULL}/layer_{layer_idx:02d}"
    L = lambda n: np.load(f"{ld}/{n}.npy").astype(np.float64)
    return {"Wq": L("Wq"), "Wk": L("Wk"), "Wv": L("Wv"), "Wo": L("Wo"),
            "Wgate": L("Wgate"), "Wup": L("Wup"), "Wdown": L("Wdown"),
            "g1": L("g1"), "g2": L("g2")}


def encode_layer_irps(ctx, encoder, w, R_P, pack_gate_up=True):
    """Pre-encode IRP plaintexts for one layer's projection weights.
    Returns (Wq_baked, diag_wq_irp, diag_wo_irp, diag_gate_irp,
             diag_up_irp, diag_down_irp).

    If pack_gate_up=True (default): Wgate and Wup are packed into a single
    complex IRP (Wgate in real, Wup in imag). diag_gate_irp holds the
    packed plaintexts and diag_up_irp is None. The downstream
    fhe_mlp_irp_bootstrap detects None and uses one matvec + a conjugation-
    based extract to recover gate_ct and up_ct. Halves the MLP IRP
    encoding time (~4s/layer saved) and the gate+up matvec count.
    """
    Wq_baked = w["Wq"].copy()
    for h in range(N_HEADS):
        s, e = h*D_HEAD, (h+1)*D_HEAD
        Wq_baked[s:e, :] = R_P @ w["Wq"][s:e, :]
    diag_wq_irp = encode_irp_diagonals_host(
        ctx, encoder, Wq_baked.T, NUM_SLOTS, D_TOTAL, SCALE,
        baby_steps=BABY_STEPS_IRP_SQUARE)
    diag_wo_irp = encode_irp_diagonals_host(
        ctx, encoder, w["Wo"].T, NUM_SLOTS, D_TOTAL, SCALE,
        baby_steps=BABY_STEPS_IRP_SQUARE)
    Wgate_pad = np.zeros((D_MODEL, D_PAD_MLP), dtype=np.float64)
    Wgate_pad[:, :D_HIDDEN] = w["Wgate"].T
    Wup_pad = np.zeros((D_MODEL, D_PAD_MLP), dtype=np.float64)
    Wup_pad[:, :D_HIDDEN] = w["Wup"].T
    Wdown_pad = np.zeros((D_PAD_MLP, D_MODEL), dtype=np.float64)
    Wdown_pad[:D_HIDDEN, :] = w["Wdown"].T
    if pack_gate_up:
        from blocks.irp import encode_irp_diagonals_rect_pair_host
        diag_gate_irp = encode_irp_diagonals_rect_pair_host(
            ctx, encoder, Wgate_pad, Wup_pad,
            NUM_SLOTS, D_MODEL, D_PAD_MLP, SCALE,
            baby_steps=BABY_STEPS_IRP_MLP)
        diag_up_irp = None
    else:
        diag_gate_irp = encode_irp_diagonals_rect_host(
            ctx, encoder, Wgate_pad, NUM_SLOTS, D_MODEL, D_PAD_MLP, SCALE,
            baby_steps=BABY_STEPS_IRP_MLP)
        diag_up_irp = encode_irp_diagonals_rect_host(
            ctx, encoder, Wup_pad, NUM_SLOTS, D_MODEL, D_PAD_MLP, SCALE,
            baby_steps=BABY_STEPS_IRP_MLP)
    diag_down_irp = encode_irp_diagonals_rect_host(
        ctx, encoder, Wdown_pad, NUM_SLOTS, D_PAD_MLP, D_MODEL, SCALE,
        baby_steps=BABY_STEPS_IRP_MLP)
    return Wq_baked, diag_wq_irp, diag_wo_irp, diag_gate_irp, diag_up_irp, diag_down_irp


# ============================ Plaintext helpers ============================
def rmsnorm_np(x, g, eps=EPSILON):
    rms = np.sqrt((x**2).mean(-1, keepdims=True) + eps)
    return (x / rms) * g

def rotate_half_np(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)

def apply_rope_np(x_btd, cos_td, sin_td):
    return x_btd * cos_td[:, None, :] + rotate_half_np(x_btd) * sin_td[:, None, :]

def rope_matrix_np(cos_p, sin_p):
    d = cos_p.shape[0]; h = d // 2
    M = np.zeros((d, d), dtype=np.float64)
    for i in range(h):
        M[i, i]         = cos_p[i]
        M[i, h + i]     = -sin_p[i]
        M[h + i, h + i] = cos_p[h + i]
        M[h + i, i]     = sin_p[h + i]
    return M

def silu_np(x): return x * (1.0 / (1.0 + np.exp(-x)))

def rms_z_window(z):
    """Symmetric multiplicative window around z."""
    return (z * (1.0 - RMS_Z_MARGIN), z * (1.0 + RMS_Z_MARGIN))

BOOT_CALIB_MARGIN = 1.5  # safety margin over numpy-predicted max|.| at each
                          # bootstrap_safe site, to absorb FHE-side noise drift.


def compute_layer_max_abs(x_btd, w, cos_all, sin_all, P, margin=BOOT_CALIB_MARGIN):
    """Trace numpy forward and record max|.| at every bootstrap_safe site.
    Returns a dict of per-site max_abs values (margined) for this layer's
    actual input distribution. Used by fhe_attention_irp_bootstrap and
    fhe_mlp_irp_bootstrap to calibrate the in-block CKKS bootstraps."""
    g1, g2 = w["g1"], w["g2"]
    Wq, Wk, Wv, Wo = w["Wq"], w["Wk"], w["Wv"], w["Wo"]
    Wgate, Wup, Wdown = w["Wgate"], w["Wup"], w["Wdown"]

    xn = rmsnorm_np(x_btd, g1)  # post-rms1
    Q_full = (xn @ Wq.T).reshape(NUM_TOKENS, N_HEADS, D_HEAD)
    K_full = (xn @ Wk.T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD)
    V_full = (xn @ Wv.T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD)
    Q_full = apply_rope_np(Q_full, cos_all, sin_all)
    K_full = apply_rope_np(K_full, cos_all, sin_all)
    K_full = np.repeat(K_full, N_KV_GROUPS, axis=1)
    V_full = np.repeat(V_full, N_KV_GROUPS, axis=1)
    q_max = float(np.abs(Q_full[P]).max())
    scores = np.einsum('hd,thd->ht', Q_full[P], K_full) / math.sqrt(D_HEAD)
    c_per_head = scores.max(0) + 0.5
    scores_post_C = scores - c_per_head[None, :]
    scores_max = float(np.abs(scores_post_C).max())
    weights = np.exp(scores_post_C - scores_post_C.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)
    attn_p = np.einsum('ht,thd->hd', weights, V_full).reshape(N_HEADS * D_HEAD)
    o_p = attn_p @ Wo.T
    x_mid_full = x_btd.copy(); x_mid_full[P] = x_btd[P] + o_p
    x_mid_max = float(np.abs(x_mid_full[P]).max())
    x_mid_n = rmsnorm_np(x_mid_full, g2)
    rms2_out_max = float(np.abs(x_mid_n[P]).max())
    gate_pre = x_mid_n[P] @ Wgate.T
    gate_max = float(np.abs(gate_pre).max())
    gate_silu = silu_np(gate_pre)
    up = x_mid_n[P] @ Wup.T
    up_max = float(np.abs(up).max())
    h = gate_silu * up
    h_max = float(np.abs(h).max())
    return {
        "x_in":     float(np.abs(x_btd[P]).max()) * margin,  # pre-rms1 (boot before rms1)
        "rms1_out": float(np.abs(xn[P]).max()) * margin,      # post-rms1 (boot before attn)
        "x_mid":    x_mid_max * margin,                        # post-residual1 (boot before rms2)
        "rms2_out": rms2_out_max * margin,                    # post-rms2 (boot before mlp)
        "q":        q_max * margin,                            # post-Wq IRP
        "scores":   scores_max * margin,                       # post-attn_A scores - C
        "gate":     gate_max * margin,                         # post-Wgate IRP (pre-silu)
        "up":       up_max * margin,                           # post-Wup IRP
        "h":        h_max * margin,                            # post-silu*up (swiglu output)
    }


def compute_layer_z(x_btd, g1, g2, Wq, Wk, Wv, Wo, cos_all, sin_all, P):
    """Return (z_rms1, z_rms2) — the mean(x²)+EPSILON values that the rms1
    and rms2 polynomials must cover for this layer's query position P."""
    z1 = float((x_btd[P]**2).mean() + EPSILON)
    xn = rmsnorm_np(x_btd, g1)
    Q_full = (xn @ Wq.T).reshape(NUM_TOKENS, N_HEADS, D_HEAD)
    K_full = (xn @ Wk.T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD)
    V_full = (xn @ Wv.T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD)
    Q_full = apply_rope_np(Q_full, cos_all, sin_all)
    K_full = apply_rope_np(K_full, cos_all, sin_all)
    K_full = np.repeat(K_full, N_KV_GROUPS, axis=1)
    V_full = np.repeat(V_full, N_KV_GROUPS, axis=1)
    Q_p = Q_full[P]
    scores_p = np.einsum('hd,thd->ht', Q_p, K_full) / math.sqrt(D_HEAD)
    w_p = np.exp(scores_p - scores_p.max(-1, keepdims=True))
    w_p = w_p / w_p.sum(-1, keepdims=True)
    attn_p = np.einsum('ht,thd->hd', w_p, V_full).reshape(N_HEADS * D_HEAD)
    o_p = attn_p @ Wo.T
    x_mid_P = x_btd[P] + o_p
    z2 = float((x_mid_P**2).mean() + EPSILON)
    return z1, z2

def forward_decoder_np(x_btd, g1, g2, Wq, Wk, Wv, Wo, Wgate, Wup, Wdown,
                        cos_all, sin_all, P):
    """Exact numpy decoder forward. Returns y_btd [NUM_TOKENS, D_MODEL].
    Only y_btd[P] is the full decoded output; other rows are passthroughs."""
    xn = rmsnorm_np(x_btd, g1)
    Q_full = (xn @ Wq.T).reshape(NUM_TOKENS, N_HEADS, D_HEAD)
    K_full = (xn @ Wk.T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD)
    V_full = (xn @ Wv.T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD)
    Q_full = apply_rope_np(Q_full, cos_all, sin_all)
    K_full = apply_rope_np(K_full, cos_all, sin_all)
    K_full = np.repeat(K_full, N_KV_GROUPS, axis=1)  # [NUM_TOKENS, N_HEADS, D_HEAD]
    V_full = np.repeat(V_full, N_KV_GROUPS, axis=1)
    Q_p = Q_full[P]  # [N_HEADS, D_HEAD]
    scores_p = np.einsum('hd,thd->ht', Q_p, K_full) / math.sqrt(D_HEAD)
    w_p = np.exp(scores_p - scores_p.max(-1, keepdims=True))
    w_p = w_p / w_p.sum(-1, keepdims=True)  # [N_HEADS, NUM_TOKENS]
    attn_p = np.einsum('ht,thd->hd', w_p, V_full).reshape(N_HEADS * D_HEAD)
    o_p = attn_p @ Wo.T  # [D_MODEL]
    x_mid = x_btd.copy(); x_mid[P] = x_btd[P] + o_p
    x_mid_n = rmsnorm_np(x_mid, g2)
    gate = silu_np(x_mid_n @ Wgate.T)
    up = x_mid_n @ Wup.T
    h = gate * up
    out = h @ Wdown.T  # [NUM_TOKENS, D_MODEL]
    y = x_mid.copy(); y[P] = x_mid[P] + out[P]
    return y

def encrypt_layer_inputs(ctx, encoder, sk, fresh_ci, x_btd, g1, Wq_baked, Wk, Wv,
                          cos_all, sin_all, P):
    """Build and encrypt x_ct/k_ct/v_ct and compute c_per_head for a decoder layer."""
    xn = rmsnorm_np(x_btd, g1)
    K = (xn @ Wk.T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD)
    V = (xn @ Wv.T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD)
    K = apply_rope_np(K, cos_all, sin_all)
    K_full = np.repeat(K, N_KV_GROUPS, axis=1).reshape(NUM_TOKENS, D_TOTAL)
    V_full = np.repeat(V, N_KV_GROUPS, axis=1).reshape(NUM_TOKENS, D_TOTAL)
    Q_np = (xn[P] @ Wq_baked.T).reshape(N_HEADS, D_HEAD)
    K_full_h = K_full.reshape(NUM_TOKENS, N_HEADS, D_HEAD)
    scores_np = (Q_np[None, :, :] * K_full_h).sum(-1) / math.sqrt(D_HEAD)
    c_per_head = scores_np.max(0) + 0.5
    x_slots = np.zeros(NUM_SLOTS); k_slots = np.zeros(NUM_SLOTS); v_slots = np.zeros(NUM_SLOTS)
    x_slots[::T_MODEL][:D_MODEL] = x_btd[P]
    K_full_h2 = K_full.reshape(NUM_TOKENS, N_HEADS, D_HEAD)
    V_full_h2 = V_full.reshape(NUM_TOKENS, N_HEADS, D_HEAD)
    for h_idx in range(N_HEADS):
        for j in range(D_HEAD):
            base = (h_idx * D_HEAD + j) * T_MODEL
            for tok in range(NUM_TOKENS):
                k_slots[base + tok] = K_full_h2[tok, h_idx, j]
                v_slots[base + tok] = V_full_h2[tok, h_idx, j]
    x_ct = sk.encrypt_symmetric(ctx, encoder.encode_double_vector(ctx, x_slots.tolist(), SCALE, fresh_ci))
    k_ct = sk.encrypt_symmetric(ctx, encoder.encode_double_vector(ctx, k_slots.tolist(), SCALE, fresh_ci))
    v_ct = sk.encrypt_symmetric(ctx, encoder.encode_double_vector(ctx, v_slots.tolist(), SCALE, fresh_ci))
    return x_ct, k_ct, v_ct, c_per_head

def _probe(tag, ctx, encoder, sk, ct):
    v = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                 dtype=np.float64)
    print(f"  [probe] {tag:30s} chain={ct.chain_index():2d} "
          f"max|.|={np.abs(v).max():.4e} mean|.|={np.abs(v).mean():.4e}")


def _probe_diff(tag, ctx, encoder, sk, ct, np_ref,
                stride, count, orderless=False):
    """Decrypt ct, extract meaningful slots, compare to np_ref.

    stride/count: meaningful slots are at positions [0, stride, 2*stride, ...,
    (count-1)*stride].  If orderless=True, compare sorted magnitudes (for
    permuted MLP intermediates).  np_ref must be a 1-D array of length `count`
    in the same order as the slot extraction (or any order when orderless=True).
    """
    raw = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                   dtype=np.float64)
    fhe_vals = raw[::stride][:count]
    np_vals  = np.asarray(np_ref, dtype=np.float64).ravel()[:count]
    if orderless:
        fhe_vals = np.sort(np.abs(fhe_vals))
        np_vals  = np.sort(np.abs(np_vals))
    diff = fhe_vals - np_vals
    fhe_max  = float(np.abs(fhe_vals).max()) if len(fhe_vals) else 0.0
    np_max   = float(np.abs(np_vals).max())  if len(np_vals)  else 0.0
    diff_max = float(np.abs(diff).max())     if len(diff)     else 0.0
    diff_rms = float(np.linalg.norm(diff) / math.sqrt(max(len(diff), 1)))
    print(f"[probe-L31] {tag:30s}  fhe_max={fhe_max:.2e}  np_max={np_max:.2e}"
          f"  diff_max={diff_max:.2e}  diff_rms={diff_rms:.2e}")


def run_decoder_fhe(engine, ctx, encoder, sk, relin_key, galois_key,
                    x_ct, k_ct, v_ct, c_per_head,
                    diag_wq_irp, diag_wo_irp, mask_attn_pt,
                    diag_gate_irp, diag_up_irp, diag_down_irp,
                    sub_mask_mlp_wide_pt, sub_mask_mlp_tall_pt, input_mask_mlp_pt,
                    rms1_w, rms2_w, rms1_p, rms2_p,
                    boot_before, label="layer", probe=False,
                    max_abs_calib=None, silu_coeffs=None, probe_np=None):
    """Run one decoder layer in FHE. Returns (y_full_np, total_ms, stage_times).
    y_full_np is the decrypted output in stride-T_MODEL layout.
    probe_np: dict of numpy reference intermediates (populated by caller for L31)."""
    stage_times = {}
    stage_times.setdefault("bootstrap", 0.0)
    if probe:
        _probe("input x_ct", ctx, encoder, sk, x_ct)
        _probe("input k_ct", ctx, encoder, sk, k_ct)
        _probe("input v_ct", ctx, encoder, sk, v_ct)
        if probe_np is not None:
            _probe_diff("1.input_x", ctx, encoder, sk, x_ct,
                        probe_np["x_btd_P"], stride=T_MODEL, count=D_MODEL)

    # Outer bootstrap calibrations between named layers. Default = layer-0
    # tuning; overridden per-layer by max_abs_calib (Stage 3a').
    _BOOT_MAX_ABS = {
        "rms1":      (max_abs_calib or {}).get("x_in",     1.0),
        "attention": (max_abs_calib or {}).get("rms1_out", 1.0),
        "rms2":      (max_abs_calib or {}).get("x_mid",    1.0),
        "mlp":       (max_abs_calib or {}).get("rms2_out", 1.0),
    }

    def _maybe_boot(name, ct):
        if not boot_before.get(name, False):
            return ct
        t0 = time.perf_counter()
        ct = bootstrap_safe(engine, ctx, encoder, ct,
                            max_abs=_BOOT_MAX_ABS[name], slot_count=NUM_SLOTS)
        stage_times["bootstrap"] += (time.perf_counter() - t0) * 1000
        print(f"  [plan] bootstrap before {name}: chain={ct.chain_index()}")
        return ct

    t_total0 = time.perf_counter()

    # rms1
    x_ct = _maybe_boot("rms1", x_ct)
    t0 = time.perf_counter()
    x_norm = rmsnorm_forward_stride_t(ctx, encoder, relin_key, galois_key,
                                       x_ct, rms1_w, rms1_p, t=T_MODEL)
    stage_times["rms1"] = (time.perf_counter() - t0) * 1000
    print(f"  rms1 done. chain={x_norm.chain_index()}")
    if probe: _probe("post-rms1 x_norm", ctx, encoder, sk, x_norm)
    if probe and probe_np is not None:
        _probe_diff("2.post_rms1", ctx, encoder, sk, x_norm,
                    probe_np["xn_P"], stride=T_MODEL, count=D_MODEL)

    # attention
    x_norm = _maybe_boot("attention", x_norm)
    t0 = time.perf_counter()
    attn_out = fhe_attention_irp_bootstrap(
        engine, ctx, encoder, relin_key, galois_key,
        x_norm, diag_wq_irp, diag_wo_irp, mask_attn_pt,
        k_ct, v_ct, c_per_head, stage_times=stage_times,
        max_abs_calib=max_abs_calib, probe_np=probe_np if probe else None,
        sk=sk if probe else None)
    stage_times["attention"] = (time.perf_counter() - t0) * 1000
    print(f"  attention done. chain={attn_out.chain_index()}")
    if probe: _probe("post-attention attn_out", ctx, encoder, sk, attn_out)
    if probe and probe_np is not None:
        _probe_diff("13.post_Wo", ctx, encoder, sk, attn_out,
                    probe_np["o_P"], stride=T_MODEL, count=D_MODEL)

    # residual1
    x_mid_ct = residual(ctx, x_ct, attn_out)
    print(f"  residual1 done. chain={x_mid_ct.chain_index()}")
    if probe: _probe("post-residual1 x_mid", ctx, encoder, sk, x_mid_ct)
    if probe and probe_np is not None:
        _probe_diff("14.post_residual1", ctx, encoder, sk, x_mid_ct,
                    probe_np["x_mid_P"], stride=T_MODEL, count=D_MODEL)

    # rms2
    x_mid_ct = _maybe_boot("rms2", x_mid_ct)
    if probe and probe_np is not None:
        _probe_diff("15.post_rms2_boot", ctx, encoder, sk, x_mid_ct,
                    probe_np["x_mid_P"], stride=T_MODEL, count=D_MODEL)
    t0 = time.perf_counter()
    x_mid_norm = rmsnorm_forward_stride_t(ctx, encoder, relin_key, galois_key,
                                           x_mid_ct, rms2_w, rms2_p, t=T_MODEL)
    stage_times["rms2"] = (time.perf_counter() - t0) * 1000
    print(f"  rms2 done. chain={x_mid_norm.chain_index()}")
    if probe: _probe("post-rms2 x_mid_norm", ctx, encoder, sk, x_mid_norm)
    if probe and probe_np is not None:
        _probe_diff("16.post_rms2", ctx, encoder, sk, x_mid_norm,
                    probe_np["x_mid_n_P"], stride=T_MODEL, count=D_MODEL)

    # mlp
    x_mid_norm = _maybe_boot("mlp", x_mid_norm)
    t0 = time.perf_counter()
    mlp_out = fhe_mlp_irp_bootstrap(
        engine, ctx, encoder, relin_key, galois_key,
        x_mid_norm,
        diag_gate_irp, diag_up_irp, diag_down_irp,
        sub_mask_mlp_wide_pt, sub_mask_mlp_tall_pt, input_mask_mlp_pt,
        stage_times=stage_times, max_abs_calib=max_abs_calib,
        silu_coeffs=silu_coeffs,
        probe_np=probe_np if probe else None,
        sk=sk if probe else None)
    stage_times["mlp"] = (time.perf_counter() - t0) * 1000
    print(f"  mlp done. chain={mlp_out.chain_index()}")
    if probe: _probe("post-mlp mlp_out", ctx, encoder, sk, mlp_out)

    y_ct = residual(ctx, x_mid_ct, mlp_out)
    if probe and probe_np is not None:
        _probe_diff("25.post_residual2", ctx, encoder, sk, y_ct,
                    probe_np["y_P"], stride=T_MODEL, count=D_MODEL)
    total_ms = (time.perf_counter() - t_total0) * 1000

    print(f"  decrypt y_ct at chain={y_ct.chain_index()} scale={y_ct.scale():.3e}")
    y_full = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, y_ct)),
                       dtype=np.float64)

    main_keys = ["rms1", "attention", "rms2", "mlp", "bootstrap"]
    print(f"Per-stage runtime (ms) [{label}]:")
    for k in main_keys:
        if k in stage_times:
            print(f"  {k:30s} {stage_times[k]:8.1f}")
    sub_keys = sorted(k for k in stage_times if k not in main_keys)
    for k in sub_keys:
        print(f"    {k:30s} {stage_times[k]:8.1f}")
    print(f"  {'total':30s} {total_ms:8.1f}")

    return y_full, total_ms, stage_times, y_ct


# ============================ Attention forward (IRP + bootstrap) ============================
def fhe_attention_irp_bootstrap(engine, ctx, encoder, relin_key,
                                 galois_key,
                                 x_norm,
                                 diag_wq_irp, diag_wo_irp,
                                 mask_attn_pt,
                                 k_ct, v_ct, c_per_head,
                                 stage_times=None, max_abs_calib=None,
                                 probe_np=None, sk=None):
    """Stage-2 IRP-native attention. Q stays in stride-T_MODEL packing through
    the entire SDPA. K/V cache packed interleaved across t tokens within a
    single ciphertext (Cachemir §5.1). No sk-touching relayouts in the
    decoder body.
    probe_np/sk: when not None, emit [probe-L31] diff lines at each sub-stage.
    """
    # Layer-0 defaults; overridden per-layer via max_abs_calib (Stage 3a').
    _calib = {"q": 2.5, "scores": 45.10}
    if max_abs_calib is not None:
        _calib.update({k: max_abs_calib[k] for k in ("q", "scores") if k in max_abs_calib})

    def _t(): return time.perf_counter()
    def _rec(name, t0):
        if stage_times is None: return
        stage_times.setdefault(name, 0.0)
        stage_times[name] += (time.perf_counter() - t0) * 1000.0

    # ---- x_norm is already stride-t (rmsnorm_forward_stride_t output);
    # mod-switch to USER_LEVEL_IRP_ATTN so plaintext masks line up. ----
    t0 = _t()
    irp_attn_ci = engine.user_level_chain_index(USER_LEVEL_IRP_ATTN)
    if x_norm.chain_index() < irp_attn_ci:
        x_irp = phantom.mod_switch_to(ctx, x_norm, irp_attn_ci)
    else:
        x_irp = x_norm
    _rec("layout_shift", t0)

    # ---- Wq via IRP. q_ct stays in stride-T_MODEL after IRP — this is the
    # exact layout compute_qkt_irp expects. The IRP-internal mask multiplies
    # at scale SCALE^2; one extra rescale brings the scale back to SCALE so
    # the downstream ct·ct in compute_qkt_irp sees matching scales. ----
    t0 = _t()
    q_ct = irp_matvec_host(ctx, encoder, galois_key, x_irp, diag_wq_irp,
                      NUM_SLOTS, D_TOTAL, baby_steps=BABY_STEPS_IRP_SQUARE,
                      mask_pt=mask_attn_pt)
    q_ct = phantom.rescale_to_next(ctx, q_ct)
    q_ct.set_scale(SCALE)
    _rec("wq_irp", t0)
    if probe_np is not None and sk is not None:
        _probe_diff("3.post_Wq", ctx, encoder, sk, q_ct,
                    probe_np["Q_P_flat"], stride=T_MODEL, count=D_TOTAL)

    # ---- bootstrap to refresh chain to 16 so Stage A has the full SDPA
    # budget. Without this, q_ct at chain ~28 would push compute_qkt_irp +
    # mask*scale past chain 29 (= NSL_MAX) and overflow. ----
    t0 = _t()
    q_ct = bootstrap_safe(engine, ctx, encoder, q_ct,
                          max_abs=_calib["q"], slot_count=NUM_SLOTS)
    _rec("bootstrap", t0)
    if probe_np is not None and sk is not None:
        _probe_diff("4.post_q_boot", ctx, encoder, sk, q_ct,
                    probe_np["Q_P_flat"], stride=T_MODEL, count=D_TOTAL)

    # ---- compute_qkt_irp + mask*scale + sub(C[h]). ----
    # Output: scores at slot[h*D_HEAD*T_MODEL + tok] = m[tok, h], with
    # mid-head junk that the mask*scale step zeros out.
    t0 = _t()
    phantom.mod_switch_to_inplace(ctx, k_ct, q_ct.chain_index())
    scores_ct = compute_qkt_irp(ctx, encoder, relin_key, galois_key,
                                 q_ct, k_ct, D_HEAD, D_TOTAL, T_MODEL)
    nominal = scores_ct.scale()
    inv_sqrt_d = 1.0 / math.sqrt(float(D_HEAD))
    ms_pt = qkt_irp_mask_scale_plaintext(
        ctx, encoder, D_HEAD, D_TOTAL, NUM_TOKENS, T_MODEL,
        inv_sqrt_d, scores_ct.chain_index(), SCALE)
    scores_ct = phantom.multiply_plain(ctx, scores_ct, ms_pt)
    scores_ct = phantom.rescale_to_next(ctx, scores_ct)
    scores_ct.set_scale(nominal)
    sub_pt = qkt_irp_per_head_sub_plaintext(
        ctx, encoder, D_HEAD, D_TOTAL, NUM_TOKENS, T_MODEL,
        c_per_head, scores_ct.chain_index(), scores_ct.scale())
    scores_ct = phantom.sub_plain(ctx, scores_ct, sub_pt)
    _rec("attn_A", t0)
    if probe_np is not None and sk is not None:
        # scores layout: slot[h*D_HEAD*T_MODEL + tok] = scores[tok, h]
        # Build flat reference: for each head h, for each token tok:
        #   ref[h*D_HEAD*T_MODEL + tok] = (scores_np[tok,h] - c_per_head[h])
        scores_ref_slots = np.zeros(N_HEADS * D_HEAD * T_MODEL, dtype=np.float64)
        scores_np_ref = probe_np["scores_minus_c"]  # [NUM_TOKENS, N_HEADS]
        for h_i in range(N_HEADS):
            for tok in range(NUM_TOKENS):
                scores_ref_slots[h_i * D_HEAD * T_MODEL + tok] = scores_np_ref[tok, h_i]
        # Extract FHE slots at positions 0..N_HEADS*D_HEAD*T_MODEL-1
        raw_s = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, scores_ct)),
                         dtype=np.float64)
        fhe_s = raw_s[:N_HEADS * D_HEAD * T_MODEL]
        # Only compare the meaningful tok slots (tok < NUM_TOKENS within each head block)
        meaningful_mask = np.zeros(N_HEADS * D_HEAD * T_MODEL, dtype=bool)
        for h_i in range(N_HEADS):
            for tok in range(NUM_TOKENS):
                meaningful_mask[h_i * D_HEAD * T_MODEL + tok] = True
        fhe_s_m = fhe_s[meaningful_mask]
        ref_s_m = scores_ref_slots[meaningful_mask]
        diff_s = fhe_s_m - ref_s_m
        print(f"[probe-L31] {'5.post_scores':30s}  fhe_max={np.abs(fhe_s_m).max():.2e}"
              f"  np_max={np.abs(ref_s_m).max():.2e}"
              f"  diff_max={np.abs(diff_s).max():.2e}"
              f"  diff_rms={np.linalg.norm(diff_s)/math.sqrt(len(diff_s)):.2e}")

    # ---- bootstrap before damped squarings. ----
    t0 = _t()
    scores_ct = bootstrap_safe(engine, ctx, encoder, scores_ct,
                               max_abs=_calib["scores"], slot_count=NUM_SLOTS)
    _rec("bootstrap", t0)
    if probe_np is not None and sk is not None:
        raw_s2 = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, scores_ct)),
                          dtype=np.float64)
        fhe_s2_m = raw_s2[:N_HEADS * D_HEAD * T_MODEL][meaningful_mask]
        diff_s2 = fhe_s2_m - ref_s_m
        print(f"[probe-L31] {'6.post_scores_boot':30s}  fhe_max={np.abs(fhe_s2_m).max():.2e}"
              f"  np_max={np.abs(ref_s_m).max():.2e}"
              f"  diff_max={np.abs(diff_s2).max():.2e}"
              f"  diff_rms={np.linalg.norm(diff_s2)/math.sqrt(len(diff_s2)):.2e}")

    # ---- ps_exp_init + damped squarings. ----
    t0 = _t()
    damps = softmax_damping_schedule(NUM_SQUARINGS, NUM_TOKENS, EXTRA_SCALE, TARGET_MAG)
    e_ct = phantom.ps_exp_init(
        ctx, encoder, relin_key, scores_ct,
        NUM_TOKENS, NUM_SQUARINGS, EXTRA_SCALE)
    phantom.square_iterations_damped_inplace(ctx, encoder, relin_key, e_ct, damps)
    _rec("attn_B", t0)
    if probe_np is not None and sk is not None:
        # After ps_exp_init + damped squarings the meaningful slots hold
        # approx: (extra_scale * NUM_TOKENS^(-1/2^k))^(2^k) * exp(score-c)
        # = (0.5 * 4^(-1/16))^16 * exp(score-c)
        # Compute the exact scalar factor:
        _scale_factor = (EXTRA_SCALE * (NUM_TOKENS ** (-1.0 / (2 ** NUM_SQUARINGS)))) ** (2 ** NUM_SQUARINGS)
        exp_ref = _scale_factor * np.exp(ref_s_m)  # [N_HEADS*NUM_TOKENS]
        raw_e = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, e_ct)),
                         dtype=np.float64)
        fhe_e_m = raw_e[:N_HEADS * D_HEAD * T_MODEL][meaningful_mask]
        diff_e = fhe_e_m - exp_ref
        print(f"[probe-L31] {'7.post_ps_exp':30s}  fhe_max={np.abs(fhe_e_m).max():.2e}"
              f"  np_max={np.abs(exp_ref).max():.2e}"
              f"  diff_max={np.abs(diff_e).max():.2e}"
              f"  diff_rms={np.linalg.norm(diff_e)/math.sqrt(len(diff_e)):.2e}")

    # ---- bootstrap before finalize_softmax. ----
    # The IRP layout has very different fill-rate: only N_HEADS*NUM_TOKENS=128
    # of NUM_SLOTS=32768 slots carry meaningful data after the upcoming
    # pre-finalize mask, so the global slot mean is dominated by the polynomial
    # constant evaluated at zero (poly(0) ~ 0.449 with the deg-4 Chebyshev fit
    # used by ps_exp_init+damped squarings). Mean-subtract before bootstrap to
    # keep |centered| <= TARGET_MAG and avoid the bootstrap_safe scale-down
    # path (which is rejected at max_user_level).
    t0 = _t()
    _PRE_FINSMX_MEAN = 0.4487
    mean_pt_pre = encoder.encode_double_vector(
        ctx, [_PRE_FINSMX_MEAN] * NUM_SLOTS, e_ct.scale(), e_ct.chain_index())
    e_ct = phantom.sub_plain(ctx, e_ct, mean_pt_pre)
    if probe_np is not None and sk is not None:
        raw_e2 = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, e_ct)),
                          dtype=np.float64)
        fhe_e2_m = raw_e2[:N_HEADS * D_HEAD * T_MODEL][meaningful_mask]
        exp_ref_sub = exp_ref - _PRE_FINSMX_MEAN
        diff_e2 = fhe_e2_m - exp_ref_sub
        print(f"[probe-L31] {'8.post_mean_sub':30s}  fhe_max={np.abs(fhe_e2_m).max():.2e}"
              f"  np_max={np.abs(exp_ref_sub).max():.2e}"
              f"  diff_max={np.abs(diff_e2).max():.2e}"
              f"  diff_rms={np.linalg.norm(diff_e2)/math.sqrt(len(diff_e2)):.2e}")
    e_ct = bootstrap_safe(engine, ctx, encoder, e_ct,
                          max_abs=TARGET_MAG, slot_count=NUM_SLOTS)
    if probe_np is not None and sk is not None:
        raw_e3 = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, e_ct)),
                          dtype=np.float64)
        fhe_e3_m = raw_e3[:N_HEADS * D_HEAD * T_MODEL][meaningful_mask]
        diff_e3 = fhe_e3_m - exp_ref_sub
        print(f"[probe-L31] {'9.post_finsmx_boot':30s}  fhe_max={np.abs(fhe_e3_m).max():.2e}"
              f"  np_max={np.abs(exp_ref_sub).max():.2e}"
              f"  diff_max={np.abs(diff_e3).max():.2e}"
              f"  diff_rms={np.linalg.norm(diff_e3)/math.sqrt(len(diff_e3)):.2e}")
    mean_pt_post = encoder.encode_double_vector(
        ctx, [_PRE_FINSMX_MEAN] * NUM_SLOTS, e_ct.scale(), e_ct.chain_index())
    e_ct = phantom.add_plain(ctx, e_ct, mean_pt_post)
    _rec("bootstrap", t0)

    # ---- pre-finalize_softmax mask + finalize + score*V (all IRP-native). ----
    t0 = _t()
    # Zero non-meaningful slots before finalize_softmax. Mask shape: keep
    # slot[h*D_HEAD*T_MODEL + tok] for h<N_HEADS, tok<NUM_TOKENS.
    e_nominal = e_ct.scale()
    mask_pt = qkt_irp_mask_scale_plaintext(
        ctx, encoder, D_HEAD, D_TOTAL, NUM_TOKENS, T_MODEL,
        1.0, e_ct.chain_index(), SCALE)
    e_ct = phantom.multiply_plain(ctx, e_ct, mask_pt)
    e_ct = phantom.rescale_to_next(ctx, e_ct)
    e_ct.set_scale(e_nominal)
    if probe_np is not None and sk is not None:
        raw_e4 = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, e_ct)),
                          dtype=np.float64)
        fhe_e4_m = raw_e4[:N_HEADS * D_HEAD * T_MODEL][meaningful_mask]
        diff_e4 = fhe_e4_m - exp_ref
        print(f"[probe-L31] {'10.post_mask':30s}  fhe_max={np.abs(fhe_e4_m).max():.2e}"
              f"  np_max={np.abs(exp_ref).max():.2e}"
              f"  diff_max={np.abs(diff_e4).max():.2e}"
              f"  diff_rms={np.linalg.norm(diff_e4)/math.sqrt(len(diff_e4)):.2e}")

    # IRP-native softmax: cyclic-broadcast trick (rotate -NUM_TOKENS) makes
    # sum_reduce_stride(stride=1, count=NUM_TOKENS) broadcast the full per-head
    # sum to every valid token slot.
    weights_ct = finalize_softmax_irp_t(
        ctx, encoder, relin_key, galois_key, e_ct, NUM_TOKENS, ITERS)
    if probe_np is not None and sk is not None:
        # After finalize_softmax weights[tok, h] = softmax(scores[tok, h])
        # softmax_ref[tok, h] for each head h; slot[h*D_HEAD*T_MODEL + tok]
        softmax_ref = probe_np["softmax_weights"]  # [NUM_TOKENS, N_HEADS]
        raw_w = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, weights_ct)),
                         dtype=np.float64)
        fhe_w_m = raw_w[:N_HEADS * D_HEAD * T_MODEL][meaningful_mask]
        ref_w_m = np.zeros(N_HEADS * NUM_TOKENS, dtype=np.float64)
        idx = 0
        for h_i in range(N_HEADS):
            for tok in range(NUM_TOKENS):
                ref_w_m[idx] = softmax_ref[tok, h_i]
                idx += 1
        diff_w = fhe_w_m - ref_w_m
        print(f"[probe-L31] {'11.post_finalize_smx':30s}  fhe_max={np.abs(fhe_w_m).max():.2e}"
              f"  np_max={np.abs(ref_w_m).max():.2e}"
              f"  diff_max={np.abs(diff_w).max():.2e}"
              f"  diff_rms={np.linalg.norm(diff_w)/math.sqrt(len(diff_w)):.2e}")

    # IRP-native score×V: weights_ct in cyclic-broadcast layout × interleaved
    # V_cache → stride-T_MODEL output ready for Wo IRP.
    weights_ci = weights_ct.chain_index()
    phantom.mod_switch_to_inplace(ctx, v_ct, weights_ci)
    # Score×V output mask consumes one chain level (multiply_plain + rescale).
    # Output mask lives at the chain after the ct·ct + reduce in score_v_irp,
    # which is weights_ci + 1 (one rescale inside score_times_v_irp).
    sv_mask = score_v_irp_output_mask_plaintext(
        ctx, encoder, D_HEAD, D_TOTAL, T_MODEL,
        weights_ci + 1, SCALE)
    attn_irp = score_times_v_irp(
        ctx, encoder, relin_key, galois_key,
        weights_ct, v_ct,
        D_HEAD, D_TOTAL, NUM_TOKENS, T_MODEL,
        sv_mask)
    _rec("attn_C", t0)
    if probe_np is not None and sk is not None:
        # attn_irp: stride-T_MODEL, slot[h*D_HEAD*T_MODEL + d] = attn[h, d]
        # ref: attn_P [N_HEADS*D_HEAD] = weights @ V_full reshaped
        _probe_diff("12.post_score_v", ctx, encoder, sk, attn_irp,
                    probe_np["attn_P"], stride=T_MODEL, count=N_HEADS * D_HEAD)

    # ---- Wo via IRP. attn_irp is already stride-T_MODEL at d=D_TOTAL —
    # exactly the layout Wo IRP expects, no relayout needed. The Wo IRP
    # galois keys live at USER_LEVEL_IRP_ATTN (target chain 26); the
    # incoming attn_irp may be at a shallower chain (smaller user level)
    # depending on the SDPA depth. mod_switch_to to align if needed. ----
    t0 = _t()
    irp_attn_ci = engine.user_level_chain_index(USER_LEVEL_IRP_ATTN)
    if attn_irp.chain_index() < irp_attn_ci:
        attn_irp = phantom.mod_switch_to(ctx, attn_irp, irp_attn_ci)
    o_ct = irp_matvec_host(ctx, encoder, galois_key, attn_irp, diag_wo_irp,
                      NUM_SLOTS, D_TOTAL, baby_steps=BABY_STEPS_IRP_SQUARE,
                      mask_pt=mask_attn_pt)
    _rec("wo_irp", t0)

    # ---- Wo IRP output is stride-t at d=D_TOTAL=D_MODEL: directly compatible
    # with the stride-t residual stream (same stride T_MODEL). No relayout
    # needed; rescale once to bring the IRP-internal SCALE^2 back to SCALE. ----
    t0 = _t()
    o_ct = phantom.rescale_to_next(ctx, o_ct)
    o_ct.set_scale(SCALE)
    _rec("layout_shift", t0)
    # Stage 13 probe (post-Wo) is emitted in run_decoder_fhe after return
    return o_ct


# ============================ MLP forward (IRP + bootstrap) ============================
def fhe_mlp_irp_bootstrap(engine, ctx, encoder, relin_key,
                            galois_key,
                            x_mid_norm,
                            diag_gate_irp, diag_up_irp, diag_down_irp,
                            sub_mask_wide_pt, sub_mask_tall_pt, input_mask_pt,
                            stage_times=None, max_abs_calib=None,
                            silu_coeffs=None, silu_norm_factor=None,
                            silu_t_coeffs=None, silu_D=None,
                            probe_np=None, sk=None,
                            verbose_mag=False):
    """MLP (SwiGLU) forward.
    probe_np/sk: when not None, emit [probe-L31] diff lines at each sub-stage.
    Wide (gate/up) IRP outputs are in permuted stride-t' layout; comparison is orderless.
    verbose_mag: when True (and sk is provided), print FHE max/mean magnitude at each stage.
    """
    def _vp(name, ct):
        if not verbose_mag or sk is None:
            return
        v = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                     dtype=np.float64)
        print(f"    [mlp-mag] {name:<24s} chain={ct.chain_index()}  "
              f"max|.|={np.abs(v).max():.4e}  mean|.|={np.abs(v).mean():.4e}")
    # Layer-0 defaults; overridden per-layer via max_abs_calib (Stage 3a').
    _calib = {"gate": 1.66, "up": 1.78, "h": 1.26}
    if max_abs_calib is not None:
        _calib.update({k: max_abs_calib[k] for k in ("gate", "up", "h") if k in max_abs_calib})

    def _t(): return time.perf_counter()
    def _rec(name, t0):
        if stage_times is None: return
        stage_times.setdefault(name, 0.0)
        stage_times[name] += (time.perf_counter() - t0) * 1000.0

    # ---- x_mid_norm is already stride-t (rmsnorm_forward_stride_t output);
    # mod-switch to USER_LEVEL_IRP_MLP so plaintext masks line up. ----
    t0 = _t()
    irp_mlp_ci = engine.user_level_chain_index(USER_LEVEL_IRP_MLP)
    if x_mid_norm.chain_index() < irp_mlp_ci:
        x_irp = phantom.mod_switch_to(ctx, x_mid_norm, irp_mlp_ci)
    else:
        x_irp = x_mid_norm
    _rec("layout_shift", t0)
    _vp("input_x_irp", x_irp)

    # Wide IRP (gate/up) output layout: permuted stride-t' where
    # t' = N / D_PAD_MLP = 32768 / 16384 = 2.
    # Meaningful slots at positions 0, 2, 4, ..., 2*(D_PAD_MLP-1).
    # Values at slot 2*c_perm = y[c' + q*D_MODEL] where c'=c_perm//alpha, q=c_perm%alpha,
    # alpha = D_PAD_MLP // D_MODEL = 4. Comparison done orderless (sorted magnitudes).
    _T_PRIME_WIDE = NUM_SLOTS // D_PAD_MLP   # = 2

    if diag_up_irp is None:
        # ---- Packed gate+up matvec: Wgate is in real, Wup in imag of the same
        # IRP plaintext set. One matvec produces a complex ct with
        # gate=re, up=im. We DEFER the real/imag extract until AFTER the
        # bootstrap (gate_up_ct is already the merged form merge_bootstrap
        # would construct; bootstrapping it directly saves the merge step).
        # Halves matvec time + halves IRP encoding time.
        t0 = _t()
        gate_up_ct = irp_matvec_rect_host(
            ctx, encoder, galois_key, x_irp, diag_gate_irp,
            NUM_SLOTS, D_MODEL, D_PAD_MLP,
            baby_steps=BABY_STEPS_IRP_MLP,
            sub_mask_pt=sub_mask_wide_pt)
        _rec("mlp_gate_up_packed", t0)
        gate_up_ct = phantom.rescale_to_next(ctx, gate_up_ct)
        gate_up_ct.set_scale(SCALE)
        # Mark gate_ct / up_ct as None — handled by the bootstrap block below.
        gate_ct = gate_up_ct
        up_ct = None  # signals "packed, in gate_ct's real+imag"
    else:
        # Separate gate, up matvecs (legacy path).
        t0 = _t()
        gate_ct = irp_matvec_rect_host(ctx, encoder, galois_key, x_irp, diag_gate_irp,
                                    NUM_SLOTS, D_MODEL, D_PAD_MLP,
                                    baby_steps=BABY_STEPS_IRP_MLP,
                                    sub_mask_pt=sub_mask_wide_pt)
        _rec("mlp_gate", t0)
        gate_ct = phantom.rescale_to_next(ctx, gate_ct)
        gate_ct.set_scale(SCALE)
        _vp("post_Wgate", gate_ct)
        if probe_np is not None and sk is not None:
            _probe_diff("17.post_Wgate", ctx, encoder, sk, gate_ct,
                        probe_np["gate_P"], stride=_T_PRIME_WIDE, count=D_PAD_MLP,
                        orderless=True)
        t0 = _t()
        up_ct = irp_matvec_rect_host(ctx, encoder, galois_key, x_irp, diag_up_irp,
                                  NUM_SLOTS, D_MODEL, D_PAD_MLP,
                                  baby_steps=BABY_STEPS_IRP_MLP,
                                  sub_mask_pt=sub_mask_wide_pt)
        _rec("mlp_up", t0)
        up_ct = phantom.rescale_to_next(ctx, up_ct)
        up_ct.set_scale(SCALE)
        _vp("post_Wup", up_ct)
        if probe_np is not None and sk is not None:
            _probe_diff("20.post_Wup", ctx, encoder, sk, up_ct,
                        probe_np["up_P"], stride=_T_PRIME_WIDE, count=D_PAD_MLP,
                        orderless=True)

    # ---- Bootstrap and split (gate, up). ----
    t0 = _t()
    pair_max_abs = max(_calib["gate"], _calib["up"])
    if up_ct is None:
        # Packed path: gate_ct already holds gate+i·up. Just bootstrap (use
        # sqrt(2)*max_abs for the complex magnitude bound), then extract
        # via conjugation.
        gate_ct = bootstrap_safe(engine, ctx, encoder, gate_ct,
                                   max_abs=pair_max_abs * 1.42,
                                   slot_count=NUM_SLOTS)
        from blocks.irp import extract_real_imag_pair
        gate_ct, up_ct = extract_real_imag_pair(
            ctx, encoder, galois_key, gate_ct, NUM_SLOTS, SCALE)
    else:
        from blocks.bootstrap import merge_bootstrap
        gate_ct, up_ct = merge_bootstrap(
            engine, ctx, encoder, gate_ct, up_ct,
            max_abs=pair_max_abs, slot_count=NUM_SLOTS, galois_key=galois_key)
    _rec("bootstrap", t0)
    _vp("post_gate_boot", gate_ct)
    _vp("post_up_boot", up_ct)
    if probe_np is not None and sk is not None:
        _probe_diff("18.post_gate_boot", ctx, encoder, sk, gate_ct,
                    probe_np["gate_P"], stride=_T_PRIME_WIDE, count=D_PAD_MLP,
                    orderless=True)
        _probe_diff("21.post_up_boot", ctx, encoder, sk, up_ct,
                    probe_np["up_P"], stride=_T_PRIME_WIDE, count=D_PAD_MLP,
                    orderless=True)

    # ---- silu(gate). ----
    t0 = _t()
    if silu_t_coeffs is not None and silu_D is not None:
        from blocks.silu import silu_clenshaw
        silu_gate = silu_clenshaw(engine, ctx, encoder, relin_key, gate_ct,
                                    silu_D, silu_t_coeffs, NUM_SLOTS,
                                    galois_key=galois_key)
    else:
        silu_gate = silu(ctx, encoder, relin_key, gate_ct, coeffs=silu_coeffs,
                         norm_factor=silu_norm_factor,
                         slot_count=NUM_SLOTS if silu_norm_factor is not None else None)
    _rec("mlp_silu", t0)
    _vp("post_silu", silu_gate)
    if probe_np is not None and sk is not None:
        _probe_diff("19.post_silu", ctx, encoder, sk, silu_gate,
                    probe_np["silu_gate_P"], stride=_T_PRIME_WIDE, count=D_PAD_MLP,
                    orderless=True)

    # ---- h = silu_gate * up. ----
    t0 = _t()
    s_ci = silu_gate.chain_index()
    u_ci = up_ct.chain_index()
    if u_ci < s_ci:
        up_ct = phantom.mod_switch_to(ctx, up_ct, s_ci)
    elif u_ci > s_ci:
        silu_gate = phantom.mod_switch_to(ctx, silu_gate, u_ci)
    silu_gate.set_scale(up_ct.scale())
    h_ct = phantom.multiply_and_relin(ctx, silu_gate, up_ct, relin_key)
    h_ct = phantom.rescale_to_next(ctx, h_ct)
    h_ct.set_scale(SCALE)
    _rec("mlp_swiglu", t0)
    _vp("post_swiglu_h", h_ct)
    if probe_np is not None and sk is not None:
        _probe_diff("22.post_swiglu", ctx, encoder, sk, h_ct,
                    probe_np["h_P"], stride=_T_PRIME_WIDE, count=D_PAD_MLP,
                    orderless=True)

    # ---- Refresh h via homomorphic bootstrap (preserves permuted layout). ----
    t0 = _t()
    h_fresh = bootstrap_safe(engine, ctx, encoder, h_ct,
                             max_abs=_calib["h"], slot_count=NUM_SLOTS)
    # Mod-switch to IRP_MLP chain so plaintext chain indices align and GPU
    # memory stays within budget (chain 16 has 30 primes; chain 26 has 5).
    irp_mlp_chain = engine.user_level_chain_index(USER_LEVEL_IRP_MLP)
    h_fresh = phantom.mod_switch_to(ctx, h_fresh, irp_mlp_chain)
    _rec("bootstrap", t0)
    _vp("post_h_boot", h_fresh)
    if probe_np is not None and sk is not None:
        _probe_diff("23.post_h_boot", ctx, encoder, sk, h_fresh,
                    probe_np["h_P"], stride=_T_PRIME_WIDE, count=D_PAD_MLP,
                    orderless=True)

    # ---- out = Wdown @ h  (rect tall). ----
    t0 = _t()
    out_ct = irp_matvec_rect_host(ctx, encoder, galois_key, h_fresh, diag_down_irp,
                               NUM_SLOTS, D_PAD_MLP, D_MODEL,
                               baby_steps=BABY_STEPS_IRP_MLP,
                               sub_mask_pt=sub_mask_tall_pt,
                               input_mask_pt=input_mask_pt)
    _rec("mlp_down", t0)

    # ---- Wdown IRP output is stride-t at d=D_MODEL: directly compatible with
    # the stride-t residual stream. Rescale once to bring SCALE^2 -> SCALE. ----
    t0 = _t()
    out_ct = phantom.rescale_to_next(ctx, out_ct)
    out_ct.set_scale(SCALE)
    out_periodic = out_ct
    _rec("layout_shift", t0)
    _vp("post_Wdown_out", out_ct)
    if probe_np is not None and sk is not None:
        _probe_diff("24.post_Wdown", ctx, encoder, sk, out_ct,
                    probe_np["out_P"], stride=T_MODEL, count=D_MODEL)
    return out_periodic


# ============================ Driver ============================
def main():
    # probe v1: 4-token "[BOS] The quick brown" embed + RoPE tables + layer-0
    # ref_out (used for the legacy comparison line). Per-layer weights load
    # from probe v2 inside the per-layer loop.
    L = lambda n: np.load(f"{PROBE}/{n}.npy")
    embed   = L("embed");  ref_out = L("ref_out")
    cos_all, sin_all = L("rope_cos"), L("rope_sin")
    R_P = rope_matrix_np(cos_all[P], sin_all[P])

    # PyTorch reference hidden states for the same 4-token prompt:
    # shape (33, 4, 4096) = (input_embed + 32 layer outputs, NUM_TOKENS, D_MODEL).
    pytorch_ref = np.load(f"{PROBE_FULL}/ref_acts/qbrown4_bos.npy").astype(np.float64)
    print(f"PyTorch reference loaded: shape={pytorch_ref.shape}")

    # ---- Step union (full rotation step inventory). ----
    # Stage-2 IRP-native SDPA: stride-T_MODEL throughout. No periodic relayout
    # in the decoder, so no replicate_required_steps and the SDPA step inventory
    # is the IRP-native one (qkt_irp + softmax_irp_t + score_v_irp).
    sdpa_steps = sdpa_irp_required_steps(D_HEAD, D_TOTAL, NUM_TOKENS, T_MODEL)
    # Stride-t rmsnorm uses {T_MODEL, 2*T_MODEL, ..., (D_MODEL/2)*T_MODEL}
    # instead of {1, 2, ..., D_MODEL/2}.
    rms_steps  = rmsnorm_required_steps_stride_t(D_MODEL, T_MODEL)
    irp_attn_steps = irp_required_steps(NUM_SLOTS, D_TOTAL,
                                          baby_steps=BABY_STEPS_IRP_SQUARE)
    irp_mlp_w_steps = irp_required_steps_rect(NUM_SLOTS, D_MODEL, D_PAD_MLP,
                                                baby_steps=BABY_STEPS_IRP_MLP)
    irp_mlp_t_steps = irp_required_steps_rect(NUM_SLOTS, D_PAD_MLP, D_MODEL,
                                                baby_steps=BABY_STEPS_IRP_MLP)
    user_steps = sorted(set(list(rms_steps) + list(sdpa_steps)
                            + list(irp_attn_steps) + list(irp_mlp_w_steps)
                            + list(irp_mlp_t_steps)))

    # ---- Per-step galois target chain assignment. ----
    #
    # target_chain_index=T means the key covers ciphertexts at chain >= T
    # (size_Ql = size_Q - (T-1); larger T = fewer limbs = smaller key).
    # Each step's target = shallowest chain at which it fires in the pipeline.
    #
    # freshest_chain_index = 16 (fixed: 1+3(C2S)+9(ER)+3(S2C); invariant to NSL).
    # NSL=14 → pre_boot chain = 16+13 = 29.
    #
    # Pipeline chain trace (between bootstraps each stage restarts at 16):
    #
    # rms1/rms2 (right after bootstrap at chain 16):
    #   sum_reduce stride-T_MODEL {8,16,...,8192} fire at chain 16 → target=16
    #   (compute_qkt_irp reduce {8..512} also overlap with rms steps; reduce
    #    fires at chain ~28 inside Stage A. Min target wins → 16.)
    #
    # finalize_softmax_irp_t (stage C, after bootstrap B→C):
    #   mask+rescale (1 level) → chain 17; finalize_softmax receives e_ct at chain 17
    #   - rotate(-NUM_TOKENS) cyclic-replica fill at chain 17  → target=17
    #   - sum_reduce stride=1 {1, 2} at chain 17               → target=17
    #     ({1, 2} also fire later in score_v reduce at chain 24 — min wins → 17.)
    #
    # score_times_v_irp broadcast {-T_MODEL*2^s : s<log2(D_HEAD)} (stage C):
    #   finalize_softmax output at chain 17+6=23 (6 Goldschmidt levels) → target=23
    #
    # IRP-only steps (all 44, both attn and MLP):
    #   IRP plaintexts encoded for ct at USER_LEVEL_IRP_ATTN=10 → chain 26 → target=26
    #   (All IRP variants — preprocess, babies, giants, reduce — fire at this chain.)
    #   compute_qkt_irp Q preprocess {(D_TOTAL-1)*2^s} fires at chain 27 (post-Wq IRP
    #   mask+rescale); these steps overlap with Wq IRP preprocess → target=26.

    FRESHEST_CHAIN = 16    # invariant to NSL for our bootstrap pipeline
    # rms inner-sum fires at chain 17 (after x^2 consumes one level), not at
    # the freshest chain — confirmed by per-call rotation audit. Same for
    # qkt_q_preprocess (post-Wq IRP rescale).
    TARGET_RMS          = FRESHEST_CHAIN + 1    # 17: rms (positive sum_reduce strides)
    TARGET_FINALIZE     = FRESHEST_CHAIN + 1    # 17: finalize_softmax cyclic + sum_reduce
    TARGET_SCORE_V      = FRESHEST_CHAIN + 7    # 23: score_v broadcast (6 Goldschmidt + 1 mask)
    TARGET_IRP          = FRESHEST_CHAIN + USER_LEVEL_IRP_ATTN  # 26: all IRP ops (ul=10)

    rms_set      = set(rms_steps)
    sdpa_set     = set(sdpa_steps)
    irp_all_set  = set(irp_attn_steps) | set(irp_mlp_w_steps) | set(irp_mlp_t_steps)
    irp_only_set = irp_all_set - rms_set - sdpa_set

    # New-pipeline SDPA-only steps:
    #   {-2^s : s<log2(T_MODEL)} = {-1, -2, -4} : compute_qkt_irp Q preprocess
    #     fires at chain 16 (post-bootstrap right after Wq IRP) → target 16.
    #   {-NUM_TOKENS} = {-4}  : finalize_softmax cyclic-replica fires at chain 17.
    #     Step -4 is shared with Q preprocess; min target wins → 16.
    #   {1, 2}                : finalize sum_reduce + score_v reduce  (also in
    #     IRP @ 26); finalize fires at chain 17 → target 17.
    #   {-T_MODEL*2^s}        : score_v broadcast  (target 23)
    qkt_q_preprocess_steps = {-int(1 << s) for s in range(int(round(math.log2(T_MODEL))))}
    sdpa_finalize_steps    = {1, 2}
    sdpa_score_v_steps     = {-int(T_MODEL * (1 << s)) for s in range(int(round(math.log2(D_HEAD))))}

    target_chain_indices = []
    for s in user_steps:
        if s in rms_set:
            target_chain_indices.append(TARGET_RMS)           # 16
        elif s in qkt_q_preprocess_steps:
            target_chain_indices.append(TARGET_RMS)           # 16 (post-bootstrap qkt)
        elif s in sdpa_finalize_steps:
            target_chain_indices.append(TARGET_FINALIZE)      # 17
        elif s in sdpa_score_v_steps:
            target_chain_indices.append(TARGET_SCORE_V)       # 23
        elif s in irp_only_set:
            target_chain_indices.append(TARGET_IRP)           # 26
        else:
            # Fallback (should not happen with correct step enumeration)
            target_chain_indices.append(TARGET_RMS)

    # Resolve galois-element collisions: two rotation steps can share the same
    # galois element (e.g. step -4 and step +32764 both map to elt 84145 for
    # N=65536).  The engine generates one key per galois element and the last
    # write wins, so a deep-chain target (large T, small beta_k) can silently
    # overwrite a shallow-chain target (small T, large beta_k).  If the key is
    # then applied at a chain that needs beta_ct > beta_k the kernel reads an
    # out-of-bounds pointer → illegal memory access.  Fix: for each galois
    # element keep the MINIMUM target (= shallowest chain = largest beta_k).
    def _galois_elt(step):
        m = 2 * N
        power = (step % (N // 2)) + (N // 2) if step < 0 else step % (N // 2)
        return pow(3, power, m)

    elt_min_target = {}
    for s, t in zip(user_steps, target_chain_indices):
        e = _galois_elt(s)
        if e not in elt_min_target or t < elt_min_target[e]:
            elt_min_target[e] = t

    resolved = []
    for s, t in zip(user_steps, target_chain_indices):
        e = _galois_elt(s)
        resolved_t = elt_min_target[e]
        if resolved_t != t:
            print(f"  [collision fix] step={s} elt={e}: target {t} -> {resolved_t}")
        resolved.append(resolved_t)
    target_chain_indices = resolved

    by_target = {}
    for s, t in zip(user_steps, target_chain_indices):
        by_target.setdefault(t, []).append(s)
    print(f"Per-step galois target chain assignment:")
    for t in sorted(by_target):
        steps_at_t = by_target[t]
        print(f"  chain={t}: {len(steps_at_t):3d} steps  {sorted(steps_at_t)[:5]}"
              f"{'...' if len(steps_at_t)>5 else ''}")
    print(f"  total user steps: {len(user_steps)}")

    cfg = phantom.ckks_engine_config()
    cfg.log_n = LOG_N
    cfg.user_scale = SCALE
    cfg.num_scale_levels = NUM_SCALE_LEVELS
    cfg.sparse_hw = SPARSE_HW
    cfg.num_special_primes = NUM_SPECIAL_PRIMES
    cfg.include_user_rotations = False
    cfg.user_rotation_steps = user_steps
    cfg.user_rotation_target_chain_indices = target_chain_indices

    print(f"Constructing CKKSEngine: logN={LOG_N} num_scale_levels={NUM_SCALE_LEVELS} "
          f"num_special={NUM_SPECIAL_PRIMES} #user_steps={len(user_steps)}")
    t0 = time.perf_counter()
    engine = phantom.ckks_engine(cfg)
    print(f"engine ctor: {time.perf_counter()-t0:.1f}s  max_user_level={engine.max_user_level()}")

    ctx = engine.context()
    encoder = engine.encoder()
    sk = engine.secret_key()
    relin_key = engine.relin_key()
    galois_key = engine.galois_key()
    fresh_ci = engine.user_level_chain_index(0)
    print(f"freshest chain_index={fresh_ci}")

    # ---- Pre-encode IRP masks (layer-independent). Per-layer Wq/Wo/Wgate/
    # Wup/Wdown IRP plaintexts encode inside the loop because each layer has
    # different weights and the full set won't fit in GPU memory. ----
    irp_attn_chain = engine.user_level_chain_index(USER_LEVEL_IRP_ATTN)
    irp_mlp_chain  = engine.user_level_chain_index(USER_LEVEL_IRP_MLP)
    mask_attn_pt = encode_irp_mask(ctx, encoder, NUM_SLOTS, D_TOTAL, SCALE, irp_attn_chain)
    sub_mask_mlp_wide_pt = encode_irp_mask_rect(
        ctx, encoder, NUM_SLOTS, D_MODEL, D_PAD_MLP, SCALE, irp_mlp_chain)
    sub_mask_mlp_tall_pt = encode_irp_mask_rect(
        ctx, encoder, NUM_SLOTS, D_PAD_MLP, D_MODEL, SCALE, irp_mlp_chain + 1)
    input_mask_mlp_pt = encode_irp_mask(ctx, encoder, NUM_SLOTS, D_MODEL,
                                          SCALE, irp_mlp_chain)
    def _make_rms_params(zmin, zmax):
        p = phantom.rmsnorm_params()
        p.d_model    = D_MODEL
        p.epsilon    = EPSILON
        p.z_min      = zmin
        p.z_max      = zmax
        p.poly_degree = RMS_POLY_DEG
        return p
    # ---- Run + measure ----
    print(f"\nLLaMA-3.1-8B layer-0 IRP+bootstrap, prompt='The quick brown fox', query position {P}")
    print(f"FHE config: scale=2^{int(math.log2(SCALE))}  galois_steps={len(user_steps)}")

    # ---- Cachemir Section 6: bootstrap placement via DAG shortest-path. ----
    # Decompose the decoder pipeline into outer-boundary layers. Each entry is:
    #   (name, depth, runtime_ms, can_bootstrap_at_input, output_level, requires_fresh_input)
    # Where:
    #   - depth          : multiplicative depth ℓ(i) consumed by the layer
    #   - output_level   : if the layer ends with a free decrypt+re-encrypt to
    #                      USER_LEVEL_FRESH=0 the output level is fixed at
    #                      max_user_level; expressed as 13 here
    #   - requires_fresh_input : rmsnorm galois keys are only generated at the
    #                      freshest chain (TARGET_RMS=16), so rms* must enter
    #                      with input_level == max_user_level
    #
    # In-module bootstrapping (paper §6 last paragraphs): the attention block's
    # internal bootstraps (between attn_A/attn_B and attn_B/attn_C) are forced
    # by depth (attn_B is 12 levels deep; attn_C is 6+ levels deep with input
    # level 1 after attn_B) — these belong inside the block and are absorbed
    # into the block's runtime_ms. The DAG search here decides only the
    # *outer* bootstrap placement (between the 6 outer-boundary layers).
    NSL_MAX = NUM_SCALE_LEVELS - 1   # = 13
    # T_BOOT calibrated from the baseline run (910 ms / 5 calls ≈ 182 ms each).
    T_BOOT_MS = 182.0
    # With stride-t residual stream the attention/mlp output relayouts to
    # USER_LEVEL_FRESH are gone; the blocks now end at chain after Wo/Wdown IRP
    # plus one extra rescale (to bring SCALE^2 back to SCALE). That is
    # user_level = USER_LEVEL_IRP_*+2 = 12 = NSL_MAX-1. Internal bootstraps
    # within attention/mlp are still accounted for in their runtime_ms.
    OUTPUT_LEVEL_AFTER_IRP = USER_LEVEL_IRP_ATTN + 2  # = 12
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
    print("\n=== Cachemir §6 bootstrap placement (DAG shortest-path) ===")
    print(render_plan_table(plan))
    boot_before = {plan.layers[s.layer_idx].name: s.bootstrap_before for s in plan.steps}

    NUM_DECODERS = 32
    # Stage 3a: each layer is tested in isolation against PyTorch ground truth.
    # x_btd at layer L is set to pytorch_ref[L] (PyTorch's hidden state at L's
    # input). FHE output at P should then match pytorch_ref[L+1, P]. This avoids
    # the "numpy stack only advances y[P]" approximation that drifts from real
    # LLaMA prefill semantics at layer 1+.
    for layer_idx in range(NUM_DECODERS):
        x_btd = pytorch_ref[layer_idx].copy()  # [NUM_TOKENS, D_MODEL]
        print(f"\n=========== Decoder {layer_idx} ===========")
        t_layer0 = time.perf_counter()
        # Real per-layer LLaMA-3.1-8B weights
        w = load_layer_weights(layer_idx)
        Wq_baked, diag_wq_irp, diag_wo_irp, diag_gate_irp, diag_up_irp, diag_down_irp = \
            encode_layer_irps(ctx, encoder, w, R_P)
        t_irp = (time.perf_counter() - t_layer0) * 1000
        print(f"  [irp-encode] layer {layer_idx} done in {t_irp:.0f}ms")

        # numpy reference for this layer (per-layer weights)
        y_btd_np = forward_decoder_np(x_btd, w["g1"], w["g2"],
                                       w["Wq"], w["Wk"], w["Wv"], w["Wo"],
                                       w["Wgate"], w["Wup"], w["Wdown"],
                                       cos_all, sin_all, P)
        # per-layer rmsnorm calibration
        z1_l, z2_l = compute_layer_z(x_btd, w["g1"], w["g2"],
                                      w["Wq"], w["Wk"], w["Wv"], w["Wo"],
                                      cos_all, sin_all, P)
        z1_min, z1_max = rms_z_window(z1_l)
        z2_min, z2_max = rms_z_window(z2_l)
        # Per-layer bootstrap_safe calibrations from numpy forward.
        max_abs_calib = compute_layer_max_abs(x_btd, w, cos_all, sin_all, P)
        # Per-layer silu polynomial fit to this layer's actual gate range.
        # max_abs_calib["gate"] is the numpy gate-magnitude × BOOT_CALIB_MARGIN
        # (1.5x). For silu we use a tighter margin (1.05x) because: (1) FHE
        # noise on gate is ~1e-3, far less than the 1.5x bootstrap_safe margin
        # needs; (2) deg-14 Chebyshev fit error scales rapidly with domain
        # width, so a wider-than-necessary domain costs precision near 0
        # where most gate slots cluster.
        silu_max = max_abs_calib["gate"] / BOOT_CALIB_MARGIN  # un-margined
        silu_domain = (-silu_max * 1.2, silu_max * 1.2)
        silu_coeffs = fit_silu_coeffs(silu_domain, deg=14)
        print(f"  [calib] layer {layer_idx}: z1={z1_l:.3e} z2={z2_l:.3e}; "
              f"q={max_abs_calib['q']:.2f} scores={max_abs_calib['scores']:.2f} "
              f"gate={max_abs_calib['gate']:.2f} (silu ±{silu_domain[1]:.2f}) "
              f"up={max_abs_calib['up']:.2f} h={max_abs_calib['h']:.2f}")
        rms1_p = _make_rms_params(z1_min, z1_max)
        rms2_p = _make_rms_params(z2_min, z2_max)
        rms1_w = setup_rmsnorm_weights(ctx, encoder, rms1_p, w["g1"].tolist(), stride=T_MODEL)
        rms2_w = setup_rmsnorm_weights(ctx, encoder, rms2_p, w["g2"].tolist(), stride=T_MODEL)
        # encrypt this layer's inputs (boundary sk-touch — Path A: K/V re-encrypted
        # per layer from numpy y_l_np; x re-encrypted too, no across-layer ct carry).
        x_ct, k_ct, v_ct, c_per_head = encrypt_layer_inputs(
            ctx, encoder, sk, fresh_ci, x_btd,
            w["g1"], Wq_baked, w["Wk"], w["Wv"],
            cos_all, sin_all, P)

        # ---- Build probe_np dict for L31 per-stage diff probes. ----
        # All intermediates computed from the same weights and x_btd that the
        # FHE pipeline uses, so FHE-vs-numpy diffs isolate FHE error alone.
        _pnp = None
        if layer_idx == 31:
            _xn = rmsnorm_np(x_btd, w["g1"])               # post-rms1, [NUM_TOKENS, D_MODEL]
            _K = ((_xn @ w["Wk"].T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD))
            _V = ((_xn @ w["Wv"].T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD))
            _K = apply_rope_np(_K, cos_all, sin_all)
            _K_full = np.repeat(_K, N_KV_GROUPS, axis=1)   # [NUM_TOKENS, N_HEADS, D_HEAD]
            _V_full = np.repeat(_V, N_KV_GROUPS, axis=1)
            # Q at position P: Wq_baked already includes RoPE for position P
            _Q_P = (_xn[P] @ Wq_baked.T).reshape(N_HEADS, D_HEAD)  # [N_HEADS, D_HEAD]
            # scores[tok, h] = Q_P[h] . K_full[tok, h] / sqrt(D_HEAD)
            _scores = np.einsum('hd,thd->th', _Q_P, _K_full) / math.sqrt(D_HEAD)
            _c_per_head = _scores.max(0) + 0.5              # [N_HEADS]
            _scores_minus_c = _scores - _c_per_head[None, :]  # [NUM_TOKENS, N_HEADS]
            # softmax (true, not FHE approx)
            _exp_s = np.exp(_scores_minus_c - _scores_minus_c.max(0, keepdims=True))
            _softmax_w = _exp_s / _exp_s.sum(0, keepdims=True)  # [NUM_TOKENS, N_HEADS]
            # attention output at P: weights @ V, then Wo
            _attn_P = np.einsum('th,thd->hd', _softmax_w, _V_full).reshape(N_HEADS * D_HEAD)
            _o_P = _attn_P @ w["Wo"].T                      # [D_MODEL]
            _x_mid = x_btd.copy(); _x_mid[P] = x_btd[P] + _o_P
            _x_mid_n = rmsnorm_np(_x_mid, w["g2"])
            _gate_P = _x_mid_n[P] @ w["Wgate"].T            # [D_HIDDEN]
            # Pad gate/up to D_PAD_MLP to match what the FHE IRP produces
            _gate_P_pad = np.zeros(D_PAD_MLP, dtype=np.float64)
            _gate_P_pad[:D_HIDDEN] = _gate_P
            _silu_gate_P_pad = np.zeros(D_PAD_MLP, dtype=np.float64)
            _silu_gate_P_pad[:D_HIDDEN] = silu_np(_gate_P)
            _up_P = _x_mid_n[P] @ w["Wup"].T               # [D_HIDDEN]
            _up_P_pad = np.zeros(D_PAD_MLP, dtype=np.float64)
            _up_P_pad[:D_HIDDEN] = _up_P
            _h_P = silu_np(_gate_P) * _up_P                 # [D_HIDDEN]
            _h_P_pad = np.zeros(D_PAD_MLP, dtype=np.float64)
            _h_P_pad[:D_HIDDEN] = _h_P
            _out_P = _h_P @ w["Wdown"].T                    # [D_MODEL]
            _y_P = _x_mid[P] + _out_P
            _pnp = {
                "x_btd_P":       x_btd[P],                  # [D_MODEL]
                "xn_P":          _xn[P],                     # [D_MODEL]
                "Q_P_flat":      _Q_P.reshape(-1),           # [D_TOTAL]
                "scores_minus_c": _scores_minus_c,           # [NUM_TOKENS, N_HEADS]
                "softmax_weights": _softmax_w,               # [NUM_TOKENS, N_HEADS]
                "attn_P":        _attn_P,                    # [N_HEADS*D_HEAD]
                "o_P":           _o_P,                       # [D_MODEL]
                "x_mid_P":       _x_mid[P],                  # [D_MODEL]
                "x_mid_n_P":     _x_mid_n[P],               # [D_MODEL]
                "gate_P":        _gate_P_pad,                # [D_PAD_MLP]
                "silu_gate_P":   _silu_gate_P_pad,           # [D_PAD_MLP]
                "up_P":          _up_P_pad,                  # [D_PAD_MLP]
                "h_P":           _h_P_pad,                   # [D_PAD_MLP]
                "out_P":         _out_P,                     # [D_MODEL]
                "y_P":           _y_P,                       # [D_MODEL]
            }

        # FHE forward
        y_full_fhe, total_ms, stage_times, _y_ct_out = run_decoder_fhe(
            engine, ctx, encoder, sk, relin_key, galois_key,
            x_ct, k_ct, v_ct, c_per_head,
            diag_wq_irp, diag_wo_irp, mask_attn_pt,
            diag_gate_irp, diag_up_irp, diag_down_irp,
            sub_mask_mlp_wide_pt, sub_mask_mlp_tall_pt, input_mask_mlp_pt,
            rms1_w, rms2_w, rms1_p, rms2_p, boot_before,
            label=f"decoder{layer_idx}", max_abs_calib=max_abs_calib,
            silu_coeffs=silu_coeffs,
            probe=(layer_idx == 31), probe_np=_pnp)
        y_p_fhe = y_full_fhe[::T_MODEL][:D_MODEL]
        # accuracy vs numpy reference at P
        err_np = y_p_fhe - y_btd_np[P]
        # accuracy vs PyTorch reference at P (qbrown4_bos.npy[layer+1, P])
        y_p_torch = pytorch_ref[layer_idx + 1, P]
        err_torch = y_p_fhe - y_p_torch
        print(f"Decoder {layer_idx}:")
        print(f"  ‖y_fhe‖           = {np.linalg.norm(y_p_fhe):.4f}")
        print(f"  ‖y_np (forward_np)‖ = {np.linalg.norm(y_btd_np[P]):.4f}")
        print(f"  ‖y_torch‖         = {np.linalg.norm(y_p_torch):.4f}")
        print(f"  vs numpy:   max|err| = {float(np.abs(err_np).max()):.3e}  "
              f"rel-RMS = {float(np.linalg.norm(err_np)/np.linalg.norm(y_btd_np[P])):.3e}")
        print(f"  vs pytorch: max|err| = {float(np.abs(err_torch).max()):.3e}  "
              f"rel-RMS = {float(np.linalg.norm(err_torch)/np.linalg.norm(y_p_torch)):.3e}")
        print(f"  total = {total_ms:.1f} ms (incl. {t_irp:.0f}ms IRP encode)")
        if layer_idx == 0:
            # legacy probe-v1 ref_out comparison (matches pytorch[1] within ~1e-4)
            err_hf = y_p_fhe - ref_out[P]
            print(f"  vs probe v1 ref_out: max|err| = {float(np.abs(err_hf).max()):.3e}")


if __name__ == "__main__":
    main()
