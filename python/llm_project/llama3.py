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

# NOTE: This module is now a plaintext-reference + constants provider for the
# dense MRPC pipeline (llama3_mrpc.py). The legacy IRP/old-linear FHE driver
# (run_decoder_fhe / fhe_attention_irp_bootstrap / fhe_mlp_irp_bootstrap /
# main) and its blocks.irp / blocks.attention-IRP imports were removed during
# the dense rewrite. Only numpy/math/phantom and module constants are used by
# the surviving helpers below.


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


def load_layer_weights_subset(layer_idx, keys=("Wq", "Wk", "Wv",
                                                  "g1", "g2")):
    """Load a subset of per-layer weights. Returns a dict with just `keys`.

    DEFAULT IS THE 5-KEY PER-EXAMPLE HOT-PATH SUBSET (Wq/Wk/Wv/g1/g2).
    The R_P-independent weights (Wo/Wgate/Wup/Wdown) are NOT included
    by default because they are served from the shared rp_indep_cache
    on the per-example FHE path. The parallel sweep precomputes layer
    calibration ONCE at startup (one representative example, full weights
    loaded one layer at a time then freed) so per-example workers never
    need the big weights and the in-RAM preload stays at ~12 GB total
    (5 keys × 32 layers) instead of ~60 GB.

    Callers that want the full 9-key set can pass `keys=(...)` explicitly
    or use `load_layer_weights` directly.
    """
    ld = f"{PROBE_FULL}/layer_{layer_idx:02d}"
    return {k: np.load(f"{ld}/{k}.npy").astype(np.float64) for k in keys}


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

