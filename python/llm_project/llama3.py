"""
LLaMA-3.1-8B layer-0 single decoder, Phase 1: homomorphic refresh in MLP block
via boot_centered; layout shifts (relayout_periodic_to_irp / relayout_irp_to_periodic)
still use decrypt+re-encrypt and are pending Phase 2-3 implementation.
Cachemir IRP plaintext-encoding swap with per-step galois target chain indices
to fit on a 32 GB GPU.

The pre-IRP version of this file (BSGS Wq/Wo + complex BSGS Wgate/Wup/Wdown)
peaked at ~30,580 MiB and OOMed during the first `engine.bootstrap_inplace`.
This version mirrors `llama3_simulation.py`'s host-stored IRP plaintext
encoding (Wq, Wo, Wgate, Wup, Wdown) and uses the new
`CKKSEngineConfig.user_rotation_target_chain_indices` to assign each user
rotation step the smallest galois key (deepest chain target) compatible with
its actual call depth. The pre-IRP attention/MLP plaintext bulk (~30 GiB)
collapses to ~3 GiB; the per-step galois bundle shrinks the engine's static
GPU footprint by another several GiB.

Pipeline:
  rms1 -> bootstrap -> [shift to IRP] -> Wq IRP -> [shift to periodic d_total]
       -> compute_qkt -> mask*scale -> sub(C[h]) -> bootstrap -> ps_exp + damped sq
       -> bootstrap -> mask -> finalize_softmax -> score_times_v -> mask + replicate
       -> [shift to IRP] -> Wo IRP -> [shift to periodic d_model] -> +x_ct (residual1)
       -> bootstrap -> rms2 -> bootstrap -> [shift to IRP] -> Wgate IRP (wide) -> silu
                                                          -> Wup IRP (wide) -> ct*ct
                                                          -> [refresh] -> Wdown IRP (tall)
                                                          -> [shift to periodic d_model]
       -> +x_mid (residual2) -> decrypt y_ct
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
    mask_scale_plaintext, score_mask_plaintext,
    sdpa_required_steps,
)
from blocks.softmax import softmax_damping_schedule
from blocks.silu import silu
from blocks.linear import replicate_required_steps
from blocks.bootstrap import boot_centered
from blocks.bootstrap_placement import (
    build_layers_from_table, find_optimal_placement, render_plan_table,
)
from blocks.residual import residual
from blocks.rmsnorm import rmsnorm_forward, rmsnorm_required_steps, setup_rmsnorm_weights


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
RMS1_Z_MIN, RMS1_Z_MAX = 8e-5, 1.3e-4
RMS2_Z_MIN, RMS2_Z_MAX = 1.4e-4, 2.1e-4

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
#   MLP gate/up: 1 level each IRP → boot_centered refresh (fresh chain ~13 ul above msg)
#   silu:        ~4 levels  (deg-8 poly)  ← fits within freshened budget
#   swiglu:      1 level    (ct*ct)       → boot_centered refresh before Wdown
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


# ============================ Layout shim helpers ============================
# Decrypt + re-encrypt at a chosen user level (chain index = engine.user_level_chain_index(L)).
# This reuses the same refresh mechanism used by llama3_simulation.py for
# layout shifts between IRP-interleaved and periodic packings; the only
# difference here is that we encrypt at a chain in the engine's user segment.
def _re_encrypt_slots(engine, ctx, encoder, sk, slots, user_level, scale=SCALE):
    chain_index = engine.user_level_chain_index(user_level)
    return sk.encrypt_symmetric(
        ctx, encoder.encode_double_vector(ctx, slots.tolist(), scale, chain_index))

def relayout_periodic_to_irp(engine, ctx, encoder, sk, ct, d_periodic, d_irp,
                              user_level, scale=SCALE):
    dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                   dtype=np.float64)[:d_periodic]
    t = NUM_SLOTS // d_irp
    slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    n = min(d_periodic, d_irp)
    slots[::t][:n] = dec[:n]
    return _re_encrypt_slots(engine, ctx, encoder, sk, slots, user_level, scale)

def relayout_irp_to_periodic(engine, ctx, encoder, sk, ct, d_in, d_out, user_level, scale=SCALE):
    dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                   dtype=np.float64)
    t_in = NUM_SLOTS // d_in
    valid = dec[::t_in][:d_in]
    slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    periods = NUM_SLOTS // d_out
    n = min(d_in, d_out)
    for k in range(periods):
        slots[k*d_out : k*d_out + n] = valid[:n]
    return _re_encrypt_slots(engine, ctx, encoder, sk, slots, user_level, scale)


# ============================ Attention forward (IRP + bootstrap) ============================
def fhe_attention_irp_bootstrap(engine, ctx, encoder, sk, relin_key,
                                 galois_key,
                                 x_norm,
                                 diag_wq_irp, diag_wo_irp,
                                 mask_attn_pt,
                                 k_ct, v_ct, c_per_head,
                                 stage_times=None):
    def _t(): return time.perf_counter()
    def _rec(name, t0):
        if stage_times is None: return
        stage_times.setdefault(name, 0.0)
        stage_times[name] += (time.perf_counter() - t0) * 1000.0

    # ---- shift periodic d_model -> IRP-interleaved at d=d_total. ----
    t0 = _t()
    x_irp = relayout_periodic_to_irp(engine, ctx, encoder, sk, x_norm,
                                       D_MODEL, D_TOTAL,
                                       user_level=USER_LEVEL_IRP_ATTN)
    _rec("layout_shift", t0)

    # ---- Wq via IRP. ----
    t0 = _t()
    q_ct = irp_matvec_host(ctx, encoder, galois_key, x_irp, diag_wq_irp,
                      NUM_SLOTS, D_TOTAL, baby_steps=BABY_STEPS_IRP_SQUARE,
                      mask_pt=mask_attn_pt)
    _rec("wq_irp", t0)

    # ---- shift interleaved -> periodic d_total at fresh chain (for SDPA). ----
    t0 = _t()
    q_ct = relayout_irp_to_periodic(engine, ctx, encoder, sk, q_ct,
                                     D_TOTAL, D_TOTAL,
                                     user_level=USER_LEVEL_FRESH)
    _rec("layout_shift", t0)

    # ---- compute_qkt + mask*scale + sub(C[h]). ----
    t0 = _t()
    phantom.mod_switch_to_inplace(ctx, k_ct, q_ct.chain_index())
    scores_ct = phantom.compute_qkt(ctx, relin_key, galois_key, q_ct, [k_ct], D_HEAD)[0]
    nominal = scores_ct.scale()
    inv_sqrt_d = 1.0 / math.sqrt(float(D_HEAD))
    ms_pt = mask_scale_plaintext(
        ctx, encoder, D_HEAD, D_TOTAL, NUM_TOKENS,
        inv_sqrt_d, scores_ct.chain_index(), SCALE)
    scores_ct = phantom.multiply_plain(ctx, scores_ct, ms_pt)
    scores_ct = phantom.rescale_to_next(ctx, scores_ct)
    scores_ct.set_scale(nominal)
    sub_mask = np.zeros(NUM_SLOTS, dtype=np.float64)
    for tt in range(NUM_TOKENS):
        for h in range(N_HEADS):
            sub_mask[tt * D_TOTAL + h * D_HEAD] = c_per_head[h]
    sub_pt = encoder.encode_double_vector(
        ctx, sub_mask.tolist(), scores_ct.scale(), scores_ct.chain_index())
    scores_ct = phantom.sub_plain(ctx, scores_ct, sub_pt)
    _rec("attn_A", t0)

    # ---- bootstrap before damped squarings. ----
    t0 = _t()
    scores_ct = boot_centered(engine, ctx, encoder, sk, scores_ct)
    _rec("bootstrap", t0)

    # ---- ps_exp_init + damped squarings. ----
    t0 = _t()
    damps = softmax_damping_schedule(NUM_SQUARINGS, NUM_TOKENS, EXTRA_SCALE, TARGET_MAG)
    e_ct = phantom.ps_exp_init(
        ctx, encoder, relin_key, scores_ct,
        NUM_TOKENS, NUM_SQUARINGS, EXTRA_SCALE)
    phantom.square_iterations_damped_inplace(ctx, encoder, relin_key, e_ct, damps)
    _rec("attn_B", t0)

    # ---- bootstrap before finalize_softmax. ----
    t0 = _t()
    e_ct = boot_centered(engine, ctx, encoder, sk, e_ct)
    _rec("bootstrap", t0)

    # ---- mask + finalize_softmax + score*V + mask + replicate. ----
    t0 = _t()
    mask_arr = np.zeros(NUM_SLOTS, dtype=np.float64)
    for tt in range(NUM_TOKENS):
        for h in range(N_HEADS):
            mask_arr[tt * D_TOTAL + h * D_HEAD] = 1.0
    e_nominal = e_ct.scale()
    mask_pt = encoder.encode_double_vector(
        ctx, mask_arr.tolist(), SCALE, e_ct.chain_index())
    e_ct = phantom.multiply_plain(ctx, e_ct, mask_pt)
    e_ct = phantom.rescale_to_next(ctx, e_ct)
    e_ct.set_scale(e_nominal)

    weights_ct = phantom.finalize_softmax(
        ctx, encoder, relin_key, galois_key, e_ct,
        NUM_SLOTS // D_TOTAL, D_TOTAL, ITERS)

    weights_ci = weights_ct.chain_index()
    phantom.mod_switch_to_inplace(ctx, v_ct, weights_ci)
    sv_mask = score_mask_plaintext(
        ctx, encoder, D_HEAD, D_TOTAL, NUM_TOKENS, weights_ci, SCALE)
    attn_h = phantom.score_times_v(
        ctx, relin_key, galois_key, [weights_ct], [v_ct],
        sv_mask, D_HEAD, D_TOTAL, NUM_TOKENS)

    b0 = np.zeros(NUM_SLOTS, dtype=np.float64)
    b0[:D_TOTAL] = 1.0
    b0_pt = encoder.encode_double_vector(
        ctx, b0.tolist(), SCALE, attn_h.chain_index())
    attn_h = phantom.multiply_plain(ctx, attn_h, b0_pt)
    attn_h = phantom.rescale_to_next(ctx, attn_h)
    attn_h = phantom.replicate(ctx, galois_key, attn_h, D_TOTAL, NUM_SLOTS)
    _rec("attn_C", t0)

    # ---- shift periodic d_total -> IRP-interleaved at d=d_total. ----
    t0 = _t()
    attn_irp = relayout_periodic_to_irp(engine, ctx, encoder, sk, attn_h,
                                          D_TOTAL, D_TOTAL,
                                          user_level=USER_LEVEL_IRP_ATTN)
    _rec("layout_shift", t0)

    # ---- Wo via IRP. ----
    t0 = _t()
    o_ct = irp_matvec_host(ctx, encoder, galois_key, attn_irp, diag_wo_irp,
                      NUM_SLOTS, D_TOTAL, baby_steps=BABY_STEPS_IRP_SQUARE,
                      mask_pt=mask_attn_pt)
    _rec("wo_irp", t0)

    # ---- shift IRP -> periodic d_model (fresh user-segment chain for residual). ----
    t0 = _t()
    o_periodic = relayout_irp_to_periodic(engine, ctx, encoder, sk, o_ct,
                                            D_TOTAL, D_MODEL,
                                            user_level=USER_LEVEL_FRESH)
    _rec("layout_shift", t0)
    return o_periodic


# ============================ MLP forward (IRP + bootstrap) ============================
def fhe_mlp_irp_bootstrap(engine, ctx, encoder, sk, relin_key,
                            galois_key,
                            x_mid_norm,
                            diag_gate_irp, diag_up_irp, diag_down_irp,
                            sub_mask_wide_pt, sub_mask_tall_pt, input_mask_pt,
                            stage_times=None):
    def _t(): return time.perf_counter()
    def _rec(name, t0):
        if stage_times is None: return
        stage_times.setdefault(name, 0.0)
        stage_times[name] += (time.perf_counter() - t0) * 1000.0

    # ---- shift periodic d_model -> IRP-interleaved at d=d_model. ----
    t0 = _t()
    x_irp = relayout_periodic_to_irp(engine, ctx, encoder, sk, x_mid_norm,
                                       D_MODEL, D_MODEL,
                                       user_level=USER_LEVEL_IRP_MLP)
    _rec("layout_shift", t0)

    # ---- gate = Wgate @ x  (rect wide; output in PERMUTED stride-t' layout). ----
    t0 = _t()
    gate_ct = irp_matvec_rect_host(ctx, encoder, galois_key, x_irp, diag_gate_irp,
                                NUM_SLOTS, D_MODEL, D_PAD_MLP,
                                baby_steps=BABY_STEPS_IRP_MLP,
                                sub_mask_pt=sub_mask_wide_pt)
    _rec("mlp_gate", t0)

    # gate_ct exits IRP at scale^2; rescale to SCALE before bootstrap.
    gate_ct = phantom.rescale_to_next(ctx, gate_ct)
    gate_ct.set_scale(SCALE)
    # ---- Refresh gate_ct via homomorphic bootstrap. ----
    t0 = _t()
    gate_ct = boot_centered(engine, ctx, encoder, sk, gate_ct)
    _rec("bootstrap", t0)

    # ---- silu(gate). ----
    t0 = _t()
    silu_gate = silu(ctx, encoder, relin_key, gate_ct)
    _rec("mlp_silu", t0)
    # ---- up = Wup @ x. (Re-use x_irp; same chain.) ----
    t0 = _t()
    up_ct = irp_matvec_rect_host(ctx, encoder, galois_key, x_irp, diag_up_irp,
                              NUM_SLOTS, D_MODEL, D_PAD_MLP,
                              baby_steps=BABY_STEPS_IRP_MLP,
                              sub_mask_pt=sub_mask_wide_pt)
    _rec("mlp_up", t0)

    # up_ct exits IRP at scale^2; rescale to SCALE before bootstrap.
    up_ct = phantom.rescale_to_next(ctx, up_ct)
    up_ct.set_scale(SCALE)
    # ---- Refresh up_ct via homomorphic bootstrap. ----
    t0 = _t()
    up_ct = boot_centered(engine, ctx, encoder, sk, up_ct)
    _rec("bootstrap", t0)

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

    # ---- Refresh h via homomorphic bootstrap (preserves permuted layout). ----
    t0 = _t()
    h_fresh = boot_centered(engine, ctx, encoder, sk, h_ct)
    # Mod-switch to IRP_MLP chain so plaintext chain indices align and GPU
    # memory stays within budget (chain 16 has 30 primes; chain 26 has 5).
    irp_mlp_chain = engine.user_level_chain_index(USER_LEVEL_IRP_MLP)
    h_fresh = phantom.mod_switch_to(ctx, h_fresh, irp_mlp_chain)
    _rec("bootstrap", t0)

    # ---- out = Wdown @ h  (rect tall). ----
    t0 = _t()
    out_ct = irp_matvec_rect_host(ctx, encoder, galois_key, h_fresh, diag_down_irp,
                               NUM_SLOTS, D_PAD_MLP, D_MODEL,
                               baby_steps=BABY_STEPS_IRP_MLP,
                               sub_mask_pt=sub_mask_tall_pt,
                               input_mask_pt=input_mask_pt)
    _rec("mlp_down", t0)

    # ---- shift IRP -> periodic d_model (fresh chain for residual). ----
    t0 = _t()
    out_periodic = relayout_irp_to_periodic(engine, ctx, encoder, sk, out_ct,
                                              D_MODEL, D_MODEL,
                                              user_level=USER_LEVEL_FRESH)
    _rec("layout_shift", t0)
    return out_periodic


# ============================ Driver ============================
def main():
    L = lambda n: np.load(f"{PROBE}/{n}.npy")
    embed   = L("embed");  ref_out = L("ref_out")
    g1, g2  = L("g1"), L("g2")
    Wq, Wk, Wv, Wo    = L("Wq"), L("Wk"), L("Wv"), L("Wo")
    Wgate, Wup, Wdown = L("Wgate"), L("Wup"), L("Wdown")
    cos_all, sin_all  = L("rope_cos"), L("rope_sin")

    xn = rmsnorm_np(embed, g1)
    K = (xn @ Wk.T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD)
    V = (xn @ Wv.T).reshape(NUM_TOKENS, N_KV_HEADS, D_HEAD)
    K = apply_rope_np(K, cos_all, sin_all)
    K_full = np.repeat(K, N_KV_GROUPS, axis=1).reshape(NUM_TOKENS, D_TOTAL)
    V_full = np.repeat(V, N_KV_GROUPS, axis=1).reshape(NUM_TOKENS, D_TOTAL)
    R_P = rope_matrix_np(cos_all[P], sin_all[P])
    Wq_baked = Wq.copy()
    for h in range(N_HEADS):
        s, e = h*D_HEAD, (h+1)*D_HEAD
        Wq_baked[s:e, :] = R_P @ Wq[s:e, :]
    Q_np = (xn[P] @ Wq_baked.T).reshape(N_HEADS, D_HEAD)
    K_full_h = K_full.reshape(NUM_TOKENS, N_HEADS, D_HEAD)
    scores_np = (Q_np[None, :, :] * K_full_h).sum(-1) / math.sqrt(D_HEAD)
    C_per_head = scores_np.max(0) + 0.5

    # ---- Step union (full rotation step inventory). ----
    sdpa_steps = sdpa_required_steps(D_HEAD, D_TOTAL, NUM_TOKENS, NUM_SLOTS)
    rep_steps  = replicate_required_steps(D_TOTAL, max(NUM_SLOTS, 1 << 15))
    rms_steps  = rmsnorm_required_steps(D_MODEL)
    irp_attn_steps = irp_required_steps(NUM_SLOTS, D_TOTAL,
                                          baby_steps=BABY_STEPS_IRP_SQUARE)
    irp_mlp_w_steps = irp_required_steps_rect(NUM_SLOTS, D_MODEL, D_PAD_MLP,
                                                baby_steps=BABY_STEPS_IRP_MLP)
    irp_mlp_t_steps = irp_required_steps_rect(NUM_SLOTS, D_PAD_MLP, D_MODEL,
                                                baby_steps=BABY_STEPS_IRP_MLP)
    user_steps = sorted(set(list(rms_steps) + list(sdpa_steps) + list(rep_steps)
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
    #   sum_reduce {1,2,4,...,2048} fire at chain 16 → target=16
    #   (compute_qkt inner_sum {1,..,64} also fires at 16 → same keys, no extra cost)
    #
    # finalize_softmax sum_reduce {4096,8192,16384} (stage C, after bootstrap B→C):
    #   mask+rescale (1 level) → chain 17; finalize_softmax receives e_ct at chain 17
    #   sum_reduce fires inside finalize_softmax at chain 17 → target=17
    #
    # score_times_v in-block broadcast {-1,-2,-4,-8,-16,-32,-64} (stage C):
    #   finalize_softmax output at chain 17+4=21 (4 Goldschmidt levels) → target=21
    #
    # replicate {-4096,-8192,-16384} (stage C, after score_v mask+rescale):
    #   score_v output chain ~21, mask+rescale(1) → chain 22, replicate fires → target=22
    #   Use 23 conservatively (one extra level for rounding).
    #
    # IRP-only steps (all 44, both attn and MLP):
    #   Input re-encrypted at USER_LEVEL_IRP_ATTN=8 → chain 16+8=24 → target=24
    #   (All IRP variants — preprocess, babies, giants, reduce — fire at this chain.)
    #
    # Memory estimate (size_QP=36, poly=37.7 MB/partition):
    #   12 × beta_k=3 × 37.7 = 1359 MB   (rms/qkt, target=16)
    #    3 × beta_k=3 × 37.7 =  340 MB   (finalize, target=17)
    #    7 × beta_k=2 × 37.7 =  528 MB   (score_v, target=21)
    #    3 × beta_k=2 × 37.7 =  226 MB   (replicate, target=23)
    #   44 × beta_k=2 × 37.7 = 3322 MB   (irp-only, target=24)
    #   Total keys ≈ 5.8 GB + EVK ≈ 19.4 GB → ~25 GB projected.

    FRESHEST_CHAIN = 16    # invariant to NSL for our bootstrap pipeline
    TARGET_RMS          = FRESHEST_CHAIN        # 16: rms + qkt inner_sum
    TARGET_FINALIZE     = FRESHEST_CHAIN + 1    # 17: finalize_softmax sum_reduce
    TARGET_SCORE_V      = FRESHEST_CHAIN + 5    # 21: score_v broadcast (4 Goldschmidt + 1 mask)
    TARGET_REPLICATE    = FRESHEST_CHAIN + 7    # 23: replicate (score_v out + mask+rescale)
    TARGET_IRP          = FRESHEST_CHAIN + USER_LEVEL_IRP_ATTN  # 26: all IRP ops (ul=10)

    rms_set      = set(rms_steps)
    sdpa_set     = set(sdpa_steps)
    rep_set      = set(rep_steps)
    irp_all_set  = set(irp_attn_steps) | set(irp_mlp_w_steps) | set(irp_mlp_t_steps)
    irp_only_set = irp_all_set - rms_set - sdpa_set - rep_set

    # Positive SDPA steps: {1,2,...,64} shared with rms (target=16) + {4096,8192,16384} finalize
    sdpa_finalize_steps = {4096, 8192, 16384}
    # Negative SDPA steps: {-1,...,-64} score_v broadcast
    sdpa_score_v_steps  = {-1,-2,-4,-8,-16,-32,-64}

    target_chain_indices = []
    for s in user_steps:
        if s in rms_set:
            target_chain_indices.append(TARGET_RMS)           # 16
        elif s in sdpa_finalize_steps:
            target_chain_indices.append(TARGET_FINALIZE)      # 17
        elif s in sdpa_score_v_steps:
            target_chain_indices.append(TARGET_SCORE_V)       # 21
        elif s in rep_set:
            target_chain_indices.append(TARGET_REPLICATE)     # 23
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

    # ---- Encode + encrypt initial vectors at fresh user level. ----
    x_slots = np.zeros(NUM_SLOTS); k_slots = np.zeros(NUM_SLOTS); v_slots = np.zeros(NUM_SLOTS)
    for k in range(NUM_SLOTS // D_MODEL):
        x_slots[k*D_MODEL : k*D_MODEL + D_MODEL] = embed[P]
    for tt in range(NUM_TOKENS):
        base = tt*D_TOTAL
        k_slots[base:base+D_TOTAL] = K_full[tt]
        v_slots[base:base+D_TOTAL] = V_full[tt]
    x_ct = sk.encrypt_symmetric(ctx, encoder.encode_double_vector(ctx, x_slots.tolist(), SCALE, fresh_ci))
    k_ct = sk.encrypt_symmetric(ctx, encoder.encode_double_vector(ctx, k_slots.tolist(), SCALE, fresh_ci))
    v_ct = sk.encrypt_symmetric(ctx, encoder.encode_double_vector(ctx, v_slots.tolist(), SCALE, fresh_ci))
    # ---- Pre-encode FHE weights via host IRP. ----
    print("Encoding IRP weights...")
    t_enc0 = time.perf_counter()
    diag_wq_irp = encode_irp_diagonals_host(
        ctx, encoder, Wq_baked.T, NUM_SLOTS, D_TOTAL, SCALE,
        baby_steps=BABY_STEPS_IRP_SQUARE)
    diag_wo_irp = encode_irp_diagonals_host(
        ctx, encoder, Wo.T, NUM_SLOTS, D_TOTAL, SCALE,
        baby_steps=BABY_STEPS_IRP_SQUARE)
    # IRP masks live at the chain at which the IRP runs (user_level_chain_index).
    irp_attn_chain = engine.user_level_chain_index(USER_LEVEL_IRP_ATTN)
    irp_mlp_chain  = engine.user_level_chain_index(USER_LEVEL_IRP_MLP)
    mask_attn_pt = encode_irp_mask(ctx, encoder, NUM_SLOTS, D_TOTAL, SCALE, irp_attn_chain)

    Wgate_pad = np.zeros((D_MODEL, D_PAD_MLP), dtype=np.float64)
    Wgate_pad[:, :D_HIDDEN] = Wgate.T
    Wup_pad = np.zeros((D_MODEL, D_PAD_MLP), dtype=np.float64)
    Wup_pad[:, :D_HIDDEN] = Wup.T
    Wdown_pad = np.zeros((D_PAD_MLP, D_MODEL), dtype=np.float64)
    Wdown_pad[:D_HIDDEN, :] = Wdown.T

    diag_gate_irp = encode_irp_diagonals_rect_host(
        ctx, encoder, Wgate_pad, NUM_SLOTS, D_MODEL, D_PAD_MLP, SCALE,
        baby_steps=BABY_STEPS_IRP_MLP)
    diag_up_irp = encode_irp_diagonals_rect_host(
        ctx, encoder, Wup_pad, NUM_SLOTS, D_MODEL, D_PAD_MLP, SCALE,
        baby_steps=BABY_STEPS_IRP_MLP)
    diag_down_irp = encode_irp_diagonals_rect_host(
        ctx, encoder, Wdown_pad, NUM_SLOTS, D_PAD_MLP, D_MODEL, SCALE,
        baby_steps=BABY_STEPS_IRP_MLP)
    sub_mask_mlp_wide_pt = encode_irp_mask_rect(
        ctx, encoder, NUM_SLOTS, D_MODEL, D_PAD_MLP, SCALE, irp_mlp_chain)
    sub_mask_mlp_tall_pt = encode_irp_mask_rect(
        ctx, encoder, NUM_SLOTS, D_PAD_MLP, D_MODEL, SCALE, irp_mlp_chain + 1)
    input_mask_mlp_pt = encode_irp_mask(ctx, encoder, NUM_SLOTS, D_MODEL,
                                          SCALE, irp_mlp_chain)
    print(f"  IRP encoding done in {time.perf_counter()-t_enc0:.2f}s.")
    def _make_rms_params(zmin, zmax):
        p = phantom.rmsnorm_params()
        p.d_model    = D_MODEL
        p.epsilon    = EPSILON
        p.z_min      = zmin
        p.z_max      = zmax
        p.poly_degree = RMS_POLY_DEG
        return p
    rms1_p = _make_rms_params(RMS1_Z_MIN, RMS1_Z_MAX)
    rms2_p = _make_rms_params(RMS2_Z_MIN, RMS2_Z_MAX)
    rms1_w = setup_rmsnorm_weights(ctx, encoder, rms1_p, g1.tolist())
    rms2_w = setup_rmsnorm_weights(ctx, encoder, rms2_p, g2.tolist())
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
    placement_table = [
        ("rms1",      7,  29.4, True,  None, True),
        # attention block ends with relayout_irp_to_periodic(USER_LEVEL_FRESH)
        # → fresh chain, so output_level = NSL_MAX. Internal bootstraps are
        # accounted for in the runtime (~ 521 ms incl. 2 internal boot calls).
        ("attention", 0, 521.0, True,  NSL_MAX, False),
        ("residual1", 0,   1.0, True,  None, False),
        ("rms2",      7,  27.4, True,  None, True),
        # mlp block also ends with a free decrypt+re-encrypt to fresh; further,
        # it starts with one too (so input_level doesn't matter for correctness
        # — we set requires_fresh_input=False even though tiles touch x_irp).
        ("mlp",       0, 624.1, True,  NSL_MAX, False),
        ("residual2", 0,   1.0, True,  None, False),
    ]
    layers_for_dag = build_layers_from_table(placement_table)
    plan = find_optimal_placement(layers_for_dag, NSL_MAX, T_BOOT_MS)
    print("\n=== Cachemir §6 bootstrap placement (DAG shortest-path) ===")
    print(render_plan_table(plan))
    boot_before = {plan.layers[s.layer_idx].name: s.bootstrap_before for s in plan.steps}

    stage_times = {}
    t_total0 = time.perf_counter()

    stage_times.setdefault("bootstrap", 0.0)

    def _maybe_boot(name, ct):
        """Insert a bootstrap before `name` iff the placement plan says so."""
        if not boot_before.get(name, False):
            return ct
        t0 = time.perf_counter()
        ct = boot_centered(engine, ctx, encoder, sk, ct)
        stage_times["bootstrap"] += (time.perf_counter() - t0) * 1000
        print(f"  [plan] bootstrap before {name}: chain={ct.chain_index()}")
        return ct

    # ---- rms1 ----
    x_ct = _maybe_boot("rms1", x_ct)
    t0 = time.perf_counter()
    x_norm = rmsnorm_forward(ctx, encoder, relin_key, galois_key, x_ct, rms1_w, rms1_p)
    stage_times["rms1"] = (time.perf_counter() - t0) * 1000
    print(f"  rms1 done. chain={x_norm.chain_index()}")

    # ---- attention (IRP Wq -> SDPA with 2 internal bootstraps -> IRP Wo). ----
    x_norm = _maybe_boot("attention", x_norm)
    t0 = time.perf_counter()
    attn_out = fhe_attention_irp_bootstrap(
        engine, ctx, encoder, sk, relin_key, galois_key,
        x_norm, diag_wq_irp, diag_wo_irp, mask_attn_pt,
        k_ct, v_ct, C_per_head, stage_times=stage_times)
    stage_times["attention"] = (time.perf_counter() - t0) * 1000
    print(f"  attention done. chain={attn_out.chain_index()}")

    # ---- residual1 ----
    x_mid_ct = residual(ctx, x_ct, attn_out)
    print(f"  residual1 done. chain={x_mid_ct.chain_index()}")

    # ---- rms2 ----
    x_mid_ct = _maybe_boot("rms2", x_mid_ct)
    t0 = time.perf_counter()
    x_mid_norm = rmsnorm_forward(ctx, encoder, relin_key, galois_key, x_mid_ct, rms2_w, rms2_p)
    stage_times["rms2"] = (time.perf_counter() - t0) * 1000
    print(f"  rms2 done. chain={x_mid_norm.chain_index()}")

    # ---- mlp ----
    x_mid_norm = _maybe_boot("mlp", x_mid_norm)

    # MLP: IRP Wgate / Wup / Wdown via host plaintexts.
    t0 = time.perf_counter()
    mlp_out = fhe_mlp_irp_bootstrap(
        engine, ctx, encoder, sk, relin_key, galois_key,
        x_mid_norm,
        diag_gate_irp, diag_up_irp, diag_down_irp,
        sub_mask_mlp_wide_pt, sub_mask_mlp_tall_pt, input_mask_mlp_pt,
        stage_times=stage_times)
    stage_times["mlp"] = (time.perf_counter() - t0) * 1000
    print(f"  mlp done. chain={mlp_out.chain_index()}")

    y_ct = residual(ctx, x_mid_ct, mlp_out)
    total_ms = (time.perf_counter() - t_total0) * 1000

    msg_chain = ctx.total_parm_size() - 1
    phantom.mod_switch_to_inplace(ctx, y_ct, msg_chain)
    y = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, y_ct)),
                 dtype=np.float64)[:D_MODEL]
    err = y - ref_out[P]
    max_err = float(np.abs(err).max())
    rel_rms = float(np.linalg.norm(err) / np.linalg.norm(ref_out[P]))

    print("\n=== Bootstrap-aware LLaMA-3.1-8B layer-0 (IRP, per-step galois) ===")
    print("Per-stage runtime (ms):")
    main_keys = ["rms1", "attention", "rms2", "mlp", "bootstrap"]
    for k in main_keys:
        if k in stage_times:
            print(f"  {k:30s} {stage_times[k]:8.1f}")
    sub_keys = sorted(k for k in stage_times if k not in main_keys)
    for k in sub_keys:
        print(f"    {k:30s} {stage_times[k]:8.1f}")
    print(f"  {'total':30s} {total_ms:8.1f}")
    print(f"\nAccuracy of full decoder y vs HuggingFace fp32 ref_out[{P}]:")
    print(f"  ‖y_fhe‖     = {np.linalg.norm(y):.4f}")
    print(f"  ‖ref_out‖   = {np.linalg.norm(ref_out[P]):.4f}")
    print(f"  max|err|    = {max_err:.3e}")
    print(f"  rel-RMS     = {rel_rms:.3e}")


if __name__ == "__main__":
    main()
