"""
LLaMA-3.1-8B layer-0 single-decoder FHE forward, real (non-quantized) weights.
Plaintext-shim baseline path (no CKKSEngine, bare CKKS params) with Cachemir
IRP plaintext-encoding for all linear layers (Wq, Wo, Wgate, Wup, Wdown).

Pipeline:
  rms1 -> [shift] -> Wq IRP -> [shift] -> compute_qkt -> mask*scale -> sub(C[h])
       -> ps_exp + damped sq -> mask -> finalize_softmax -> score_times_v
       -> mask + replicate -> [shift] -> Wo IRP -> [shift] -> +x_ct (residual1)
       -> [shift+rms2] -> [shift] -> Wgate IRP (wide) -> silu
                                  -> Wup IRP (wide) -> ct*ct
                                  -> [refresh] -> Wdown IRP (tall) -> [shift]
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
)
from blocks.attention import (
    mask_scale_plaintext, score_mask_plaintext,
    sdpa_required_steps,
)
from blocks.softmax import softmax_damping_schedule
from blocks.silu import silu
from blocks.linear import replicate_required_steps
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
BABY_STEPS_IRP_MLP = 16

# Chain at which IRP rotations execute. Per-level galois keys for IRP steps
# are built only down to this chain, so input ciphertexts to IRP must be
# re-encrypted at chain >= IRP_CHAIN. The IRP weight plaintexts are encoded at
# this chain so multiply_plain matches.
IRP_CHAIN = 25

# Chain at which silu/up are refreshed before silu's PS evaluation. silu (deg 8)
# consumes ~4 levels; entering at SILU_CHAIN gives output at SILU_CHAIN+4.
SILU_CHAIN = 23

EPSILON = 1e-5
P = 3  # query position (last token attends to all 4)

NUM_SQUARINGS = 5
EXTRA_SCALE = 0.5
ITERS = 6
TARGET_MAG = 0.45

RMS_POLY_DEG = 4
RMS1_Z_MIN, RMS1_Z_MAX = 8e-5, 1.3e-4
RMS2_Z_MIN, RMS2_Z_MAX = 1.4e-4, 2.1e-4

# Chain budget: 29 working primes, same as the pre-IRP baseline. Two galois
# key bundles (galois_key at chain 1 + galois_key at chain IRP_CHAIN)
# keep total key memory inside 32 GB.
BITS = [60] + [40] * 29 + [60]

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
# All shims decrypt + re-encrypt at a chosen chain index; this is the same
# refresh mechanism the original (pre-IRP) version of this file used between
# major stages.

def _re_encrypt_slots(ctx, encoder, sk, slots, chain_index, scale=SCALE):
    return sk.encrypt_symmetric(
        ctx, encoder.encode_double_vector(ctx, slots.tolist(), scale, chain_index))

def relayout_periodic_to_irp(ctx, encoder, sk, ct, d_periodic, d_irp,
                              chain_index, scale=SCALE):
    """Decrypt periodic ct (slot[k*d_periodic + i] = v[i] in [0, d_periodic)),
    re-encrypt in IRP layout slot[i*t] = v[i] (i in [0, min(d_periodic, d_irp)),
    t = NUM_SLOTS / d_irp). Re-encryption chain configurable so downstream IRP
    rotations have keys.
    """
    dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                   dtype=np.float64)[:d_periodic]
    t = NUM_SLOTS // d_irp
    slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    n = min(d_periodic, d_irp)
    slots[::t][:n] = dec[:n]
    return _re_encrypt_slots(ctx, encoder, sk, slots, chain_index, scale)

def relayout_irp_to_periodic(ctx, encoder, sk, ct, d_in, d_out, chain_index, scale=SCALE):
    """Decrypt IRP ct (slot[i*t_in] = v[i] for i in [0, d_in)), re-encrypt in
    periodic d_out layout slot[k*d_out + i] = v[i] for k in [0, NUM_SLOTS/d_out).
    """
    dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                   dtype=np.float64)
    t_in = NUM_SLOTS // d_in
    valid = dec[::t_in][:d_in]
    slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    periods = NUM_SLOTS // d_out
    n = min(d_in, d_out)
    for k in range(periods):
        slots[k*d_out : k*d_out + n] = valid[:n]
    return _re_encrypt_slots(ctx, encoder, sk, slots, chain_index, scale)


# ============================ Attention forward (IRP) ============================
def fhe_attention_irp(ctx, encoder, sk, relin_key,
                      galois_key,
                      x_norm,                             # periodic d_model
                      diag_wq_irp, diag_wo_irp,           # IRP square plaintexts
                      mask_attn_pt,                        # IRP mask (chain IRP_CHAIN)
                      k_ct, v_ct, c_per_head,
                      stage_times=None):
    def _t(): return time.perf_counter()
    def _rec(name, t0):
        if stage_times is None: return
        stage_times.setdefault(name, 0.0)
        stage_times[name] += (time.perf_counter() - t0) * 1000.0

    # ---- shift periodic d_model -> IRP-interleaved at d=d_total. ----
    t0 = _t()
    x_irp = relayout_periodic_to_irp(ctx, encoder, sk, x_norm, D_MODEL, D_TOTAL,
                                       chain_index=IRP_CHAIN)
    _rec("layout_shift", t0)

    # ---- Wq via IRP. ----
    t0 = _t()
    q_ct = irp_matvec_host(ctx, encoder, galois_key, x_irp, diag_wq_irp,
                      NUM_SLOTS, D_TOTAL, baby_steps=BABY_STEPS_IRP_SQUARE,
                      mask_pt=mask_attn_pt)
    _rec("wq_irp", t0)

    # ---- shift interleaved -> periodic d_total (fresh chain for SDPA). ----
    t0 = _t()
    q_ct = relayout_irp_to_periodic(ctx, encoder, sk, q_ct, D_TOTAL, D_TOTAL,
                                     chain_index=1)
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

    damps = softmax_damping_schedule(NUM_SQUARINGS, NUM_TOKENS, EXTRA_SCALE, TARGET_MAG)
    e_ct = phantom.ps_exp_init(
        ctx, encoder, relin_key, scores_ct,
        NUM_TOKENS, NUM_SQUARINGS, EXTRA_SCALE)
    phantom.square_iterations_damped_inplace(ctx, encoder, relin_key, e_ct, damps)

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
    _rec("sdpa", t0)

    # ---- shift periodic d_total -> IRP-interleaved at d=d_total. ----
    t0 = _t()
    attn_irp = relayout_periodic_to_irp(ctx, encoder, sk, attn_h, D_TOTAL, D_TOTAL,
                                          chain_index=IRP_CHAIN)
    _rec("layout_shift", t0)

    # ---- Wo via IRP. ----
    t0 = _t()
    o_ct = irp_matvec_host(ctx, encoder, galois_key, attn_irp, diag_wo_irp,
                      NUM_SLOTS, D_TOTAL, baby_steps=BABY_STEPS_IRP_SQUARE,
                      mask_pt=mask_attn_pt)
    _rec("wo_irp", t0)

    # ---- shift IRP -> periodic d_model for residual. ----
    t0 = _t()
    o_periodic = relayout_irp_to_periodic(ctx, encoder, sk, o_ct, D_TOTAL, D_MODEL,
                                            chain_index=1)
    _rec("layout_shift", t0)
    return o_periodic


# ============================ MLP forward (IRP) ============================
def fhe_mlp_irp(ctx, encoder, sk, relin_key,
                 galois_key,
                 x_mid_norm,                                 # periodic d_model
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
    x_irp = relayout_periodic_to_irp(ctx, encoder, sk, x_mid_norm, D_MODEL, D_MODEL,
                                       chain_index=IRP_CHAIN)
    _rec("layout_shift", t0)

    # ---- gate = Wgate @ x  (rect wide; output in PERMUTED stride-t' layout). ----
    t0 = _t()
    gate_ct = irp_matvec_rect_host(ctx, encoder, galois_key, x_irp, diag_gate_irp,
                                NUM_SLOTS, D_MODEL, D_PAD_MLP,
                                baby_steps=BABY_STEPS_IRP_MLP,
                                sub_mask_pt=sub_mask_wide_pt)
    _rec("mlp_gate", t0)

    # ---- Refresh gate_ct via decrypt+re-encrypt at SCALE so silu (ct*ct PS)
    # sees a clean ct.scale()=SCALE. This costs no extra ciphertext level
    # (re-encrypts at IRP_CHAIN+1, the chain right after the IRP mask).
    g_dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, gate_ct)),
                     dtype=np.float64)
    gate_ct = _re_encrypt_slots(ctx, encoder, sk, g_dec, SILU_CHAIN)

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

    # ---- Refresh up_ct similarly. ----
    u_dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, up_ct)),
                     dtype=np.float64)
    up_ct = _re_encrypt_slots(ctx, encoder, sk, u_dec, SILU_CHAIN)

    # ---- h = silu_gate * up. Both operands at scale=SCALE thanks to the
    # output_scale=SCALE rescale in the wide IRPs. After ct*ct -> rescale ->
    # set_scale(SCALE) the result is clean. ----
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

    # ---- Refresh h to fresh chain (preserve permuted layout). ----
    t0 = _t()
    h_dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, h_ct)),
                     dtype=np.float64)
    h_fresh = _re_encrypt_slots(ctx, encoder, sk, h_dec, IRP_CHAIN)
    _rec("layout_shift", t0)

    # ---- out = Wdown @ h  (rect tall). ----
    t0 = _t()
    out_ct = irp_matvec_rect_host(ctx, encoder, galois_key, h_fresh, diag_down_irp,
                               NUM_SLOTS, D_PAD_MLP, D_MODEL,
                               baby_steps=BABY_STEPS_IRP_MLP,
                               sub_mask_pt=sub_mask_tall_pt,
                               input_mask_pt=input_mask_pt)
    _rec("mlp_down", t0)

    # ---- shift IRP -> periodic d_model for residual. ----
    t0 = _t()
    out_periodic = relayout_irp_to_periodic(ctx, encoder, sk, out_ct,
                                              D_MODEL, D_MODEL,
                                              chain_index=1)
    _rec("layout_shift", t0)
    return out_periodic


# ============================ Decoder block ============================
def fhe_decoder(ctx, encoder, sk, relin_key,
                 galois_key,
                 x_ct,
                 diag_wq_irp, diag_wo_irp,
                 diag_gate_irp, diag_up_irp, diag_down_irp,
                 mask_attn_pt,
                 sub_mask_mlp_wide_pt, sub_mask_mlp_tall_pt, input_mask_mlp_pt,
                 rms1_w, rms2_w, rms1_p, rms2_p,
                 k_ct, v_ct, C_per_head,
                 stage_times=None):
    def _t(): return time.perf_counter()
    def _record(name, t0):
        if stage_times is not None:
            stage_times[name] = (time.perf_counter() - t0) * 1000

    # rms1 (uses SDPA bundle: rms steps {1..2048} are full-Q in galois_key).
    t0 = _t()
    x_norm = rmsnorm_forward(ctx, encoder, relin_key, galois_key, x_ct, rms1_w, rms1_p)
    _record("rms1", t0)

    # Refresh.
    xn_dec = encoder.decode_double_vector(ctx, sk.decrypt(ctx, x_norm))
    x_norm_fresh = _re_encrypt_slots(ctx, encoder, sk, np.asarray(xn_dec), 1)

    # Attention (IRP + SDPA).
    t0 = _t()
    attn_out = fhe_attention_irp(
        ctx, encoder, sk, relin_key, galois_key,
        x_norm_fresh, diag_wq_irp, diag_wo_irp, mask_attn_pt,
        k_ct, v_ct, C_per_head,
        stage_times=stage_times)
    _record("attention", t0)

    # residual1.
    x_mid_ct = residual(ctx, x_ct, attn_out)

    # Refresh + relayout to periodic d_model for rms2.
    xm_dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, x_mid_ct)),
                      dtype=np.float64)[:D_MODEL]
    slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    for k in range(NUM_SLOTS // D_MODEL):
        slots[k*D_MODEL : k*D_MODEL + D_MODEL] = xm_dec
    x_mid_fresh = _re_encrypt_slots(ctx, encoder, sk, slots, 1)

    # rms2.
    t0 = _t()
    x_mid_norm = rmsnorm_forward(
        ctx, encoder, relin_key, galois_key, x_mid_fresh, rms2_w, rms2_p)
    _record("rms2", t0)

    # MLP (IRP).
    t0 = _t()
    mlp_out = fhe_mlp_irp(
        ctx, encoder, sk, relin_key, galois_key,
        x_mid_norm,
        diag_gate_irp, diag_up_irp, diag_down_irp,
        sub_mask_mlp_wide_pt, sub_mask_mlp_tall_pt, input_mask_mlp_pt,
        stage_times=stage_times)
    _record("mlp", t0)

    return residual(ctx, x_mid_ct, mlp_out)


# ============================ Driver ============================
def main():
    L = lambda n: np.load(f"{PROBE}/{n}.npy")
    embed   = L("embed");  ref_out = L("ref_out")
    g1, g2  = L("g1"), L("g2")
    Wq, Wk, Wv, Wo    = L("Wq"), L("Wk"), L("Wv"), L("Wo")
    Wgate, Wup, Wdown = L("Wgate"), L("Wup"), L("Wdown")
    cos_all, sin_all  = L("rope_cos"), L("rope_sin")

    # ---- Plaintext shim ----
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

    # ---- FHE setup ----
    sdpa_steps = sdpa_required_steps(D_HEAD, D_TOTAL, NUM_TOKENS, NUM_SLOTS)
    rep_steps = replicate_required_steps(D_TOTAL, max(NUM_SLOTS, 1 << 15))
    irp_attn_steps = irp_required_steps(NUM_SLOTS, D_TOTAL, baby_steps=BABY_STEPS_IRP_SQUARE)
    irp_mlp_steps_w = irp_required_steps_rect(NUM_SLOTS, D_MODEL, D_PAD_MLP,
                                                baby_steps=BABY_STEPS_IRP_MLP)
    irp_mlp_steps_t = irp_required_steps_rect(NUM_SLOTS, D_PAD_MLP, D_MODEL,
                                                baby_steps=BABY_STEPS_IRP_MLP)
    rms_steps = rmsnorm_required_steps(D_MODEL)
    all_steps = sorted(set(list(sdpa_steps) + list(rep_steps)
                           + list(irp_attn_steps) + list(irp_mlp_steps_w)
                           + list(irp_mlp_steps_t) + list(rms_steps)))

    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, BITS))
    params.set_galois_elts(phantom.get_elts_from_steps(all_steps, N))

    ctx = phantom.context(params)
    sk = phantom.secret_key(); sk.generate_sparse(ctx, SPARSE_HW)
    encoder = phantom.ckks_encoder(ctx)
    relin_key = sk.gen_relinkey(ctx)

    # Single galois-key bundle with per-level chain assignments.
    #   - score_v in-block broadcast steps {-1..-64}: chain 25 (deep, small key)
    #   - replicate steps {-4096, -8192}: chain 22
    #   - softmax sum_reduce {4096, 8192, 16384}: chain 15
    #   - SDPA/RMS-shared steps {1..2048}: chain 1 (full Q, used at chain 1+
    #     by compute_qkt and rmsnorm; the IRP path can reuse them too).
    #   - IRP-only steps (babies/giants/preprocess that don't overlap above):
    #     chain IRP_CHAIN. The IRP path executes after a re-encryption at
    #     chain IRP_CHAIN, so these smaller keys suffice.
    irp_steps_set = sorted(set(irp_attn_steps) | set(irp_mlp_steps_w) | set(irp_mlp_steps_t))
    sdpa_rep_rms_steps = sorted(set(list(sdpa_steps) + list(rep_steps) + list(rms_steps)))

    STEP_MIN_CHAIN = {}
    for s in [-1, -2, -4, -8, -16, -32, -64]:
        STEP_MIN_CHAIN[s] = 25
    for s in [-4096, -8192]:
        STEP_MIN_CHAIN[s] = 22
    for s in [4096, 8192, 16384]:
        STEP_MIN_CHAIN[s] = 15
    irp_only = set(irp_steps_set) - set(sdpa_rep_rms_steps)
    for s in irp_only:
        STEP_MIN_CHAIN[s] = IRP_CHAIN
    target_levels = [STEP_MIN_CHAIN.get(s, 1) for s in all_steps]
    galois_key = sk.create_galois_keys_per_level(
        ctx, list(range(len(all_steps))), target_levels)

    # ---- Encode + encrypt ----
    x_slots = np.zeros(NUM_SLOTS); k_slots = np.zeros(NUM_SLOTS); v_slots = np.zeros(NUM_SLOTS)
    for k in range(NUM_SLOTS // D_MODEL):
        x_slots[k*D_MODEL : k*D_MODEL + D_MODEL] = embed[P]
    for tt in range(NUM_TOKENS):
        base = tt*D_TOTAL
        k_slots[base:base+D_TOTAL] = K_full[tt]
        v_slots[base:base+D_TOTAL] = V_full[tt]
    # All inputs at chain 1 (fresh). rmsnorm uses the chain-1 SDPA bundle.
    x_ct = sk.encrypt_symmetric(ctx, encoder.encode_double_vector(ctx, x_slots.tolist(), SCALE, 1))
    k_ct = sk.encrypt_symmetric(ctx, encoder.encode_double_vector(ctx, k_slots.tolist(), SCALE, 1))
    v_ct = sk.encrypt_symmetric(ctx, encoder.encode_double_vector(ctx, v_slots.tolist(), SCALE, 1))

    # ---- Pre-encode FHE weights (IRP) ----
    print("Encoding IRP weights...")
    t_enc0 = time.perf_counter()
    diag_wq_irp = encode_irp_diagonals_host(
        ctx, encoder, Wq_baked.T, NUM_SLOTS, D_TOTAL, SCALE,
        baby_steps=BABY_STEPS_IRP_SQUARE)
    diag_wo_irp = encode_irp_diagonals_host(
        ctx, encoder, Wo.T, NUM_SLOTS, D_TOTAL, SCALE,
        baby_steps=BABY_STEPS_IRP_SQUARE)
    mask_attn_pt = encode_irp_mask(ctx, encoder, NUM_SLOTS, D_TOTAL, SCALE, IRP_CHAIN)

    # MLP rect IRP. Pad d_hidden along the hidden axis.
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
        ctx, encoder, NUM_SLOTS, D_MODEL, D_PAD_MLP, SCALE, IRP_CHAIN)
    sub_mask_mlp_tall_pt = encode_irp_mask_rect(
        ctx, encoder, NUM_SLOTS, D_PAD_MLP, D_MODEL, SCALE, IRP_CHAIN + 1)
    input_mask_mlp_pt = encode_irp_mask(ctx, encoder, NUM_SLOTS, D_MODEL, SCALE, IRP_CHAIN)

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
    print(f"\nLLaMA-3.1-8B layer-0 IRP, prompt='The quick brown fox', query position {P}")
    print(f"FHE config: scale=2^{int(math.log2(SCALE))}  log_q={sum(BITS)}  galois_steps={len(all_steps)}")

    stage_times = {}
    t_total0 = time.perf_counter()
    y_ct = fhe_decoder(ctx, encoder, sk, relin_key,
                        galois_key,
                        x_ct,
                        diag_wq_irp, diag_wo_irp,
                        diag_gate_irp, diag_up_irp, diag_down_irp,
                        mask_attn_pt,
                        sub_mask_mlp_wide_pt, sub_mask_mlp_tall_pt, input_mask_mlp_pt,
                        rms1_w, rms2_w, rms1_p, rms2_p,
                        k_ct, v_ct, C_per_head,
                        stage_times=stage_times)
    total_ms = (time.perf_counter() - t_total0) * 1000

    y = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, y_ct)),
                 dtype=np.float64)[:D_MODEL]
    err = y - ref_out[P]
    max_err = float(np.abs(err).max())
    rel_rms = float(np.linalg.norm(err) / np.linalg.norm(ref_out[P]))

    print(f"\nRuntime (per stage):")
    main_keys = ["rms1", "attention", "rms2", "mlp"]
    for k in main_keys:
        if k in stage_times:
            print(f"  {k:14s} {stage_times[k]:7.1f} ms")
    sub_keys = sorted(k for k in stage_times if k not in main_keys)
    for k in sub_keys:
        print(f"    {k:14s} {stage_times[k]:7.1f} ms")
    print(f"  {'total':14s} {total_ms:7.1f} ms  (includes refresh + final decrypt)")

    print(f"\nAccuracy vs HuggingFace LLaMA-3.1 layer-0 fp32 forward at position {P}:")
    print(f"  ‖y_fhe‖   = {np.linalg.norm(y):.4f}")
    print(f"  ‖ref_out‖ = {np.linalg.norm(ref_out[P]):.4f}")
    print(f"  max|err|  = {max_err:.3e}")
    print(f"  rel-RMS   = {rel_rms:.3e}")


if __name__ == "__main__":
    main()
