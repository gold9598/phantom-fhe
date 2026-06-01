"""Standalone calib-variance study (pure numpy, no pyPhantom, no CUDA).

Measures how constant compute_layer_calib_n outputs are across 20 MRPC
examples × 32 decoder layers.

Usage:
    python3 /home/yongwoo-oh/mrpc_campaign/calib_variance_study.py
"""

import math
import os
import glob
import re
import numpy as np

# ─── LLaMA-3.1-8B constants (copied from llama3.py; no import) ────────────
LOG_N            = 16
N                = 1 << LOG_N
NUM_SLOTS        = N // 2          # 32768
SCALE            = 2.0 ** 40
SPARSE_HW        = 128
D_MODEL          = 4096
D_HEAD           = 128
N_HEADS          = 32
N_KV_HEADS       = 8
N_KV_GROUPS      = N_HEADS // N_KV_HEADS   # 4
D_TOTAL          = N_HEADS * D_HEAD        # 4096
T_MODEL          = NUM_SLOTS // D_MODEL    # 8
EPSILON          = 1e-5
NUM_SQUARINGS    = 4               # default (NSQ6 env not set)
EXTRA_SCALE      = 0.5
TARGET_MAG       = 0.45
RMS_POLY_DEG     = 4
BOOT_CALIB_MARGIN = 1.5

PROBE_FULL = "/tmp/llama_probe_full"

# ─── Numpy helpers (copied from llama3.py) ────────────────────────────────
def rmsnorm_np(x, g, eps=EPSILON):
    rms = np.sqrt((x**2).mean(-1, keepdims=True) + eps)
    return (x / rms) * g

def rotate_half_np(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)

def apply_rope_np(x_btd, cos_td, sin_td):
    return x_btd * cos_td[:, None, :] + rotate_half_np(x_btd) * sin_td[:, None, :]

def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

# ─── softmax_damping_schedule (copied from blocks/softmax.py) ─────────────
def softmax_damping_schedule(num_squarings, num_tokens, extra_scale, target_mag):
    damps = [1.0] * num_squarings
    if num_squarings == 0:
        return damps
    t = float(num_tokens)
    scale_exp = float(2 ** num_squarings)
    t_factor = t ** (-1.0 / scale_exp)
    f = extra_scale * t_factor
    for i in range(num_squarings):
        f_sq = f * f
        d = target_mag / f_sq
        damps[i] = d
        f = f_sq * d
    return damps

# ─── Cheb coeffs for exp (from engine_setup.py) ───────────────────────────
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

def _real_nt(num_tokens, query_position):
    if query_position is None:
        return num_tokens
    return min(num_tokens, query_position + 1)

def _sim_pre_finsmx_mean(scores_post_C_ht, num_tokens, real_nt=None):
    if real_nt is None:
        real_nt = num_tokens
    t_factor = float(real_nt) ** (-1.0 / float(2 ** NUM_SQUARINGS))
    lead     = EXTRA_SCALE * t_factor
    inv_se   = 1.0 / float(2 ** NUM_SQUARINGS)
    inv_pow  = 1.0
    coeffs   = np.empty(len(_EXP_CHEB_DEG4_R2))
    for i in range(len(_EXP_CHEB_DEG4_R2)):
        coeffs[i] = lead * _EXP_CHEB_DEG4_R2[i] * inv_pow
        inv_pow  *= inv_se

    scores_flat = scores_post_C_ht[:, :real_nt].ravel().astype(np.float64)
    y_pop = np.zeros(len(scores_flat), dtype=np.float64)
    for i in range(len(coeffs) - 1, -1, -1):
        y_pop = y_pop * scores_flat + coeffs[i]

    damps = softmax_damping_schedule(NUM_SQUARINGS, real_nt, EXTRA_SCALE, TARGET_MAG)
    for d in damps:
        y_pop = y_pop * y_pop * d

    y0 = np.zeros(1, dtype=np.float64)
    for i in range(len(coeffs) - 1, -1, -1):
        y0 = y0 * np.zeros(1) + coeffs[i]
    for d in damps:
        y0 = y0 * y0 * d
    v0 = float(y0[0])

    n_populated   = N_HEADS * real_nt
    n_unpopulated = NUM_SLOTS - n_populated
    global_mean   = (y_pop.sum() + n_unpopulated * v0) / NUM_SLOTS
    return float(global_mean)

def compute_layer_calib_n(x_btd, w, cos_all, sin_all, num_tokens, query_position,
                           margin=BOOT_CALIB_MARGIN):
    g1, g2 = w["g1"], w["g2"]
    Wq, Wk, Wv, Wo = w["Wq"], w["Wk"], w["Wv"], w["Wo"]
    Wgate, Wup, Wdown = w["Wgate"], w["Wup"], w["Wdown"]
    P_q     = query_position
    real_nt = _real_nt(num_tokens, P_q)

    z1 = float((x_btd[P_q] ** 2).mean() + EPSILON)

    xn     = rmsnorm_np(x_btd, g1)
    Q_full = (xn @ Wq.T).reshape(num_tokens, N_HEADS, D_HEAD)
    K_full = (xn @ Wk.T).reshape(num_tokens, N_KV_HEADS, D_HEAD)
    V_full = (xn @ Wv.T).reshape(num_tokens, N_KV_HEADS, D_HEAD)
    Q_full = apply_rope_np(Q_full, cos_all, sin_all)
    K_full = apply_rope_np(K_full, cos_all, sin_all)
    K_full = np.repeat(K_full, N_KV_GROUPS, axis=1)
    V_full = np.repeat(V_full, N_KV_GROUPS, axis=1)
    K_full = K_full[:real_nt]
    V_full = V_full[:real_nt]

    q_max   = float(np.abs(Q_full[P_q]).max())
    scores  = np.einsum('hd,thd->ht', Q_full[P_q], K_full) / math.sqrt(D_HEAD)
    c_per_head     = scores.max(-1) + 0.5
    scores_post_C  = scores - c_per_head[:, None]
    scores_max     = float(np.abs(scores_post_C).max())

    weights = np.exp(scores_post_C - scores_post_C.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)

    SOFTMAX_TARGET   = 1.5
    sum_t_exp        = np.exp(scores_post_C).sum(axis=-1)
    expected_max_sum = float(sum_t_exp.max() * 0.45)
    if expected_max_sum > SOFTMAX_TARGET:
        softmax_safety_scale = SOFTMAX_TARGET / expected_max_sum
    else:
        softmax_safety_scale = 1.0

    attn_p    = np.einsum('ht,thd->hd', weights, V_full).reshape(N_HEADS * D_HEAD)
    o_p       = attn_p @ Wo.T
    x_mid_full        = x_btd.copy()
    x_mid_full[P_q]   = x_btd[P_q] + o_p
    z2           = float((x_mid_full[P_q] ** 2).mean() + EPSILON)
    x_mid_max    = float(np.abs(x_mid_full[P_q]).max())
    x_mid_n      = rmsnorm_np(x_mid_full, g2)
    rms2_out_max = float(np.abs(x_mid_n[P_q]).max())

    gate_pre  = x_mid_n[P_q] @ Wgate.T
    gate_max  = float(np.abs(gate_pre).max())
    gate_silu = silu_np(gate_pre)
    up        = x_mid_n[P_q] @ Wup.T
    up_max    = float(np.abs(up).max())
    h         = gate_silu * up
    h_max     = float(np.abs(h).max())
    mlp_out_vec  = h @ Wdown.T
    mlp_out_max  = float(np.abs(mlp_out_vec).max())
    mlp_out_mean = float(mlp_out_vec.mean())
    o_max        = float(np.abs(o_p).max())
    o_mean_val   = float(o_p.mean())

    max_abs = {
        "x_in":              float(np.abs(x_btd[P_q]).max()) * margin,
        "rms1_out":          float(np.abs(xn[P_q]).max()) * margin,
        "x_mid":             x_mid_max * margin,
        "rms2_out":          rms2_out_max * margin,
        "q":                 q_max * margin,
        "scores":            scores_max * margin,
        "gate":              gate_max * margin,
        "up":                up_max * margin,
        "h":                 h_max * margin,
        "o":                 o_max * margin,
        "o_mean":            o_mean_val,
        "mlp_out":           mlp_out_max * margin,
        "mlp_out_mean":      mlp_out_mean,
        "softmax_safety_scale": softmax_safety_scale,
        "pre_finsmx_mean":   _sim_pre_finsmx_mean(
                                 scores_post_C, real_nt, real_nt=real_nt),
    }
    return z1, z2, max_abs


# ─── Main study ────────────────────────────────────────────────────────────
def load_layer_weights(layer_idx):
    ld = f"{PROBE_FULL}/layer_{layer_idx:02d}"
    L  = lambda n: np.load(f"{ld}/{n}.npy").astype(np.float64)
    return {"Wq": L("Wq"), "Wk": L("Wk"), "Wv": L("Wv"), "Wo": L("Wo"),
            "Wgate": L("Wgate"), "Wup": L("Wup"), "Wdown": L("Wdown"),
            "g1": L("g1"), "g2": L("g2")}

def pick_20_examples(csv_path):
    """Read CSV, sort by num_tokens, pick 20 evenly-spaced examples."""
    rows = []
    with open(csv_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
                nt  = int(parts[1])
                rows.append((idx, nt))
            except ValueError:
                continue
    rows.sort(key=lambda r: r[1])
    n = len(rows)
    indices = [int(round(i * (n - 1) / 19)) for i in range(20)]
    selected = [rows[i] for i in indices]
    return selected  # list of (idx, num_tokens)

def load_ptref(idx, num_tokens):
    path = f"/home/yongwoo-oh/mrpc_campaign/ptref/mrpc_ptref_idx{idx}_n{num_tokens}.npz"
    d = np.load(path)
    return d["ref"]   # shape (33, num_tokens, 4096)

def main():
    csv_path = "/home/yongwoo-oh/mrpc_campaign/quant-32bit_auto.csv"
    rope_cos = np.load("/home/yongwoo-oh/mrpc_campaign/llama_probe_full/rope_cos.npy")
    rope_sin = np.load("/home/yongwoo-oh/mrpc_campaign/llama_probe_full/rope_sin.npy")

    examples = pick_20_examples(csv_path)
    print(f"Selected 20 examples (sorted by num_tokens):")
    for i, (idx, nt) in enumerate(examples):
        print(f"  [{i:2d}] idx={idx:3d}  num_tokens={nt}")

    NUM_LAYERS = 32
    # fields we track — z1/z2 separate from max_abs keys
    FIELDS = ["z1", "z2",
              "x_in", "rms1_out", "x_mid", "rms2_out",
              "q", "scores", "gate", "up", "h", "o",
              "o_mean", "mlp_out", "mlp_out_mean",
              "softmax_safety_scale", "pre_finsmx_mean"]

    # results[field][layer][example_i] = value
    results = {f: [[None]*len(examples) for _ in range(NUM_LAYERS)]
               for f in FIELDS}

    print(f"\nLoading weights and running calibration ...")
    for layer_idx in range(NUM_LAYERS):
        print(f"  Layer {layer_idx:2d}: loading weights ...", flush=True)
        w = load_layer_weights(layer_idx)

        for ex_i, (idx, nt) in enumerate(examples):
            ref   = load_ptref(idx, nt)        # (33, nt, 4096)
            x_btd = ref[layer_idx]             # (nt, 4096) — layer input
            cos_t = rope_cos[:nt]
            sin_t = rope_sin[:nt]
            P_q   = nt - 1                     # variable-nt: query at last position

            z1, z2, max_abs = compute_layer_calib_n(
                x_btd, w, cos_t, sin_t, nt, P_q)

            results["z1"][layer_idx][ex_i]  = z1
            results["z2"][layer_idx][ex_i]  = z2
            for k in FIELDS[2:]:
                results[k][layer_idx][ex_i] = max_abs[k]

        print(f"  Layer {layer_idx:2d}: done  (ex0 z1={results['z1'][layer_idx][0]:.4f})",
              flush=True)
        del w  # free memory

    # ─── Analysis ─────────────────────────────────────────────────────────
    # idx=0 is examples[0] IF that example is idx=0 in the CSV.
    # The actual freeze reference is CSV idx=0.  Check which position it holds.
    ex0_csv_idx = [e[0] for e in examples]   # CSV indices of our 20 examples
    # The precomputed_calib is built from CSV idx=0; find its position in our 20
    # (it may or may not be in the 20).  We load it separately for the ±50% check.
    # Load idx=0 ptref directly.
    ref0 = np.load("/home/yongwoo-oh/mrpc_campaign/ptref/mrpc_ptref_idx0_n60.npz")["ref"]
    w0   = {}  # will reload per layer below

    ref_vals = {f: [None]*NUM_LAYERS for f in FIELDS}  # idx=0 values per layer
    print("\nComputing idx=0 reference values per layer ...")
    for layer_idx in range(NUM_LAYERS):
        w0 = load_layer_weights(layer_idx)
        nt0 = 60; P_q0 = nt0 - 1
        x0  = ref0[layer_idx]
        cos0 = rope_cos[:nt0]; sin0 = rope_sin[:nt0]
        z1, z2, max_abs = compute_layer_calib_n(x0, w0, cos0, sin0, nt0, P_q0)
        ref_vals["z1"][layer_idx] = z1
        ref_vals["z2"][layer_idx] = z2
        for k in FIELDS[2:]:
            ref_vals[k][layer_idx] = max_abs[k]
        del w0
    print("  idx=0 reference done.")

    # Convert to numpy arrays: shape (NUM_LAYERS, 20)
    data = {}
    for f in FIELDS:
        data[f] = np.array(results[f], dtype=np.float64)  # (32, 20)

    ref0_arr = {}
    for f in FIELDS:
        ref0_arr[f] = np.array(ref_vals[f], dtype=np.float64)  # (32,)

    # ─── Write report ─────────────────────────────────────────────────────
    out_path = "/home/yongwoo-oh/mrpc_campaign/precomputed_calib_variance.md"
    lines = []

    def A(s=""):
        lines.append(s)

    A("# Precomputed Calib Variance Study")
    A()
    A(f"**Methodology**: 20 MRPC examples (idx/num_tokens pairs evenly spaced over "
      f"the num_tokens distribution, min–max), 32 decoder layers, "
      f"`compute_layer_calib_n` re-implemented standalone (pure numpy). "
      f"The `precomputed_calib` design freezes calibration from idx=0 (num_tokens=60) "
      f"and reuses for all 408 examples.")
    A()
    A(f"**Selected examples** (sorted by num_tokens):")
    A()
    A("| pos | CSV idx | num_tokens |")
    A("|-----|---------|------------|")
    for i, (idx, nt) in enumerate(examples):
        marker = " ← idx=0 (reference)" if idx == 0 else ""
        A(f"| {i:2d} | {idx:3d} | {nt}{marker} |")
    A()

    # ── Section 1: per-field stability summary ──────────────────────────
    A("## 1. Per-field stability summary")
    A()
    A("CV = std/|mean| across examples, averaged across layers. "
      "Max/min ratio = worst single-layer ratio. "
      "'idx=0 within ±50%' = ref value is in [median×(1/1.5), median×1.5] for ALL layers.")
    A()
    A("| Field | Mean (across layer,ex) | CV (avg across layers) | Max/min ratio (worst layer) | idx=0 within ±50% of median |")
    A("|-------|------------------------|------------------------|-----------------------------|-----------------------------|")

    summary_rows = []
    for f in FIELDS:
        arr   = data[f]        # (32, 20)
        ref0v = ref0_arr[f]    # (32,)

        global_mean = float(arr.mean())
        # CV per layer (std/|mean|), then average
        layer_cv = []
        for l in range(NUM_LAYERS):
            row = arr[l]
            m = abs(row.mean())
            if m < 1e-12:
                layer_cv.append(0.0)
            else:
                layer_cv.append(float(row.std() / m))
        cv_avg = float(np.mean(layer_cv))

        # max/min ratio: worst layer
        worst_ratio = 0.0
        for l in range(NUM_LAYERS):
            row = arr[l]
            mn = float(row.min()); mx = float(row.max())
            if abs(mn) < 1e-30:
                ratio = float("inf") if mx != 0 else 1.0
            else:
                ratio = abs(mx / mn)
            if ratio > worst_ratio:
                worst_ratio = ratio

        # idx=0 within ±50% of median for all layers?
        all_ok = True
        for l in range(NUM_LAYERS):
            med = float(np.median(arr[l]))
            r   = ref0v[l]
            if abs(med) < 1e-30:
                ok = (abs(r) < 1e-30)
            else:
                ratio_to_med = abs(r / med)
                ok = (ratio_to_med <= 1.5) and (ratio_to_med >= 1.0/1.5)
            if not ok:
                all_ok = False
                break

        ok_str = "YES" if all_ok else "NO"
        A(f"| {f} | {global_mean:.4g} | {cv_avg*100:.2f}% | {worst_ratio:.3f}x | {ok_str} |")
        summary_rows.append((f, global_mean, cv_avg, worst_ratio, all_ok))

    A()

    # ── Section 2: per-layer detail for top-3 highest-CV fields ─────────
    A("## 2. Per-layer detail — top-3 highest-CV fields")
    A()
    sorted_by_cv = sorted(summary_rows, key=lambda r: r[2], reverse=True)
    top3 = sorted_by_cv[:3]
    layer_subset = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    for fname, _, cv_avg, _, _ in top3:
        arr = data[fname]
        A(f"### Field: `{fname}` (avg CV={cv_avg*100:.2f}%)")
        A()
        header_ex = " | ".join(f"ex{i}" for i in range(0, 20, 4))
        A(f"| Layer | " + " | ".join(f"ex{i}" for i in range(0, 20, 4))
          + " | idx=0 ref |")
        A("|-------|" + "--------|" * 5 + "----------|")
        for l in layer_subset:
            vals = " | ".join(f"{arr[l][i]:.4g}" for i in range(0, 20, 4))
            A(f"| L{l:2d}   | {vals} | {ref0_arr[fname][l]:.4g} |")
        A()

    # ── Section 3: headline conclusion ──────────────────────────────────
    A("## 3. Headline conclusion")
    A()
    constants   = [r[0] for r in summary_rows if r[2] < 0.05]
    stable      = [r[0] for r in summary_rows if 0.05 <= r[2] < 0.15]
    varying     = [r[0] for r in summary_rows if r[2] >= 0.15]

    A(f"**Constants** (CV < 5%): {', '.join(constants) if constants else '(none)'}")
    A()
    A(f"**Stable-but-varying** (5% ≤ CV < 15%): {', '.join(stable) if stable else '(none)'}")
    A()
    A(f"**Genuinely varying** (CV ≥ 15%): {', '.join(varying) if varying else '(none)'}")
    A()

    all_safe = all(r[4] for r in summary_rows)
    if all_safe:
        A("**Verdict**: idx=0 (the actual freeze reference) falls within ±50% of the "
          "population median for **ALL** fields × ALL layers. The 1.5× BOOT_CALIB_MARGIN "
          "is not violated by this example.")
    else:
        bad_fields = [r[0] for r in summary_rows if not r[4]]
        A(f"**Verdict**: idx=0 falls **outside** ±50% of the population median for "
          f"the following fields: {', '.join(bad_fields)}. These are where the 1.5× "
          f"margin may be insufficient.")
    A()

    # ── Section 4: risk assessment ───────────────────────────────────────
    A("## 4. Risk assessment")
    A()
    A("For each field, we compute the 99th percentile of magnitude across all "
      "(layer, example) cells and compare it to `idx=0_value × 1.5` per layer.")
    A()
    A("| Field | p99 (all layers) | worst layer: idx0×1.5 | p99 / (idx0×1.5) | Risk |")
    A("|-------|-----------------|----------------------|------------------|------|")

    risky_fields = []
    for f in FIELDS:
        arr   = data[f]
        ref0v = ref0_arr[f]
        flat  = arr.ravel()
        p99   = float(np.percentile(np.abs(flat), 99))
        # worst layer = layer where p99 / (ref0×1.5) is largest
        worst_ratio_p99 = 0.0
        for l in range(NUM_LAYERS):
            cap = abs(ref0v[l]) * 1.5
            if cap < 1e-30:
                continue
            layer_p99 = float(np.percentile(np.abs(arr[l]), 99))
            r = layer_p99 / cap
            if r > worst_ratio_p99:
                worst_ratio_p99 = r
        risk = "SAFE" if worst_ratio_p99 <= 1.0 else "FRAGILE"
        if worst_ratio_p99 > 1.0:
            risky_fields.append(f)
        # find worst layer idx0×1.5
        worst_cap = max(abs(ref0v[l]) * 1.5 for l in range(NUM_LAYERS))
        A(f"| {f} | {p99:.4g} | {worst_cap:.4g} | {worst_ratio_p99:.3f} | {risk} |")
    A()

    if not risky_fields:
        A("**Overall risk: BENIGN** — the 1.5× margin comfortably absorbs all "
          "observed variance. No field has its 99th-percentile exceed `idx=0 × 1.5` "
          "in any layer.")
    else:
        A(f"**Overall risk: MODERATE/HIGH** — fields where p99 exceeds `idx=0 × 1.5`: "
          f"{', '.join(risky_fields)}. These are bound-fragile under the 1-example design.")
    A()

    # ── Appendix: raw stats ──────────────────────────────────────────────
    A("## Appendix: per-field, per-layer CV")
    A()
    A("Rows = layers (L0–L31), columns = fields. Values are CV% per layer.")
    header_fields = " | ".join(FIELDS)
    A("| Layer | " + header_fields + " |")
    A("|-------|" + "--------|" * len(FIELDS))
    for l in range(NUM_LAYERS):
        cvs = []
        for f in FIELDS:
            row = data[f][l]
            m   = abs(row.mean())
            cv  = (row.std() / m * 100) if m > 1e-12 else 0.0
            cvs.append(f"{cv:.2f}")
        A(f"| L{l:2d} | " + " | ".join(cvs) + " |")
    A()

    report = "\n".join(lines)
    with open(out_path, "w") as fh:
        fh.write(report)
    print(f"\nReport written to {out_path}")

    # ── Console summary ──────────────────────────────────────────────────
    print("\n=== CONSOLE SUMMARY ===")
    sorted_asc = sorted(summary_rows, key=lambda r: r[2])
    print("Top-3 most STABLE (lowest CV):")
    for r in sorted_asc[:3]:
        print(f"  {r[0]:30s}  CV={r[2]*100:.2f}%")
    print("Top-3 most VARYING (highest CV):")
    for r in sorted_asc[-3:]:
        print(f"  {r[0]:30s}  CV={r[2]*100:.2f}%")
    print(f"idx=0 within ±50% of median for ALL fields/layers: {all(r[4] for r in summary_rows)}")
    if risky_fields:
        print(f"FRAGILE fields (p99 > idx0×1.5): {risky_fields}")
    else:
        print("Risk: BENIGN (no field has p99 > idx0×1.5 in any layer)")

if __name__ == "__main__":
    main()
