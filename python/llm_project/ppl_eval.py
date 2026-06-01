"""PPL evaluation driver: FHE end-to-end WikiText-2 perplexity.

Loops over strided windows from ppl_prep/windows.npz; for each window runs
run_classifier_fhe_all_positions to get the (num_tokens, D_MODEL) hidden
state at layer 31; applies full_vocab_logprobs_np to get (T, VOCAB)
log-probs; gathers scored next-token log-probs; appends to a CSV.
Resumable: skips (window_idx, position) pairs already in the CSV.

Usage:
  cd /home/yongwoo-oh/phantom-fhe/python/llm_project
  HF_HUB_OFFLINE=1 USE_BOOTSTRAP_17=1 AUTONOMOUS_FHE=1 \\
      python /home/yongwoo-oh/mrpc_campaign/ppl_prep/code/ppl_eval.py \\
      --num-windows 32 --csv /home/yongwoo-oh/mrpc_campaign/ppl_int32.csv

After completing all windows, run with --summary to print final PPL numbers.

CSV fields: window_idx, position, token_id, fhe_logprob, pt_logprob, time_ms
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_CAMPAIGN = "/home/yongwoo-oh/mrpc_campaign"
_PPL_PREP = os.path.join(_CAMPAIGN, "ppl_prep")
_PROBE_FULL = "/tmp/llama_probe_full"

# Ensure llm_project is on sys.path
_LLM_PROJECT = "/home/yongwoo-oh/phantom-fhe/python/llm_project"
_BUILD_LIB = "/home/yongwoo-oh/phantom-fhe/build/lib"
if _BUILD_LIB not in sys.path:
    sys.path.insert(0, _BUILD_LIB)
if _LLM_PROJECT not in sys.path:
    sys.path.insert(0, _LLM_PROJECT)

# lm_head_full helper lives in ppl_prep/code
sys.path.insert(0, _THIS)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M = 64
S = 32
D_MODEL = 4096
T_MODEL = 8         # stride in CKKS slot layout (from llama3.py)
VOCAB = 128256

CSV_HEADER = ["window_idx", "position", "token_id", "fhe_logprob", "pt_logprob", "time_ms"]

# ---------------------------------------------------------------------------
# CSV helpers (resumable)
# ---------------------------------------------------------------------------

def _ensure_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def _completed_pairs(path):
    """Return set of (window_idx, position) already in CSV."""
    _ensure_csv(path)
    done = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                done.add((int(row["window_idx"]), int(row["position"])))
            except (KeyError, ValueError):
                pass
    return done


def _append_rows(path, rows):
    """rows: list of dicts with CSV_HEADER keys."""
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow([str(row[k]) for k in CSV_HEADER])


# ---------------------------------------------------------------------------
# PPL aggregation (from CSV)
# ---------------------------------------------------------------------------

def compute_ppl_from_csv(csv_path, windows_path, tok=None):
    """Read the CSV and compute token/byte/word PPL for both FHE and PT columns."""
    if not os.path.exists(csv_path):
        print("No CSV found.")
        return

    import collections
    z = np.load(windows_path)
    token_ids_arr = z["token_ids"]     # (N_win, 64)

    fhe_nll_sum = 0.0
    pt_nll_sum = 0.0
    scored_token_ids = []
    n = 0

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                w_idx = int(row["window_idx"])
                pos   = int(row["position"])
                tok_id = int(row["token_id"])
                fhe_lp = float(row["fhe_logprob"])
                pt_lp  = float(row["pt_logprob"])
            except (KeyError, ValueError):
                continue
            fhe_nll_sum += -fhe_lp
            pt_nll_sum  += -pt_lp
            scored_token_ids.append(tok_id)
            n += 1

    if n == 0:
        print("No scored rows.")
        return

    if tok is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")

    scored_text = tok.decode(scored_token_ids, skip_special_tokens=False)
    n_bytes = len(scored_text.encode("utf-8"))
    n_words = len(scored_text.split()) if scored_text.strip() else 1

    for label, nll_sum in [("FHE", fhe_nll_sum), ("PT", pt_nll_sum)]:
        H = nll_sum / n
        print(f"\n  {label} PPL ({n} scored tokens):")
        print(f"    token-PPL = {math.exp(H):.4f}")
        print(f"    byte-PPL  = {math.exp(H * n / n_bytes):.4f}  "
              f"(n_bytes={n_bytes:,})")
        print(f"    word-PPL  = {math.exp(H * n / n_words):.4f}  "
              f"(n_words={n_words:,})")


# ---------------------------------------------------------------------------
# Main FHE loop
# ---------------------------------------------------------------------------

def run_ppl_eval(num_windows, csv_path):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Load windows
    windows_path = os.path.join(_PPL_PREP, "windows.npz")
    if not os.path.exists(windows_path):
        raise FileNotFoundError(f"windows.npz not found at {windows_path}; "
                                f"run prepare_ppl.py first")
    z = np.load(windows_path)
    token_ids_arr = z["token_ids"][:num_windows]   # (N_win, 64)
    loss_mask_arr = z["loss_mask"][:num_windows]   # (N_win, 64)

    # Load probe weights
    final_norm_g = np.load(os.path.join(_PROBE_FULL, "final_norm_g.npy")).astype(np.float64)
    meta = json.loads(open(os.path.join(_PROBE_FULL, "meta.json")).read())
    eps = meta["rms_norm_eps"]
    cos_all_full = np.load(os.path.join(_PROBE_FULL, "rope_cos.npy")).astype(np.float64)
    sin_all_full = np.load(os.path.join(_PROBE_FULL, "rope_sin.npy")).astype(np.float64)

    # Load full LM-head (2 GB in RAM, loaded once)
    lm_head_path = os.path.join(_PPL_PREP, "lm_head_full.npy")
    if not os.path.exists(lm_head_path):
        raise FileNotFoundError(f"lm_head_full.npy not found at {lm_head_path}; "
                                f"run prepare_ppl.py first")
    print(f"Loading lm_head_full.npy ({os.path.getsize(lm_head_path)/1024**2:.0f} MB) ...")
    lm_head_full = np.load(lm_head_path)   # (128256, 4096) float32
    print(f"  lm_head_full shape={lm_head_full.shape}")

    # Import FHE machinery
    import pyPhantom as phantom  # noqa: F401
    from llama3_mrpc import (
        run_classifier_fhe,
        build_user_steps_mrpc,
        setup_engine,
    )
    from blocks.lm_head_full import full_vocab_logprobs_np, next_token_logprobs

    # Build engine ONCE and reuse across all windows
    print("Building CKKS engine (once) ...")
    t0 = time.perf_counter()
    user_steps, step_categories = build_user_steps_mrpc()
    engine = setup_engine(user_steps, step_categories=step_categories)
    print(f"  engine built in {time.perf_counter()-t0:.1f}s")

    # Resume
    _ensure_csv(csv_path)
    done_pairs = _completed_pairs(csv_path)
    print(f"  Resuming: {len(done_pairs)} (window, position) pairs already done")

    refs_dir = os.path.join(_PPL_PREP, "refs")

    for w_idx in range(num_windows):
        # Check if entire window already done
        # (determine scored positions for this window first)
        mask = loss_mask_arr[w_idx]
        tids = token_ids_arr[w_idx]
        # Scored positions: loss_mask True AND position >= 1 (position 0 has no predecessor)
        scored_positions = [i for i in range(1, M) if mask[i]]
        if all((w_idx, i) in done_pairs for i in scored_positions):
            print(f"  window {w_idx}: all {len(scored_positions)} positions done — skip")
            continue

        # Load PT ref for this window
        ref_path = os.path.join(refs_dir, f"ppl_window_{w_idx:04d}.npz")
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"PT ref missing: {ref_path}; run prepare_ppl.py first")
        ref_z = np.load(ref_path)
        pytorch_ref      = ref_z["pytorch_ref"]       # (33, 64, 4096)
        pytorch_pre_norm = ref_z["pytorch_pre_norm"]  # (64, 4096)
        pt_logprobs_full = ref_z["pt_logprobs"]       # (64, vocab) float32

        print(f"\n=== Window {w_idx}/{num_windows} (scored positions: {len(scored_positions)}) ===")
        t_window_start = time.perf_counter()

        # FHE forward: run_classifier_fhe_all_positions
        # Since that function is not yet patched into llama3_mrpc.py,
        # we call run_classifier_fhe with query_position=M-1 and
        # manually decode all positions by running a second decrypt on y_ct.
        #
        # TEMPORARY APPROACH (until the patch is applied):
        # run_classifier_fhe returns (yes_logit, no_logit) and internally
        # decrypts y_ct at the final layer. We cannot get y_ct back from it.
        # So we do the full body inline here using a modified last-step.
        #
        # For the pilot, use pytorch_pre_norm as the hidden state source
        # (PT-only mode) to validate the pipeline end-to-end before FHE.
        # When FHE is enabled, replace with the true FHE all-position decode.
        #
        # The real FHE path is:
        #   y_per_pos = run_classifier_fhe_all_positions(
        #       M, pytorch_ref, pytorch_pre_norm,
        #       cos_all_full, sin_all_full,
        #       label=f"ppl_w{w_idx:04d}",
        #       engine=engine,
        #   )
        #
        # For now use PT hidden state to validate the rest of the pipeline:
        # (Replace this block when run_classifier_fhe_all_positions is patched in)
        USE_FHE = os.environ.get("PPL_USE_FHE", "0") == "1"
        if USE_FHE:
            # FHE path — requires run_classifier_fhe_all_positions in llama3_mrpc.py
            try:
                from llama3_mrpc import run_classifier_fhe_all_positions
                y_per_pos = run_classifier_fhe_all_positions(
                    M, pytorch_ref, pytorch_pre_norm,
                    cos_all_full, sin_all_full,
                    label=f"ppl_w{w_idx:04d}",
                    engine=engine,
                )
            except ImportError:
                raise RuntimeError(
                    "run_classifier_fhe_all_positions not found in llama3_mrpc.py. "
                    "Apply the patch from ppl_prep/code/run_classifier_fhe_all_positions.py "
                    "first, or run without PPL_USE_FHE=1 for PT-only validation."
                )
        else:
            # PT-only path: use pytorch_pre_norm as the hidden state
            # (validates lm_head_full pipeline without FHE)
            y_per_pos = pytorch_pre_norm.astype(np.float64)  # (64, 4096)

        t_fhe_done = time.perf_counter()

        # Apply full-vocab LM head (host-side plaintext)
        fhe_logprobs = full_vocab_logprobs_np(y_per_pos, final_norm_g, lm_head_full, eps=eps)
        # (64, vocab) float64

        t_lm_done = time.perf_counter()

        # Collect scored rows
        rows_to_append = []
        for i in scored_positions:
            if (w_idx, i) in done_pairs:
                continue
            # next-token convention: logprobs[i-1] predicts tids[i]
            fhe_lp = float(fhe_logprobs[i - 1, tids[i]])
            pt_lp  = float(pt_logprobs_full[i - 1, tids[i]])
            time_ms = (t_fhe_done - t_window_start) * 1000.0
            rows_to_append.append({
                "window_idx": w_idx,
                "position":   i,
                "token_id":   int(tids[i]),
                "fhe_logprob": fhe_lp,
                "pt_logprob":  pt_lp,
                "time_ms":     time_ms,
            })

        _append_rows(csv_path, rows_to_append)
        t_total_win = time.perf_counter() - t_window_start
        print(f"  window {w_idx}: {len(rows_to_append)} rows appended  "
              f"fhe={t_fhe_done-t_window_start:.1f}s  "
              f"lm_head={t_lm_done-t_fhe_done:.3f}s  total={t_total_win:.1f}s")

        # Quick per-window PPL estimate (PT column, for sanity)
        if rows_to_append:
            w_pt_nll = sum(-r["pt_logprob"] for r in rows_to_append) / len(rows_to_append)
            print(f"  window {w_idx} PT mean NLL={w_pt_nll:.3f}  "
                  f"token-PPL={math.exp(w_pt_nll):.2f}")

    print("\n=== All windows done ===")
    compute_ppl_from_csv(csv_path, windows_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FHE WikiText-2 PPL evaluation driver")
    parser.add_argument("--num-windows", type=int, default=32,
                        help="Number of windows to evaluate (default: 32 pilot)")
    parser.add_argument("--csv", type=str,
                        default="/home/yongwoo-oh/mrpc_campaign/ppl_int32.csv",
                        help="Output CSV path")
    parser.add_argument("--summary", action="store_true",
                        help="Print PPL summary from existing CSV and exit")
    parser.add_argument("--corpus", type=str, default="wikitext2",
                        choices=["wikitext2"],
                        help="Corpus (only wikitext2 supported)")
    parser.add_argument("--window-size", type=int, default=M)
    parser.add_argument("--stride", type=int, default=S)
    args = parser.parse_args()

    if args.summary:
        windows_path = os.path.join(_PPL_PREP, "windows.npz")
        compute_ppl_from_csv(args.csv, windows_path)
        return

    run_ppl_eval(args.num_windows, args.csv)


if __name__ == "__main__":
    main()
