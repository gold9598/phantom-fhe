"""408-MRPC dev sweep: run FHE pipeline on each example, accumulate
accuracy + F1 against ground-truth labels. Resumable: reads any
existing results CSV and skips completed indices.

Usage:
  python mrpc_sweep.py                     # full 408 sweep
  python mrpc_sweep.py --start 0 --end 50  # partial range
  python mrpc_sweep.py --summary           # compute metrics from current CSV

Output:
  /tmp/mrpc_sweep_results.csv
  fields: idx, num_tokens, label, pt_yes, pt_no, pt_pred,
          fhe_yes, fhe_no, fhe_pred, time_sec
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom  # noqa: F401

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")

CSV_PATH = "/tmp/mrpc_sweep_results.csv"
CSV_HEADER = [
    "idx", "num_tokens", "label", "pt_yes", "pt_no", "pt_pred",
    "fhe_yes", "fhe_no", "fhe_pred", "time_sec",
]


def _ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def _completed_indices():
    _ensure_csv()
    done = set()
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["idx"]))
            except (KeyError, ValueError):
                continue
    return done


def _append_row(row):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([str(row[k]) for k in CSV_HEADER])


def _compute_metrics():
    if not os.path.exists(CSV_PATH):
        print("No results yet.")
        return
    fhe_pred = []
    pt_pred = []
    label = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            label.append(int(row["label"]))
            fhe_pred.append(1 if row["fhe_pred"] == "Yes" else 0)
            pt_pred.append(1 if row["pt_pred"] == "Yes" else 0)
    n = len(label)
    if n == 0:
        print("Empty results.")
        return
    label = np.array(label); fhe_pred = np.array(fhe_pred); pt_pred = np.array(pt_pred)

    # MRPC convention: label=1 means paraphrase (Yes), label=0 means not.
    # Accuracy = correct fraction. F1 on positive class.
    def _ac_f1(pred):
        acc = float((pred == label).mean()) * 100
        tp = int(((pred == 1) & (label == 1)).sum())
        fp = int(((pred == 1) & (label == 0)).sum())
        fn = int(((pred == 0) & (label == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9) * 100
        return acc, f1
    pt_acc, pt_f1 = _ac_f1(pt_pred)
    fhe_acc, fhe_f1 = _ac_f1(fhe_pred)
    agree = float((fhe_pred == pt_pred).mean()) * 100
    print(f"\n=== MRPC sweep summary (n={n}) ===")
    print(f"  PT  : acc={pt_acc:.2f}  F1={pt_f1:.2f}")
    print(f"  FHE : acc={fhe_acc:.2f}  F1={fhe_f1:.2f}")
    print(f"  FHE-vs-PT prediction agreement: {agree:.2f}%")
    print(f"  Cachemir reference: acc=71.32  F1=82.19")
    return fhe_acc, fhe_f1, agree


def _run_one(idx, tok, ds, model_holder, cos_all_full, sin_all_full,
              run_classifier_fhe, capture_pytorch_ref_cached, P_local_fn):
    row = ds[idx]
    PROMPT_FMT = ("Are these two sentences paraphrases of each other?\n"
                  "Sentence 1: {s1}\nSentence 2: {s2}\n"
                  "Answer (Yes or No):")
    prompt = PROMPT_FMT.format(s1=row["sentence1"], s2=row["sentence2"])
    token_ids = tok(prompt).input_ids
    num_tokens = len(token_ids)
    P_local = num_tokens - 1

    pytorch_ref, pytorch_pre_norm, yes_pt, no_pt = capture_pytorch_ref_cached(
        idx, token_ids)
    pt_pred = "Yes" if yes_pt > no_pt else "No"

    t0 = time.perf_counter()
    yes_logit, no_logit = run_classifier_fhe(
        num_tokens, P_local, pytorch_ref, pytorch_pre_norm,
        cos_all_full, sin_all_full, label=f"mrpc_{idx}",
        debug_layer=None, max_layer=None, min_layer=None)
    elapsed = time.perf_counter() - t0
    fhe_pred = "Yes" if yes_logit > no_logit else "No"
    return {
        "idx": idx, "num_tokens": num_tokens, "label": int(row["label"]),
        "pt_yes": f"{yes_pt:.4f}", "pt_no": f"{no_pt:.4f}", "pt_pred": pt_pred,
        "fhe_yes": f"{yes_logit:.4f}", "fhe_no": f"{no_logit:.4f}",
        "fhe_pred": fhe_pred, "time_sec": f"{elapsed:.1f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=408)
    ap.add_argument("--summary", action="store_true",
                    help="Only compute summary metrics from existing CSV.")
    args = ap.parse_args()

    if args.summary:
        _compute_metrics()
        return

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from llama3 import PROBE_FULL
    from llama3_mrpc import run_classifier_fhe, capture_pytorch_ref

    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
    ds = load_dataset("nyu-mll/glue", "mrpc")["validation"]

    cos_all_full = np.load(f"{PROBE_FULL}/rope_cos.npy").astype(np.float64)
    sin_all_full = np.load(f"{PROBE_FULL}/rope_sin.npy").astype(np.float64)

    # Per-idx PT-ref disk cache (mirrors llama3_mrpc._cached_pytorch_ref).
    def _ptref_cached(idx, token_ids):
        path = f"/tmp/mrpc_ptref_idx{idx}_n{len(token_ids)}.npz"
        if os.path.exists(path):
            z = np.load(path)
            return z["ref"], z["prenorm"], float(z["yes"]), float(z["no"])
        ref, prenorm, yes_pt, no_pt = capture_pytorch_ref(token_ids)
        np.savez(path, ref=ref, prenorm=prenorm,
                 yes=np.float64(yes_pt), no=np.float64(no_pt))
        return ref, prenorm, yes_pt, no_pt

    done = _completed_indices()
    todo = [i for i in range(args.start, args.end) if i not in done]
    print(f"=== MRPC sweep [{args.start},{args.end}): {len(todo)} examples remaining "
          f"({len(done & set(range(args.start, args.end)))} already done)", flush=True)

    for k, idx in enumerate(todo):
        try:
            print(f"\n[{k+1}/{len(todo)}] idx={idx}...", flush=True)
            row = _run_one(idx, tok, ds, None, cos_all_full, sin_all_full,
                            run_classifier_fhe, _ptref_cached, None)
            _append_row(row)
            agree_str = " (agree)" if row["fhe_pred"] == row["pt_pred"] else " (DISAGREE)"
            print(f"  idx={idx}: PT={row['pt_pred']} FHE={row['fhe_pred']}"
                  f"{agree_str}  t={row['time_sec']}s", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"  idx={idx} FAILED: {type(e).__name__}: {e}", flush=True)
            # Continue with next idx.

    print()
    _compute_metrics()


if __name__ == "__main__":
    main()
