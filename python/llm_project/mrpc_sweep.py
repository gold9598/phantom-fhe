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

# Resolve build/lib and llm_project paths relative to this file so the script
# runs without modification on any host (5090 dev box, A6000 sweep box, etc.).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS_DIR))  # python/llm_project -> repo
sys.path.insert(0, os.path.join(_REPO, "build", "lib"))

sys.path.insert(0, _THIS_DIR)

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
              run_classifier_fhe, capture_pytorch_ref_cached, P_local_fn,
              rp_indep_cache, engine, rp_indep_disk_root=None,
              fixed_nt=None):
    row = ds[idx]
    PROMPT_FMT = ("Are these two sentences paraphrases of each other?\n"
                  "Sentence 1: {s1}\nSentence 2: {s2}\n"
                  "Answer (Yes or No):")
    prompt = PROMPT_FMT.format(s1=row["sentence1"], s2=row["sentence2"])
    real_token_ids = tok(prompt).input_ids
    real_nt = len(real_token_ids)
    # Fixed-nt mode: pad real tokens with EOS up to fixed_nt; query at
    # the real last-token position (so Yes/No logits aren't read from
    # a padded slot). Matches Cachemir's seq_len=512 bench setup.
    if fixed_nt is not None:
        if real_nt > fixed_nt:
            token_ids = real_token_ids[:fixed_nt]
            real_nt = fixed_nt
        else:
            token_ids = (list(real_token_ids)
                         + [tok.eos_token_id] * (fixed_nt - real_nt))
        num_tokens = fixed_nt
    else:
        token_ids = real_token_ids
        num_tokens = real_nt
    P_local = real_nt - 1

    pytorch_ref, pytorch_pre_norm, yes_pt, no_pt = capture_pytorch_ref_cached(
        idx, token_ids)
    pt_pred = "Yes" if yes_pt > no_pt else "No"

    # DIAGNOSTIC ONLY: opt-in single-layer verbose via env vars. When all
    # three are unset the args are None — byte-identical to the original
    # `debug_layer=None, max_layer=None, min_layer=None`.
    def _envint(name):
        v = os.environ.get(name)
        return int(v) if v is not None and v != "" else None
    _dbg_layer = _envint("PROBE_DEBUG_LAYER")
    _min_layer = _envint("PROBE_MIN_LAYER")
    _max_layer = _envint("PROBE_MAX_LAYER")

    t0 = time.perf_counter()
    yes_logit, no_logit = run_classifier_fhe(
        num_tokens, P_local, pytorch_ref, pytorch_pre_norm,
        cos_all_full, sin_all_full, label=f"mrpc_{idx}",
        debug_layer=_dbg_layer, max_layer=_max_layer, min_layer=_min_layer,
        rp_indep_cache=rp_indep_cache, engine=engine,
        rp_indep_disk_root=rp_indep_disk_root)
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
    ap.add_argument(
        "--fixed-nt", type=int, default=None,
        help="Pad every example's tokens to this nt with EOS so the "
             "FHE pipeline runs at a constant sequence length. Used "
             "to match Cachemir's nt=512 speed benchmark.")
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
    # The filename keys on len(token_ids) so fixed-nt and variable-nt
    # PT-refs don't collide (e.g. idx=5 has n=69 in variable mode and
    # n=512 in fixed-nt-512 mode — two different cache files).
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

    # Build the CKKS engine ONCE for the whole sweep. The rp_indep_cache
    # holds plaintexts bound to this engine's (ctx, encoder) — rebuilding
    # the engine per-example would invalidate cached plaintexts and produce
    # zeroed / wrong outputs.
    from llama3_mrpc import build_user_steps_mrpc, setup_engine
    user_steps, step_categories = build_user_steps_mrpc()
    print(f"Building shared CKKS engine ({len(user_steps)} rotation steps)...",
          flush=True)
    engine = setup_engine(user_steps, step_categories=step_categories)

    # Per-layer R_P-independent IRP cache (Wo, gate+up packed, Wdown).
    #
    # On big-RAM hosts: First example populates rp_indep_cache in-process
    # (~14s/layer × 32 = ~7.5 min one-time cost) holding ~73 GB pinned host,
    # subsequent examples reuse. Requires 128GB+ RAM.
    #
    # On small-RAM hosts (e.g. 5090 dev box, 62 GB): set rp_indep_disk_root
    # to the per-layer cache produced by build_disk_cache.py. The cache
    # is streamed from disk per-layer inside run_classifier_fhe: load
    # layer L's SCPs → encode wq → compute → drop. Peak in-RAM cost
    # stays at ~3-5 GB instead of 73 GB.
    rp_indep_cache = {}
    _disk_root = os.path.join(_REPO, "cache", "rp_indep")
    _disk_manifest = os.path.join(_disk_root, "MANIFEST.json")
    if os.path.exists(_disk_manifest):
        rp_indep_disk_root = _disk_root
        print(f"Streaming rp_indep from disk: {_disk_root}", flush=True)
    else:
        rp_indep_disk_root = None
        print(f"No disk cache at {_disk_root}; rp_indep will be built "
              f"in-process on the first example (requires 100+ GB host RAM)",
              flush=True)

    for k, idx in enumerate(todo):
        try:
            print(f"\n[{k+1}/{len(todo)}] idx={idx}...", flush=True)
            row = _run_one(idx, tok, ds, None, cos_all_full, sin_all_full,
                            run_classifier_fhe, _ptref_cached, None,
                            rp_indep_cache, engine,
                            rp_indep_disk_root=rp_indep_disk_root,
                            fixed_nt=args.fixed_nt)
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
