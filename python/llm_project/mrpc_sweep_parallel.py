"""Parallel multi-GPU 408-MRPC dev sweep.

Single Python process, N CKKSEngine instances (one per GPU), N worker
threads dispatching examples round-robin. Shared rp_indep_cache of
SingleChainPlaintexts in pinned host memory (UVA — readable by all GPUs).

Per-thread CUDA streams (cudaStreamPerThread) + GIL release on hot-path
bindings (commit 21a7dbc) let the worker threads drive CUDA concurrently.

Usage:
  python mrpc_sweep_parallel.py                          # auto-detect GPUs
  python mrpc_sweep_parallel.py --num-gpus 4             # force N GPUs
  python mrpc_sweep_parallel.py --start 0 --end 50       # partial range
  python mrpc_sweep_parallel.py --summary                # metrics only

Output:
  /tmp/mrpc_sweep_results.csv (configurable via --csv-path)
  fields: idx, num_tokens, label, pt_yes, pt_no, pt_pred,
          fhe_yes, fhe_no, fhe_pred, time_sec, gpu_id
"""
import argparse
import csv
import os
import sys
import threading
import time

import numpy as np

# Resolve build/lib and llm_project paths relative to this file so the script
# runs on any host without modification (5090 dev box, A6000/A100 sweep box).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS_DIR))  # python/llm_project -> repo
sys.path.insert(0, os.path.join(_REPO, "build", "lib"))
import pyPhantom as phantom  # noqa: F401

sys.path.insert(0, _THIS_DIR)

# 32 decoder layers in LLaMA-3.1-8B. Mirrors the hardcoded NUM_DECODERS
# in run_classifier_fhe and llama3.main(); not module-exported there.
NUM_DECODERS = 32

CSV_PATH_DEFAULT = "/tmp/mrpc_sweep_results.csv"
CSV_HEADER = [
    "idx", "num_tokens", "label", "pt_yes", "pt_no", "pt_pred",
    "fhe_yes", "fhe_no", "fhe_pred", "time_sec", "gpu_id",
]

# Serializes CSV appends across worker threads. The dict / list lookups in
# the rp_indep_cache are NOT lock-protected: the cache is read-only after
# the build phase, and CPython dict reads + tuple indexing are atomic under
# the GIL. SCP `coeffs` buffers live in pinned host memory and are safe to
# read concurrently from multiple GPUs via UVA.
CSV_LOCK = threading.Lock()

# Opt 1b: process-wide cache of pre-encoded Wq IRPs keyed by num_tokens.
# Wq encoding depends only on R_P = rope_matrix(num_tokens-1) and the
# layer weights. In a 408-MRPC sweep there are ~40 distinct num_tokens
# values, so the same encoded IRPs are reused ~10x on average. Structure:
#   SHARED_WQ_CACHE[num_tokens] -> {layer_idx -> tuple-from-encode_layer_irps}
# SHARED_WQ_CACHE_EVENTS[num_tokens] is a threading.Event() set by the
# first thread that finishes encoding for that num_tokens; other threads
# with the same num_tokens see the Event and wait. Lock guards the
# "check / create Event / publish entry" sequence only — the heavy
# encode call happens outside the lock so different num_tokens still
# parallelize across GPUs.
SHARED_WQ_CACHE = {}
SHARED_WQ_CACHE_EVENTS = {}
SHARED_WQ_CACHE_LOCK = threading.Lock()

# Opt 2: disk-persistent IRP caches. Survives process restarts so a
# rerun doesn't re-encode the ~36 GB rp_indep_cache (~7.5 min from cold)
# or re-encode any num_tokens whose wq IRPs were built in a previous
# run. Disabled via --no-disk-cache; cleared on startup via
# --clear-disk-cache.
DISK_CACHE_ROOT_DEFAULT = "/tmp/phantom_irp_cache"
# Per-run flags set by main() before threads start.
_DISK_CACHE_ROOT = None
_DISK_CACHE_ENABLED = False
# Counters logged at end of each phase. Updated under SHARED_WQ_CACHE_LOCK
# so concurrent workers don't lose updates.
_WQ_DISK_HITS = 0
_WQ_DISK_MISSES = 0
_WQ_RAM_HITS = 0


def _ensure_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def _completed_indices(path):
    _ensure_csv(path)
    done = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["idx"]))
            except (KeyError, ValueError):
                continue
    return done


def _append_row(path, row):
    with CSV_LOCK:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([str(row[k]) for k in CSV_HEADER])


def _compute_metrics(path):
    if not os.path.exists(path):
        print("No results yet.")
        return
    fhe_pred = []
    pt_pred = []
    label = []
    with open(path) as f:
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


def _build_engines(num_gpus, setup_engine_fn, user_steps, step_categories):
    """Build one CKKSEngine per GPU, sequentially. cudaSetDevice before each.

    Construction is serial because Phantom's engine init is heavy and we
    don't want to contend on shared host resources. After all engines are
    built, restore the main thread's CUDA binding to device 0 so any later
    main-thread work (CSV summary, etc) runs on a deterministic device.
    """
    engines = []
    for gpu_id in range(num_gpus):
        phantom.set_cuda_device(gpu_id)
        print(f"  Building engine on GPU {gpu_id}...", flush=True)
        eng = setup_engine_fn(user_steps, step_categories=step_categories)
        engines.append(eng)
        print(f"  engine[{gpu_id}] built on GPU {gpu_id}", flush=True)
    # Leave the main thread bound to device 0.
    phantom.set_cuda_device(0)
    return engines


def _build_shared_rp_indep_cache(engines, num_decoders,
                                  load_layer_weights_fn,
                                  encode_layer_rp_indep_irps_fn):
    """Build the R_P-independent IRP cache once, spreading the encode work
    across N worker threads (one per GPU) in parallel.

    Each thread encodes a round-robin subset of layers on its assigned GPU:
    thread i gets layers [i, i+N, i+2N, ...] where N = len(engines).

    Encoding only one layer per GPU keeps Phantom's per-device
    cudaMallocAsync pool small on each GPU, instead of concentrating
    ~15 GB of pool retention on GPU 0. SCPs end up in pinned host
    memory regardless of which GPU did the NTT, so the resulting cache
    is device-agnostic and shareable by all worker engines.

    When the disk cache is enabled and a valid snapshot exists at
    {DISK_CACHE_ROOT}/rp_indep, the on-disk SCPs are loaded instead
    of re-encoded. After a fresh build, the cache is saved back to
    disk for the next process restart.
    """
    from blocks.scp_disk_cache import (save_scp_dict_to_disk,
                                          load_scp_dict_from_disk, has_cache)
    rp_path = (os.path.join(_DISK_CACHE_ROOT, "rp_indep")
               if _DISK_CACHE_ENABLED else None)
    if rp_path and has_cache(rp_path):
        t0 = time.perf_counter()
        print(f"  loading rp_indep_cache from disk: {rp_path}", flush=True)
        cache = load_scp_dict_from_disk(rp_path)
        # Sanity check: must cover all expected layer indices.
        missing = [L for L in range(num_decoders) if L not in cache]
        if missing:
            print(f"  disk cache incomplete (missing layers {missing[:5]}..); "
                  f"rebuilding from scratch", flush=True)
        else:
            print(f"  rp_indep loaded from disk ({len(cache)} layers) in "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
            return cache

    # Parallel encode: spawn N worker threads, one per GPU.
    cache = {}
    cache_lock = threading.Lock()
    num_engines = len(engines)

    def _worker_encode_layers(gpu_id, engine):
        """Worker thread: encode assigned layers on gpu_id."""
        phantom.set_cuda_device(gpu_id)
        ctx = engine.context()
        encoder = engine.encoder()
        # Assign layers in round-robin: thread gpu_id gets layers [gpu_id, gpu_id+num_engines, ...]
        for L in range(gpu_id, num_decoders, num_engines):
            t0 = time.perf_counter()
            w = load_layer_weights_fn(L)
            result = encode_layer_rp_indep_irps_fn(ctx, encoder, w, pack_gate_up=True)
            elapsed = time.perf_counter() - t0
            # Write to shared dict under lock.
            with cache_lock:
                cache[L] = result
                print(f"  cached layer {L:02d} on gpu{gpu_id}  "
                      f"({elapsed:.1f}s)", flush=True)

    threads = []
    for gpu_id in range(num_engines):
        t = threading.Thread(
            target=_worker_encode_layers,
            name=f"rp_indep-gpu{gpu_id}",
            args=(gpu_id, engines[gpu_id]))
        threads.append(t)
        t.start()

    # Wait for all threads to finish.
    for t in threads:
        t.join()

    # Restore main thread to device 0 for any later main-thread work.
    phantom.set_cuda_device(0)

    if rp_path:
        t_save0 = time.perf_counter()
        print(f"  saving rp_indep_cache to disk: {rp_path}", flush=True)
        save_scp_dict_to_disk(cache, rp_path)
        print(f"  rp_indep saved in {time.perf_counter() - t_save0:.1f}s",
              flush=True)
    return cache


def _wq_disk_path(num_tokens):
    if not _DISK_CACHE_ENABLED:
        return None
    return os.path.join(_DISK_CACHE_ROOT, "wq", f"nt_{num_tokens}")


def _wq_extract_diagwq(wq_entry):
    """Extract only the diag_wq_irp lists (one per layer) from a
    shared_wq_cache[num_tokens] dict. The other tuple slots (Wq_baked,
    diag_wo_irp, diag_gate_irp, diag_up_irp, diag_down_irp) are either
    None or references into the rp_indep_cache and are NOT persisted —
    saving them would duplicate the ~36 GB rp_indep on disk per
    num_tokens (~1 TB across the sweep). On load they are rewired from
    the in-memory rp_indep_cache."""
    # wq_entry shape: {layer_idx: (None, diag_wq_irp, diag_wo_irp,
    #                              diag_gate_irp, diag_up_irp, diag_down_irp)}
    # We persist only {layer_idx: diag_wq_irp}.
    return {L: tup[1] for L, tup in wq_entry.items()}


def _wq_rebuild_full(diagwq_by_layer, rp_indep_cache):
    """Inverse of _wq_extract_diagwq: re-assemble the 6-tuple by
    splicing in rp_indep_cache entries. Mirrors the tuple layout produced
    by encode_layer_irps + the (None,) prefix used in run_classifier_fhe."""
    out = {}
    for L, diag_wq_irp in diagwq_by_layer.items():
        if L not in rp_indep_cache:
            raise RuntimeError(
                f"_wq_rebuild_full: layer {L} missing from rp_indep_cache; "
                f"cannot reconstruct shared_wq_cache entry")
        diag_wo_irp, diag_gate_irp, diag_up_irp, diag_down_irp = rp_indep_cache[L]
        out[L] = (None, diag_wq_irp, diag_wo_irp, diag_gate_irp,
                  diag_up_irp, diag_down_irp)
    return out


def _wq_preload_from_disk(rp_indep_cache):
    """Eagerly load every nt_*/ subdirectory under {DISK_CACHE_ROOT}/wq
    into SHARED_WQ_CACHE before workers start. Each loaded num_tokens
    becomes a HIT in run_classifier_fhe — no thread re-encodes Wq."""
    if not _DISK_CACHE_ENABLED:
        return
    from blocks.scp_disk_cache import load_scp_dict_from_disk, has_cache
    wq_root = os.path.join(_DISK_CACHE_ROOT, "wq")
    if not os.path.isdir(wq_root):
        return
    loaded = 0
    t0 = time.perf_counter()
    for name in sorted(os.listdir(wq_root)):
        if not name.startswith("nt_"):
            continue
        try:
            nt = int(name[len("nt_"):])
        except ValueError:
            continue
        path = os.path.join(wq_root, name)
        if not has_cache(path):
            continue
        try:
            diagwq = load_scp_dict_from_disk(path)
            SHARED_WQ_CACHE[nt] = _wq_rebuild_full(diagwq, rp_indep_cache)
            loaded += 1
        except Exception as e:
            print(f"  wq disk preload failed for {path}: "
                  f"{type(e).__name__}: {e}", flush=True)
    if loaded:
        print(f"  wq disk preload: {loaded} num_tokens loaded in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)


def _wq_persist_if_new(num_tokens):
    """If SHARED_WQ_CACHE[num_tokens] is populated and not yet on disk,
    save it. Called from workers AFTER run_classifier_fhe returns, so
    runtime cost is amortized into the per-example wall time. Only the
    diag_wq_irp portion is persisted — see _wq_extract_diagwq."""
    global _WQ_DISK_HITS, _WQ_DISK_MISSES, _WQ_RAM_HITS
    if not _DISK_CACHE_ENABLED:
        return
    path = _wq_disk_path(num_tokens)
    from blocks.scp_disk_cache import save_scp_dict_to_disk, has_cache
    # has_cache is a cheap file stat; OK to do without the lock since
    # the worst case is two threads both deciding to save (atomic rename
    # makes that safe; the second save just replaces the first).
    if has_cache(path):
        return
    entry = SHARED_WQ_CACHE.get(num_tokens)
    if entry is None:
        return
    try:
        save_scp_dict_to_disk(_wq_extract_diagwq(entry), path)
        with SHARED_WQ_CACHE_LOCK:
            _WQ_DISK_MISSES += 1
        print(f"  wq saved to disk: nt={num_tokens} -> {path}", flush=True)
    except Exception as e:
        print(f"  wq disk save failed for nt={num_tokens}: "
              f"{type(e).__name__}: {e}", flush=True)


def _ptref_prewarm_subprocess_main(miss_specs, gpu_id):
    """Subprocess entry point for the PT-ref pre-warm pass.

    Runs in a child process so the HF model's ~15 GB of PyTorch caching
    allocator state is reclaimed by the OS at exit — the parent process
    (which goes on to build CKKS engines and run FHE) starts with a clean
    GPU 0 instead of competing with retained PyTorch memory.

    `miss_specs` is a list of (idx, token_ids, out_path) tuples already
    prepared by the parent (so the child doesn't need to re-load the
    dataset or re-tokenize).
    """
    # Pin this child to one GPU. The parent passed `gpu_id` separately
    # from CUDA_VISIBLE_DEVICES because we set CVD before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch  # noqa: F401 — needed so device_map="cuda:0" works
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Re-import the capture helper inside the child. We can't pickle
    # closures through spawn, so we rebuild the path locally.
    sys.path.insert(0, _THIS_DIR)
    from llama3_mrpc import capture_pytorch_ref_with_model

    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
    t_load0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Meta-Llama-3.1-8B",
        torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    t_load = time.perf_counter() - t_load0
    print(f"  [ptref-subproc] HF model loaded in {t_load:.1f}s", flush=True)

    for idx, token_ids, path in miss_specs:
        print(f"  [ptref-subproc] miss: idx={idx} num_tokens={len(token_ids)}  "
              f"(running PyTorch)...", flush=True)
        ref, prenorm, yes_pt, no_pt = capture_pytorch_ref_with_model(
            model, tok, list(token_ids))
        np.savez(path, ref=ref, prenorm=prenorm,
                 yes=np.float64(yes_pt), no=np.float64(no_pt))

    t_total = time.perf_counter() - t_load0
    t_forwards = t_total - t_load
    print(f"  [ptref-subproc] pre-warmed {len(miss_specs)} new indices "
          f"in {t_total:.1f}s (model load {t_load:.1f}s + "
          f"forwards {t_forwards:.1f}s)", flush=True)


def _prewarm_ptref_cache(todo, tok, ds, capture_pytorch_ref_fn,
                          capture_pytorch_ref_with_model_fn=None,
                          gpu_id=0):
    """Pre-warm the per-idx PT-reference disk cache in a CHILD process.

    PyTorch's caching allocator retains ~15 GB on GPU 0 after the HF model
    is `del`'d, which on a 40 GB A100 leaves only ~25 GB for the worker
    engine + FHE compute. Running the pre-warm in a subprocess means the
    OS reclaims ALL GPU memory when the child exits, so the parent
    process starts with a clean GPU 0.

    After this pass, every worker call to _ptref_load hits the disk cache
    (just np.load, no GPU).

    `capture_pytorch_ref_fn` / `capture_pytorch_ref_with_model_fn` are
    accepted for backward-compat with the call sites but only the
    `with_model` variant is used inside the child (re-imported there).
    """
    PROMPT_FMT = ("Are these two sentences paraphrases of each other?\n"
                  "Sentence 1: {s1}\nSentence 2: {s2}\n"
                  "Answer (Yes or No):")

    # Determine misses in the parent (cheap: just tokenize + os.path.exists).
    miss_specs = []
    for idx in todo:
        row = ds[idx]
        prompt = PROMPT_FMT.format(s1=row["sentence1"], s2=row["sentence2"])
        token_ids = tok(prompt).input_ids
        path = f"/tmp/mrpc_ptref_idx{idx}_n{len(token_ids)}.npz"
        if not os.path.exists(path):
            miss_specs.append((idx, list(token_ids), path))

    if not miss_specs:
        print(f"  PT-ref cache: all {len(todo)} indices already on disk",
              flush=True)
        return

    # Spawn a child to run the HF forwards. `spawn` (not `fork`) is
    # required because the parent has already imported CUDA via Phantom —
    # fork-after-CUDA is unsafe and will deadlock.
    import multiprocessing as mp
    print(f"  spawning PT-ref subprocess for {len(miss_specs)} misses "
          f"(gpu_id={gpu_id})...", flush=True)
    t0 = time.perf_counter()
    ctx_mp = mp.get_context("spawn")
    proc = ctx_mp.Process(
        target=_ptref_prewarm_subprocess_main,
        args=(miss_specs, gpu_id),
        name="ptref-prewarm")
    proc.start()
    proc.join()
    elapsed = time.perf_counter() - t0
    if proc.exitcode != 0:
        raise RuntimeError(
            f"PT-ref pre-warm subprocess failed with exit code "
            f"{proc.exitcode} after {elapsed:.1f}s")
    print(f"  PT-ref subprocess completed in {elapsed:.1f}s "
          f"(GPU memory fully released)", flush=True)


def _ptref_load(idx, token_ids, capture_pytorch_ref_fn):
    """Load PT ref for a single idx from the disk cache. After _prewarm_ptref_cache
    runs, this is always a cache hit (just np.load — thread-safe; no GPU)."""
    path = f"/tmp/mrpc_ptref_idx{idx}_n{len(token_ids)}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return z["ref"], z["prenorm"], float(z["yes"]), float(z["no"])
    # Fallback (shouldn't normally hit this in parallel mode — would race
    # for cuda:0). Kept for correctness.
    ref, prenorm, yes_pt, no_pt = capture_pytorch_ref_fn(token_ids)
    np.savez(path, ref=ref, prenorm=prenorm,
             yes=np.float64(yes_pt), no=np.float64(no_pt))
    return ref, prenorm, yes_pt, no_pt


def _run_one(idx, gpu_id, tok, ds, cos_all_full, sin_all_full,
              shared_cache, engine, run_classifier_fhe_fn, capture_pytorch_ref_fn,
              shared_layer_weights=None):
    """Process a single MRPC example on the engine bound to gpu_id."""
    row = ds[idx]
    PROMPT_FMT = ("Are these two sentences paraphrases of each other?\n"
                  "Sentence 1: {s1}\nSentence 2: {s2}\n"
                  "Answer (Yes or No):")
    prompt = PROMPT_FMT.format(s1=row["sentence1"], s2=row["sentence2"])
    token_ids = tok(prompt).input_ids
    num_tokens = len(token_ids)
    P_local = num_tokens - 1

    pytorch_ref, pytorch_pre_norm, yes_pt, no_pt = _ptref_load(
        idx, token_ids, capture_pytorch_ref_fn)
    pt_pred = "Yes" if yes_pt > no_pt else "No"

    # wq-cache HIT/MISS log: useful for verifying dedup actually fires
    # across examples with the same num_tokens. Cheap (dict membership).
    _wq_status = "HIT" if num_tokens in SHARED_WQ_CACHE else "MISS"
    print(f"  [gpu{gpu_id}] idx={idx} nt={num_tokens} wq_cache={_wq_status}",
          flush=True)

    t0 = time.perf_counter()
    yes_logit, no_logit = run_classifier_fhe_fn(
        num_tokens, P_local, pytorch_ref, pytorch_pre_norm,
        cos_all_full, sin_all_full, label=f"mrpc_{idx}_gpu{gpu_id}",
        debug_layer=None, max_layer=None, min_layer=None,
        rp_indep_cache=shared_cache, engine=engine,
        shared_wq_cache=SHARED_WQ_CACHE,
        shared_wq_cache_events=SHARED_WQ_CACHE_EVENTS,
        shared_wq_cache_lock=SHARED_WQ_CACHE_LOCK,
        preloaded_weights=shared_layer_weights)
    elapsed = time.perf_counter() - t0
    # Persist the wq entry for this num_tokens if it isn't already on
    # disk. Cheap fast path when the entry exists; ~few-second save on
    # the first encounter per num_tokens.
    _wq_persist_if_new(num_tokens)
    fhe_pred = "Yes" if yes_logit > no_logit else "No"
    return {
        "idx": idx, "num_tokens": num_tokens, "label": int(row["label"]),
        "pt_yes": f"{yes_pt:.4f}", "pt_no": f"{no_pt:.4f}", "pt_pred": pt_pred,
        "fhe_yes": f"{yes_logit:.4f}", "fhe_no": f"{no_logit:.4f}",
        "fhe_pred": fhe_pred, "time_sec": f"{elapsed:.1f}", "gpu_id": gpu_id,
    }


def _worker(gpu_id, engine, chunk, shared_cache, tok, ds,
             cos_all_full, sin_all_full, csv_path,
             run_classifier_fhe_fn, capture_pytorch_ref_fn,
             shared_layer_weights=None):
    """Worker thread: process one chunk of example indices on one GPU.

    cudaSetDevice MUST be called inside the thread — CUDA device binding
    is per-thread, not per-process.
    """
    phantom.set_cuda_device(gpu_id)
    for idx in chunk:
        try:
            row = _run_one(idx, gpu_id, tok, ds, cos_all_full, sin_all_full,
                            shared_cache, engine,
                            run_classifier_fhe_fn, capture_pytorch_ref_fn,
                            shared_layer_weights=shared_layer_weights)
            _append_row(csv_path, row)
            agree = " (agree)" if row["fhe_pred"] == row["pt_pred"] else " (DISAGREE)"
            print(f"  [gpu{gpu_id}] idx={idx}: PT={row['pt_pred']} "
                  f"FHE={row['fhe_pred']}{agree}  t={row['time_sec']}s",
                  flush=True)
        except Exception as e:
            print(f"  [gpu{gpu_id}] idx={idx} FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)
            # Continue with the next idx in this thread's chunk.


def main():
    global _DISK_CACHE_ROOT, _DISK_CACHE_ENABLED
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=408)
    ap.add_argument("--summary", action="store_true",
                    help="Only compute summary metrics from existing CSV.")
    ap.add_argument("--num-gpus", type=int, default=0,
                    help="Number of GPUs to use (0 = auto-detect via "
                         "phantom.get_cuda_device_count()).")
    ap.add_argument("--csv-path", default=CSV_PATH_DEFAULT)
    ap.add_argument("--disk-cache-root", default=DISK_CACHE_ROOT_DEFAULT,
                    help="Directory where rp_indep + wq IRP caches are "
                         "persisted across process restarts.")
    ap.add_argument("--no-disk-cache", action="store_true",
                    help="Skip the disk-persistent IRP cache (debug/testing).")
    ap.add_argument("--clear-disk-cache", action="store_true",
                    help="Wipe --disk-cache-root before starting.")
    args = ap.parse_args()

    if args.summary:
        _compute_metrics(args.csv_path)
        return

    _DISK_CACHE_ROOT = args.disk_cache_root
    _DISK_CACHE_ENABLED = not args.no_disk_cache
    if args.clear_disk_cache and os.path.isdir(_DISK_CACHE_ROOT):
        import shutil
        print(f"  --clear-disk-cache: removing {_DISK_CACHE_ROOT}", flush=True)
        shutil.rmtree(_DISK_CACHE_ROOT)
    if _DISK_CACHE_ENABLED:
        os.makedirs(_DISK_CACHE_ROOT, exist_ok=True)
        print(f"  disk cache: {_DISK_CACHE_ROOT}", flush=True)
    else:
        print(f"  disk cache: DISABLED", flush=True)

    num_gpus = args.num_gpus or phantom.get_cuda_device_count()
    if num_gpus <= 0:
        raise RuntimeError("No CUDA devices visible — cannot run sweep.")
    print(f"=== Parallel MRPC sweep with {num_gpus} GPU(s) ===", flush=True)

    # Heavy imports are kept inside main() so `python -c "import
    # mrpc_sweep_parallel"` is cheap (mirrors mrpc_sweep.py).
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from llama3 import (PROBE_FULL, load_layer_weights,
                          load_layer_weights_subset,
                          encode_layer_rp_indep_irps)
    from llama3_mrpc import (run_classifier_fhe, capture_pytorch_ref,
                              capture_pytorch_ref_with_model,
                              build_user_steps_mrpc, setup_engine)

    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
    ds = load_dataset("nyu-mll/glue", "mrpc")["validation"]

    cos_all_full = np.load(f"{PROBE_FULL}/rope_cos.npy").astype(np.float64)
    sin_all_full = np.load(f"{PROBE_FULL}/rope_sin.npy").astype(np.float64)

    # Resume: which indices are already in the CSV?
    done = _completed_indices(args.csv_path)
    todo = [i for i in range(args.start, args.end) if i not in done]
    print(f"=== MRPC sweep [{args.start},{args.end}): {len(todo)} remaining "
          f"({len(done & set(range(args.start, args.end)))} already done)",
          flush=True)
    if not todo:
        print("Nothing to do.")
        _compute_metrics(args.csv_path)
        return

    # ---- Phase 1: build CKKS engines (sequential, one per GPU) ----
    user_steps, step_categories = build_user_steps_mrpc()
    print(f"Building {num_gpus} CKKS engines "
          f"({len(user_steps)} rotation steps each)...", flush=True)
    t_eng0 = time.perf_counter()
    engines = _build_engines(num_gpus, setup_engine, user_steps, step_categories)
    print(f"  all engines built in {time.perf_counter() - t_eng0:.1f}s",
          flush=True)

    # ---- Phase 2: shared rp_indep cache (one-time, ~7.5 min on first run) ----
    # With --disk-cache (default), this is loaded from disk after the
    # first run completes; subsequent process restarts skip the encode.
    print(f"Building shared rp_indep_cache ({NUM_DECODERS} layers)...",
          flush=True)
    t_cache0 = time.perf_counter()
    shared_cache = _build_shared_rp_indep_cache(
        engines, NUM_DECODERS, load_layer_weights, encode_layer_rp_indep_irps)
    print(f"  cache built in {time.perf_counter() - t_cache0:.1f}s", flush=True)

    # ---- Phase 2.5: pre-load per-example layer weights ONCE on main thread ----
    # py-spy showed all 4 workers concurrently stuck in load_layer_weights at
    # llama3.py:150-152 — 9× np.load + .astype(float64), ~128 MB allocations
    # per matrix, fighting on a single disk and the glibc malloc/mmap lock.
    # That serialized the workers despite no GIL issue. Of the 9 weights,
    # only 5 (Wq/Wk/Wv/g1/g2) are touched on the per-example hot path; the
    # other 4 (Wo/Wgate/Wup/Wdown) are R_P-independent and are now served
    # from the disk-persistent rp_indep_cache. Pre-loading just the subset
    # once costs ~5s instead of ~30s × per-example wall-clock contention.
    print("Pre-loading per-example layer weights (Wq/Wk/Wv/g1/g2)...",
          flush=True)
    t_pw0 = time.perf_counter()
    shared_layer_weights = {
        L: load_layer_weights_subset(L) for L in range(NUM_DECODERS)
    }
    print(f"  weights loaded in {time.perf_counter() - t_pw0:.1f}s",
          flush=True)

    # ---- Phase 2b: preload wq IRP entries from disk into SHARED_WQ_CACHE ----
    # Each loaded num_tokens becomes a HIT in run_classifier_fhe so no
    # worker re-encodes Wq for it. Misses (new num_tokens) are encoded
    # at runtime and saved back to disk by _wq_persist_if_new.
    if _DISK_CACHE_ENABLED:
        print(f"Preloading wq disk cache...", flush=True)
        _wq_preload_from_disk(shared_cache)

    # ---- Phase 3: pre-warm PT-ref disk cache on main thread (cuda:0) ----
    # Avoids N threads racing on cuda:0 with concurrent 8B model loads.
    # HF model loaded once for all misses, deleted before workers start.
    # After this pass every worker call is np.load from disk (thread-safe).
    print(f"Pre-warming PT-ref disk cache...", flush=True)
    _prewarm_ptref_cache(todo, tok, ds, capture_pytorch_ref,
                         capture_pytorch_ref_with_model_fn=capture_pytorch_ref_with_model)

    # ---- Phase 4: round-robin dispatch across worker threads ----
    chunks = [todo[i::num_gpus] for i in range(num_gpus)]
    print(f"Dispatching {len(todo)} examples across {num_gpus} GPUs "
          f"(chunks: {[len(c) for c in chunks]})", flush=True)

    threads = []
    for gpu_id in range(num_gpus):
        if not chunks[gpu_id]:
            continue  # nothing for this GPU; skip spawning a thread
        t = threading.Thread(
            target=_worker, name=f"worker-gpu{gpu_id}", args=(
                gpu_id, engines[gpu_id], chunks[gpu_id], shared_cache,
                tok, ds, cos_all_full, sin_all_full, args.csv_path,
                run_classifier_fhe, capture_pytorch_ref,
                shared_layer_weights))
        threads.append(t)
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user; waiting for in-flight examples to "
              "finish their current row write...", flush=True)
        for t in threads:
            t.join()

    print()
    print(f"=== Cache summary ===")
    print(f"  wq num_tokens in RAM at exit: {len(SHARED_WQ_CACHE)}")
    print(f"  wq disk misses encoded this run: {_WQ_DISK_MISSES}")
    _compute_metrics(args.csv_path)


if __name__ == "__main__":
    main()
