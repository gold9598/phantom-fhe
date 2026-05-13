"""Pre-encode Wq IRPs per (layer, num_tokens) to disk.

The streaming `run_classifier_fhe` currently does the Wq IRP encode JIT
inside each layer iteration (~5-7s/layer): _build_irp_slots in numpy +
256 calls to encode_single_chain_plaintext. That dominates per-layer
wall time vs the ~3s FHE compute.

This script pre-encodes all (layer, num_tokens) combinations that the
MRPC dev sweep actually hits, and persists them per-layer to disk.
Then the streaming code can load Wq IRPs from disk (~1 s/layer of
sequential read) instead of re-encoding. Combined with the rp_indep
disk cache, per-layer prep work drops below FHE compute time, so a
producer thread can fully shadow it.

Disk layout (mirrors build_disk_cache.py rp_indep convention):
  <out>/wq/nt_<N>/MANIFEST.json
  <out>/wq/nt_<N>/layer_NN/...     (one save_scp_dict_to_disk dir per layer)

Total disk: ~40 distinct MRPC num_tokens * 32 layers * ~128 MB =
~160 GB. Per-iteration peak host RSS: ~3 GB (one layer's weights +
Wq SCPs in pinned memory + transient encoder scratch).

Usage:
  python build_wq_disk_cache.py
  python build_wq_disk_cache.py --start 0 --end 50    # subset of MRPC
  python build_wq_disk_cache.py --out /path/to/cache
"""
import argparse
import ctypes
import gc
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MALLOC_ARENA_MAX", "2")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, os.path.join(_REPO, "build", "lib"))
sys.path.insert(0, _THIS_DIR)

import pyPhantom as phantom  # noqa: E402

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


def _nt_dir(out_root, num_tokens):
    return os.path.join(out_root, "wq", f"nt_{num_tokens}")


def _layer_dir(nt_root, layer_idx):
    return os.path.join(nt_root, f"layer_{layer_idx:02d}")


def has_per_nt_layer_cache(nt_root, num_layers):
    """True iff manifest exists and every layer subdir has a valid SCP dict."""
    manifest = os.path.join(nt_root, "MANIFEST.json")
    if not os.path.exists(manifest):
        return False
    try:
        from blocks.scp_disk_cache import has_cache  # noqa: WPS433
    except Exception:
        return False
    for L in range(num_layers):
        if not has_cache(_layer_dir(nt_root, L)):
            return False
    return True


def load_wq_layer_from_disk(out_root, num_tokens, layer_idx):
    """Inverse of the per-(layer, nt) save. Returns the diag_wq_irp list
    (a list of SingleChainPlaintexts) for this (num_tokens, layer)."""
    from blocks.scp_disk_cache import load_scp_dict_from_disk
    sub = load_scp_dict_from_disk(_layer_dir(_nt_dir(out_root, num_tokens),
                                                layer_idx))
    # sub format: {layer_idx: diag_wq_irp_list}
    return sub[layer_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.path.join(_REPO, "cache"),
        help="Output root (the script writes to <out>/wq/nt_<N>/...). "
             "Default: <repo>/cache")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=408)
    ap.add_argument("--num-decoders", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--probe", default="/tmp/llama_probe_full")
    ap.add_argument(
        "--nt", type=int, action="append", default=None,
        help="Build the Wq cache for a specific num_tokens value "
             "(can be repeated for multiple values). Skips MRPC "
             "enumeration. Use for fixed-nt speed-benchmark setups "
             "(e.g. --nt 512 to match Cachemir's benchmark).")
    args = ap.parse_args()

    from llama3 import load_layer_weights, PROBE_FULL  # noqa: F401
    from llama3_mrpc import build_user_steps_mrpc, setup_engine
    from blocks.scp_disk_cache import save_scp_dict_to_disk
    from llama3 import encode_layer_wq_irp, rope_matrix_np

    # 1. Resolve which num_tokens values to encode for.
    if args.nt:
        nt_list = sorted(set(args.nt))
        print(f"[build_wq] explicit nt list: {nt_list}", flush=True)
    else:
        print(f"[build_wq] enumerating MRPC[{args.start},{args.end}) ...",
              flush=True)
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
        ds = load_dataset("nyu-mll/glue", "mrpc")["validation"]
        PROMPT_FMT = ("Are these two sentences paraphrases of each other?\n"
                      "Sentence 1: {s1}\nSentence 2: {s2}\n"
                      "Answer (Yes or No):")
        nt_set = set()
        for idx in range(args.start, args.end):
            row = ds[idx]
            prompt = PROMPT_FMT.format(s1=row["sentence1"], s2=row["sentence2"])
            nt = len(tok(prompt).input_ids)
            nt_set.add(nt)
        nt_list = sorted(nt_set)
        print(f"[build_wq] {len(nt_list)} distinct num_tokens: "
              f"{nt_list[:10]}{'...' if len(nt_list) > 10 else ''}",
              flush=True)

    # 2. Build engine.
    phantom.set_cuda_device(args.gpu)
    user_steps, step_categories = build_user_steps_mrpc()
    print(f"[build_wq] building engine on GPU {args.gpu}...", flush=True)
    t_eng0 = time.perf_counter()
    engine = setup_engine(user_steps, step_categories=step_categories)
    ctx = engine.context()
    encoder = engine.encoder()
    print(f"[build_wq] engine built in {time.perf_counter() - t_eng0:.1f}s",
          flush=True)

    # 3. RoPE tables.
    cos_all = np.load(f"{args.probe}/rope_cos.npy").astype(np.float64)
    sin_all = np.load(f"{args.probe}/rope_sin.npy").astype(np.float64)

    # 4. Per (num_tokens, layer) encode + save. Each layer's wq IRP is
    # ~128 MB pinned host; we drop after each save so peak host RSS
    # stays at ~3 GB (weights + one layer's wq SCPs + transient scratch).
    out_root = args.out
    t_total0 = time.perf_counter()
    nt_done = 0
    for nt in nt_list:
        nt_root = _nt_dir(out_root, nt)
        if has_per_nt_layer_cache(nt_root, args.num_decoders):
            print(f"[build_wq] nt={nt}: already complete, skip", flush=True)
            nt_done += 1
            continue
        os.makedirs(nt_root, exist_ok=True)
        P_local = nt - 1
        R_P = rope_matrix_np(cos_all[P_local], sin_all[P_local])

        print(f"[build_wq] nt={nt}: encoding {args.num_decoders} layers...",
              flush=True)
        t_nt0 = time.perf_counter()
        for L in range(args.num_decoders):
            layer_dir = _layer_dir(nt_root, L)
            if os.path.isdir(layer_dir) and \
               os.path.exists(os.path.join(layer_dir, "index.json")):
                # Resume support: skip already-done layers.
                continue
            t0 = time.perf_counter()
            w = load_layer_weights(L)
            _Wq_baked, diag_wq_irp = encode_layer_wq_irp(
                ctx, encoder, w, R_P)
            t_enc = time.perf_counter() - t0
            t_s0 = time.perf_counter()
            save_scp_dict_to_disk({L: diag_wq_irp}, layer_dir)
            t_save = time.perf_counter() - t_s0
            print(f"  [nt={nt} L={L:02d}] encode={t_enc:.1f}s "
                  f"save={t_save:.1f}s", flush=True)
            del w, diag_wq_irp, _Wq_baked
            gc.collect()
            _malloc_trim()

        # Per-nt manifest so loaders can validate completeness.
        manifest = {
            "layout": "wq per-layer",
            "num_layers": args.num_decoders,
            "num_tokens": nt,
        }
        with open(os.path.join(nt_root, "MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        nt_done += 1
        print(f"[build_wq] nt={nt} done in "
              f"{time.perf_counter() - t_nt0:.1f}s "
              f"({nt_done}/{len(nt_list)})", flush=True)

    print(f"[build_wq] all done: {nt_done}/{len(nt_list)} num_tokens "
          f"in {time.perf_counter() - t_total0:.1f}s total. "
          f"cache at {os.path.join(out_root, 'wq')}", flush=True)


if __name__ == "__main__":
    main()
