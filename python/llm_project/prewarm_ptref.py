"""Standalone PT-ref disk-cache prewarm.

Loads the HF LLaMA-3.1-8B model ONCE, runs PyTorch reference for every
MRPC dev example that doesn't already have an /tmp/mrpc_ptref_idx{i}_n{n}.npz
file, then exits. Subprocess isolation guarantees that PyTorch's caching
allocator (~15 GB retained on GPU 0) is released back to the OS before
the FHE sweep starts — without this, on a 32 GB 5090 GPU the residual
PyTorch memory + the CKKS engine + FHE working set OOMs the GPU.

Usage:
  python prewarm_ptref.py                      # all 408 dev examples
  python prewarm_ptref.py --start 0 --end 100  # partial range
  python prewarm_ptref.py --force              # re-generate even if cached

Output: /tmp/mrpc_ptref_idx{idx}_n{num_tokens}.npz  (one per example)
"""
import argparse
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=408)
    ap.add_argument("--force", action="store_true",
                    help="Re-generate even if .npz already exists.")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument(
        "--fixed-nt", type=int, default=None,
        help="Pad every example's token_ids to this length with EOS "
             "before running the PT-ref forward pass. Use to populate "
             "the PT-ref cache for a fixed-nt MRPC sweep.")
    args = ap.parse_args()

    # Ensure llm_project + build/lib on path so capture_pytorch_ref_with_model
    # imports cleanly.
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.dirname(os.path.dirname(_THIS_DIR))
    sys.path.insert(0, _THIS_DIR)
    sys.path.insert(0, os.path.join(_REPO, "build", "lib"))

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from llama3_mrpc import capture_pytorch_ref_with_model

    PROMPT_FMT = ("Are these two sentences paraphrases of each other?\n"
                  "Sentence 1: {s1}\nSentence 2: {s2}\n"
                  "Answer (Yes or No):")

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("nyu-mll/glue", "mrpc")["validation"]

    # Collect misses without touching the GPU.
    miss_specs = []
    for idx in range(args.start, args.end):
        row = ds[idx]
        prompt = PROMPT_FMT.format(s1=row["sentence1"], s2=row["sentence2"])
        real_ids = tok(prompt).input_ids
        if args.fixed_nt is not None:
            if len(real_ids) > args.fixed_nt:
                token_ids = real_ids[:args.fixed_nt]
            else:
                token_ids = (list(real_ids)
                             + [tok.eos_token_id] *
                             (args.fixed_nt - len(real_ids)))
        else:
            token_ids = real_ids
        n = len(token_ids)
        out_path = f"/tmp/mrpc_ptref_idx{idx}_n{n}.npz"
        if (not args.force) and os.path.exists(out_path):
            continue
        miss_specs.append((idx, token_ids, out_path))

    print(f"=== prewarm_ptref [{args.start},{args.end}): "
          f"{len(miss_specs)} misses, "
          f"{(args.end - args.start) - len(miss_specs)} already cached",
          flush=True)
    if not miss_specs:
        print("All PT-refs already on disk. Nothing to do.")
        return

    # Single HF load, all forwards, exit. The OS reclaims the
    # PyTorch caching allocator (~15 GB GPU) when this process exits.
    print(f"Loading {args.model} (fp16) onto cuda:0...", flush=True)
    t_load0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    t_load = time.perf_counter() - t_load0
    print(f"  HF model loaded in {t_load:.1f}s", flush=True)

    t_forwards0 = time.perf_counter()
    for k, (idx, token_ids, out_path) in enumerate(miss_specs):
        ref, prenorm, yes_pt, no_pt = capture_pytorch_ref_with_model(
            model, tok, list(token_ids))
        np.savez(out_path, ref=ref, prenorm=prenorm,
                 yes=np.float64(yes_pt), no=np.float64(no_pt))
        if (k + 1) % 25 == 0 or k == len(miss_specs) - 1:
            print(f"  [{k+1}/{len(miss_specs)}] idx={idx}  "
                  f"elapsed={time.perf_counter() - t_forwards0:.1f}s",
                  flush=True)
    t_forwards = time.perf_counter() - t_forwards0
    print(f"=== prewarm done: {len(miss_specs)} new refs in "
          f"{t_forwards:.1f}s (model load {t_load:.1f}s separate)",
          flush=True)


if __name__ == "__main__":
    main()
