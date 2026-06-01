"""PPL preparation script: deliverables 1-3 + PT smoke test.

Outputs (all in /home/yongwoo-oh/mrpc_campaign/ppl_prep/):
  windows.npz          - (256,64) token windows + metadata
  corpus_stats.json    - byte/word/token counts
  lm_head_full.npy     - (128256, 4096) fp32 lm_head matrix
  refs/ppl_window_NNNN.npz - per-window PT reference captures (256 files)
  pt_ppl_results.json  - PT-only PPL smoke test results

Usage:
  python prepare_ppl.py [--num-windows 256] [--skip-pt-capture]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M = 64          # window size
S = 32          # stride
MAX_WINDOWS = 256
D_MODEL = 4096
VOCAB = 128256
PROBE_FULL = "/home/yongwoo-oh/mrpc_campaign/llama_probe_full"
PPL_PREP = "/home/yongwoo-oh/mrpc_campaign/ppl_prep"
MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B"


# ---------------------------------------------------------------------------
# Deliverable 1: Dataset + windows
# ---------------------------------------------------------------------------
def build_windows(num_windows=MAX_WINDOWS):
    print("=== Deliverable 1: Dataset + windows ===")
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("  Loading wikitext-2-raw-v1 test split ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    corpus_text = "\n".join(row["text"] for row in ds)
    n_bytes = len(corpus_text.encode("utf-8"))
    # whitespace words: split on any whitespace, ignore empty tokens
    words = corpus_text.split()
    n_words = len(words)
    print(f"  corpus bytes={n_bytes:,}  words={n_words:,}")

    print("  Loading LLaMA-3 BPE tokenizer ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    all_ids = tok(corpus_text, add_special_tokens=False).input_ids
    all_ids = [int(x) for x in all_ids]
    n_tokens = len(all_ids)
    print(f"  tokens={n_tokens:,}  (tokens/byte={n_tokens/n_bytes:.4f}  tokens/word={n_tokens/n_words:.4f})")

    # Build strided windows: ceil((N-M)/S)+1 total, cap at num_windows
    n_total_windows = math.ceil((n_tokens - M) / S) + 1
    print(f"  total strided windows (uncapped): {n_total_windows}")
    n_windows = min(n_total_windows, num_windows)
    print(f"  building {n_windows} windows (M={M}, S={S})")

    token_ids_arr = np.zeros((n_windows, M), dtype=np.int64)
    loss_mask_arr = np.zeros((n_windows, M), dtype=bool)
    byte_offsets_arr = np.zeros((n_windows, 2), dtype=np.int64)
    word_offsets_arr = np.zeros((n_windows, 2), dtype=np.int64)

    # Pre-compute per-token byte/word offsets using the tokenizer's decode
    # We decode each token individually to get its text then accumulate byte offsets.
    # This is slow for 279k tokens; use a vectorized approach instead:
    # decode subsequences at window boundaries.
    for w in range(n_windows):
        start_tok = w * S
        end_tok = start_tok + M
        token_ids_arr[w] = all_ids[start_tok:end_tok]
        # loss mask: window 0 scores all M positions; others score last S positions
        if w == 0:
            loss_mask_arr[w, :] = True
        else:
            loss_mask_arr[w, M - S:] = True
        # byte offsets: decode the tokens up to start and up to end
        text_before = tok.decode(all_ids[:start_tok], skip_special_tokens=False)
        text_window = tok.decode(all_ids[start_tok:end_tok], skip_special_tokens=False)
        byte_start = len(text_before.encode("utf-8"))
        byte_end = byte_start + len(text_window.encode("utf-8"))
        byte_offsets_arr[w] = [byte_start, byte_end]
        # word offsets (whitespace split count up to start)
        words_before = len(text_before.split()) if text_before.strip() else 0
        words_window = len(text_window.split()) if text_window.strip() else 0
        word_offsets_arr[w] = [words_before, words_before + words_window]
        if (w + 1) % 32 == 0:
            print(f"    window {w+1}/{n_windows} done")

    out_path = os.path.join(PPL_PREP, "windows.npz")
    np.savez(out_path,
             token_ids=token_ids_arr,
             loss_mask=loss_mask_arr,
             byte_offsets=byte_offsets_arr,
             word_offsets=word_offsets_arr)
    sz = os.path.getsize(out_path)
    print(f"  saved {out_path}  ({sz/1024:.1f} KB)")

    stats = {"n_bytes": n_bytes, "n_words": n_words, "n_tokens": n_tokens,
             "n_windows": n_windows, "M": M, "S": S}
    stats_path = os.path.join(PPL_PREP, "corpus_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  saved {stats_path}")

    return token_ids_arr, loss_mask_arr, byte_offsets_arr, word_offsets_arr, stats, tok


# ---------------------------------------------------------------------------
# Deliverable 2: Full LM-head matrix
# ---------------------------------------------------------------------------
def ensure_lm_head_full(model=None):
    """Check for lm_head_full.npy; dump from model if missing."""
    print("=== Deliverable 2: Full LM-head matrix ===")
    out_path = os.path.join(PPL_PREP, "lm_head_full.npy")

    # Check if already saved
    if os.path.exists(out_path):
        sz = os.path.getsize(out_path)
        print(f"  found {out_path}  ({sz/1024**2:.0f} MB) — skipping re-dump")
        lm_head = np.load(out_path)
        print(f"  shape={lm_head.shape}  dtype={lm_head.dtype}")
        return lm_head

    # Check PROBE_FULL for lm_head.npy (from extract_llama_probe.py)
    probe_path = os.path.join(PROBE_FULL, "lm_head.npy")
    if os.path.exists(probe_path):
        print(f"  found {probe_path} — copying to {out_path}")
        lm_head = np.load(probe_path).astype(np.float32)
        np.save(out_path, lm_head)
        sz = os.path.getsize(out_path)
        print(f"  shape={lm_head.shape}  dtype={lm_head.dtype}  size={sz/1024**2:.0f} MB")
        return lm_head

    # Also check /tmp/llama_probe_full/lm_head.npy
    tmp_probe = "/tmp/llama_probe_full/lm_head.npy"
    if os.path.exists(tmp_probe):
        print(f"  found {tmp_probe} — copying to {out_path}")
        lm_head = np.load(tmp_probe).astype(np.float32)
        np.save(out_path, lm_head)
        sz = os.path.getsize(out_path)
        print(f"  shape={lm_head.shape}  dtype={lm_head.dtype}  size={sz/1024**2:.0f} MB")
        return lm_head

    # Must dump from model
    print("  lm_head.npy not found in probe dirs; extracting from PT model ...")
    if model is None:
        import torch
        from transformers import AutoModelForCausalLM
        t0 = time.perf_counter()
        model_local = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0")
        model_local.eval()
        print(f"  model loaded in {time.perf_counter()-t0:.1f}s")
        lm_head = model_local.lm_head.weight.detach().cpu().to(torch.float32).numpy()
        del model_local
        import torch as _torch
        _torch.cuda.empty_cache()
    else:
        import torch
        lm_head = model.lm_head.weight.detach().cpu().to(torch.float32).numpy()

    np.save(out_path, lm_head)
    sz = os.path.getsize(out_path)
    print(f"  shape={lm_head.shape}  dtype={lm_head.dtype}  size={sz/1024**2:.0f} MB")

    # Also save to PROBE_FULL for next time
    probe_save = os.path.join(PROBE_FULL, "lm_head.npy")
    if not os.path.exists(probe_save):
        np.save(probe_save, lm_head)
        print(f"  also saved to {probe_save}")

    return lm_head


# ---------------------------------------------------------------------------
# Deliverable 3: PyTorch reference capture
# ---------------------------------------------------------------------------
def capture_all_windows(token_ids_arr, num_windows):
    """Load PT model once, capture all windows, free model."""
    print("=== Deliverable 3: PyTorch reference capture ===")
    refs_dir = os.path.join(PPL_PREP, "refs")
    os.makedirs(refs_dir, exist_ok=True)

    # Check which windows already captured
    done = set()
    for i in range(num_windows):
        p = os.path.join(refs_dir, f"ppl_window_{i:04d}.npz")
        if os.path.exists(p):
            done.add(i)
    if len(done) == num_windows:
        print(f"  all {num_windows} refs already captured — skipping PT model load")
        return

    import torch
    from transformers import AutoModelForCausalLM

    # Check GPU memory
    free_gb = (torch.cuda.get_device_properties(0).total_memory
               - torch.cuda.memory_allocated(0)) / 1024**3
    print(f"  GPU free: {free_gb:.1f} GB (need ~16 GB for fp16 model)")
    if free_gb < 14:
        raise RuntimeError(f"Insufficient GPU memory: {free_gb:.1f} GB free, need ~16 GB")

    print(f"  Loading PT model (fp16, cuda:0) ...")
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    t_load_wall = time.perf_counter() - t_load
    print(f"  model loaded in {t_load_wall:.1f}s")

    # Also extract lm_head_full while model is loaded
    lm_head_out = os.path.join(PPL_PREP, "lm_head_full.npy")
    if not os.path.exists(lm_head_out):
        lm_head = model.lm_head.weight.detach().cpu().to(torch.float32).numpy()
        np.save(lm_head_out, lm_head)
        probe_save = os.path.join(PROBE_FULL, "lm_head.npy")
        if not os.path.exists(probe_save):
            np.save(probe_save, lm_head)
        sz = os.path.getsize(lm_head_out)
        print(f"  lm_head_full.npy saved ({sz/1024**2:.0f} MB)")
        del lm_head

    todo = [i for i in range(num_windows) if i not in done]
    print(f"  capturing {len(todo)} windows (skipping {len(done)} already done) ...")
    t_capture_start = time.perf_counter()

    for w in todo:
        tids = token_ids_arr[w].tolist()
        input_ids = torch.tensor([tids], device="cuda:0")
        pre_norm_capture = {}
        hook = model.model.norm.register_forward_pre_hook(
            lambda m, inp: (pre_norm_capture.update(x=inp[0].clone()), None)[1])
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        hook.remove()

        # pytorch_ref: (33, 64, 4096) float64
        pytorch_ref = np.stack([
            h_.squeeze(0).detach().cpu().to(torch.float32).numpy().astype(np.float64)
            for h_ in out.hidden_states
        ], axis=0)
        # pytorch_pre_norm: (64, 4096) float64 — pre-final-RMSNorm
        pytorch_pre_norm = (pre_norm_capture['x'].squeeze(0)
                            .detach().cpu().to(torch.float32).numpy()
                            .astype(np.float64))
        # pt_logprobs: (64, vocab) float32 — log_softmax over full vocab
        # logprob at position i predicts token at i+1; position 63 unused (last)
        logits_all = out.logits[0].detach().cpu().to(torch.float32)  # (64, vocab)
        import torch.nn.functional as F
        pt_logprobs = F.log_softmax(logits_all, dim=-1).numpy().astype(np.float32)

        out_path = os.path.join(refs_dir, f"ppl_window_{w:04d}.npz")
        np.savez_compressed(out_path,
                            pytorch_ref=pytorch_ref,
                            pytorch_pre_norm=pytorch_pre_norm,
                            pt_logprobs=pt_logprobs,
                            token_ids=token_ids_arr[w])

        # Free intermediates
        del out, input_ids, logits_all, pytorch_ref, pytorch_pre_norm, pt_logprobs
        del pre_norm_capture
        torch.cuda.empty_cache()

        if (w + 1) % 16 == 0 or w == todo[-1]:
            elapsed = time.perf_counter() - t_capture_start
            rate = (todo.index(w) + 1) / elapsed if elapsed > 0 else 0
            eta = (len(todo) - todo.index(w) - 1) / rate if rate > 0 else 0
            print(f"    window {w+1}/{num_windows} done  "
                  f"({elapsed:.0f}s elapsed, {rate:.2f} win/s, ETA {eta:.0f}s)")

    t_capture_wall = time.perf_counter() - t_capture_start
    print(f"  capture done: {len(todo)} windows in {t_capture_wall:.1f}s "
          f"({t_capture_wall/max(len(todo),1):.2f}s/win)")

    # Free model + GPU memory
    del model
    torch.cuda.empty_cache()
    print("  PT model freed + cuda cache cleared")
    return t_load_wall, t_capture_wall


# ---------------------------------------------------------------------------
# Deliverable 5: PT-only PPL smoke test
# ---------------------------------------------------------------------------
def compute_pt_ppl(token_ids_arr, loss_mask_arr, num_windows, tok):
    """Compute PT-only PPL from captured refs using pt_logprobs."""
    print("=== Deliverable 5: PT-only PPL smoke test ===")
    refs_dir = os.path.join(PPL_PREP, "refs")

    sum_nll = 0.0           # sum of -log P(t_i | t_{<i}) over scored positions
    n_scored_tokens = 0
    scored_token_ids_list = []  # collect scored token ids for byte/word PPL

    for w in range(num_windows):
        ref_path = os.path.join(refs_dir, f"ppl_window_{w:04d}.npz")
        z = np.load(ref_path)
        pt_logprobs = z["pt_logprobs"]        # (64, vocab) float32
        tids = token_ids_arr[w]               # (64,) int64
        mask = loss_mask_arr[w]               # (64,) bool

        # Standard LM convention: logprobs[i] = log P(t_{i+1} | t_{<=i})
        # so position i predicts t_{i+1}; position 63 has no target.
        # For scored position i (where mask[i] is True AND i < M-1):
        #   contribution = pt_logprobs[i-1, tids[i]]  (log P of tids[i] given context up to i-1)
        # BUT: mask says which positions contribute to PPL sum.
        # In the plan's convention: loss_mask[i] = True means position i is scored,
        # meaning we sum -log P(tids[i] | context).
        # This corresponds to pt_logprobs[i-1, tids[i]] = log P at position i-1 predicting tids[i].
        # Position 0 can be scored only if there's a token before it — but window 0
        # has the full 64 tokens scored; the prediction for position 0 would require
        # a token at position -1 which doesn't exist. Handle: skip position 0 for window 0
        # since there's no valid "previous context" logit.
        # Actually the loss_mask marks positions whose log-prob contributes to H.
        # For window 0: positions 0..63 are all scored but position 0 has no prediction.
        # For window w>0: positions M-S..M-1 (32..63) are scored; position 32 is predicted
        #   by logprobs[31], which is valid since position 31 is context.
        # So we use: for scored position i, add pt_logprobs[i-1, tids[i]] when i >= 1.

        for i in range(M):
            if not mask[i]:
                continue
            if i == 0:
                # No previous context logit available; skip (only affects window 0 position 0)
                continue
            logp = float(pt_logprobs[i - 1, tids[i]])
            sum_nll += -logp
            n_scored_tokens += 1
            scored_token_ids_list.append(int(tids[i]))

    # Decode scored token ids for byte/word PPL
    scored_text = tok.decode(scored_token_ids_list, skip_special_tokens=False)
    n_scored_bytes = len(scored_text.encode("utf-8"))
    n_scored_words = len(scored_text.split()) if scored_text.strip() else 1

    H = sum_nll / n_scored_tokens
    ppl_token = math.exp(H)
    ppl_byte = math.exp(H * n_scored_tokens / n_scored_bytes)
    ppl_word = math.exp(H * n_scored_tokens / n_scored_words)

    print(f"\n  ===== PT-only PPL on {num_windows} windows =====")
    print(f"  Scored tokens:     {n_scored_tokens:,}")
    print(f"  Scored bytes:      {n_scored_bytes:,}")
    print(f"  Scored words:      {n_scored_words:,}")
    print(f"  tokens/byte:       {n_scored_tokens/n_scored_bytes:.4f}")
    print(f"  tokens/word:       {n_scored_tokens/n_scored_words:.4f}")
    print(f"  Mean NLL (H):      {H:.4f} nats/token")
    print(f"  PT token-PPL:      {ppl_token:.4f}")
    print(f"  PT byte-PPL:       {ppl_byte:.4f}")
    print(f"  PT word-PPL:       {ppl_word:.4f}")
    print(f"  ================================================")

    results = {
        "num_windows": num_windows,
        "n_scored_tokens": n_scored_tokens,
        "n_scored_bytes": n_scored_bytes,
        "n_scored_words": n_scored_words,
        "tokens_per_byte": n_scored_tokens / n_scored_bytes,
        "tokens_per_word": n_scored_tokens / n_scored_words,
        "mean_nll_nats": H,
        "pt_token_ppl": ppl_token,
        "pt_byte_ppl": ppl_byte,
        "pt_word_ppl": ppl_word,
        "M": M,
        "S": S,
    }
    out_path = os.path.join(PPL_PREP, "pt_ppl_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved {out_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-windows", type=int, default=MAX_WINDOWS)
    parser.add_argument("--skip-pt-capture", action="store_true",
                        help="Skip PT model load/capture (assume refs already exist)")
    args = parser.parse_args()

    t_total = time.perf_counter()

    # Check if windows.npz already exists
    windows_path = os.path.join(PPL_PREP, "windows.npz")
    stats_path = os.path.join(PPL_PREP, "corpus_stats.json")
    if os.path.exists(windows_path) and os.path.exists(stats_path):
        print(f"  windows.npz + corpus_stats.json already exist — loading ...")
        z = np.load(windows_path)
        token_ids_arr = z["token_ids"]
        loss_mask_arr = z["loss_mask"]
        byte_offsets_arr = z["byte_offsets"]
        word_offsets_arr = z["word_offsets"]
        with open(stats_path) as f:
            stats = json.load(f)
        # Re-load tokenizer for PPL smoke test
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        print("  loading tokenizer for smoke test ...")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Clamp to requested num_windows
        n_windows = min(args.num_windows, token_ids_arr.shape[0])
        token_ids_arr = token_ids_arr[:n_windows]
        loss_mask_arr = loss_mask_arr[:n_windows]
    else:
        token_ids_arr, loss_mask_arr, byte_offsets_arr, word_offsets_arr, stats, tok = \
            build_windows(args.num_windows)
        n_windows = stats["n_windows"]

    print(f"\n  Corpus stats: {json.dumps(stats, indent=2)}")

    # Deliverable 2: lm_head_full
    lm_head = ensure_lm_head_full()

    # Deliverable 3: PT capture
    if not args.skip_pt_capture:
        result = capture_all_windows(token_ids_arr, n_windows)
        if result is not None:
            t_load_wall, t_capture_wall = result
            print(f"\n  PT model load wall: {t_load_wall:.1f}s")
            print(f"  PT capture wall ({n_windows} windows): {t_capture_wall:.1f}s")
    else:
        print("  --skip-pt-capture: skipping PT model load")

    # Verify refs exist
    refs_dir = os.path.join(PPL_PREP, "refs")
    n_refs = sum(1 for i in range(n_windows)
                 if os.path.exists(os.path.join(refs_dir, f"ppl_window_{i:04d}.npz")))
    print(f"\n  refs present: {n_refs}/{n_windows}")

    # Deliverable 5: PT-only PPL smoke test
    if n_refs >= n_windows:
        compute_pt_ppl(token_ids_arr, loss_mask_arr, n_windows, tok)
    else:
        print(f"  WARNING: only {n_refs}/{n_windows} refs present; skipping PT PPL smoke test")

    print(f"\n=== Total wall: {time.perf_counter()-t_total:.1f}s ===")


if __name__ == "__main__":
    main()
