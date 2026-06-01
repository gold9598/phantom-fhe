"""Resume PT reference capture for windows that don't yet have a valid ref file.

Skips windows whose ppl_window_{i:04d}.npz already exists AND has size >= MIN_SIZE bytes.
Captures a contiguous range [start_win, end_win) in one model-load.

Usage:
  python resume_capture.py --start 82 --end 122
  python resume_capture.py --start 0  --end 256  # full idempotent run
"""
import argparse
import os
import sys
import time

import numpy as np

PPL_PREP  = "/home/yongwoo-oh/mrpc_campaign/ppl_prep"
REFS_DIR  = os.path.join(PPL_PREP, "refs")
MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B"
M = 64
MIN_SIZE  = 35_000_000   # bytes; healthy file ~41 MB, truncated ~22 MB


def needs_capture(w):
    p = os.path.join(REFS_DIR, f"ppl_window_{w:04d}.npz")
    if not os.path.exists(p):
        return True
    return os.path.getsize(p) < MIN_SIZE


def capture_range(token_ids_arr, start, end):
    todo = [w for w in range(start, end) if needs_capture(w)]
    if not todo:
        print(f"  [{start},{end}) all present — skip")
        return 0

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    free_gb = (torch.cuda.get_device_properties(0).total_memory
               - torch.cuda.memory_allocated(0)) / 1024**3
    print(f"  GPU free: {free_gb:.1f} GB")

    print(f"  Loading PT model (fp16, cuda:0) for [{start},{end}): {len(todo)} windows ...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    print(f"  model loaded in {time.perf_counter()-t0:.1f}s")

    t_cap = time.perf_counter()
    for idx, w in enumerate(todo):
        tids = token_ids_arr[w].tolist()
        input_ids = torch.tensor([tids], device="cuda:0")
        pre_norm_capture = {}
        hook = model.model.norm.register_forward_pre_hook(
            lambda m, inp: (pre_norm_capture.update(x=inp[0].clone()), None)[1])
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        hook.remove()

        pytorch_ref = np.stack([
            h_.squeeze(0).detach().cpu().to(torch.float32).numpy().astype(np.float64)
            for h_ in out.hidden_states
        ], axis=0)   # (33, 64, 4096)

        pytorch_pre_norm = (pre_norm_capture['x'].squeeze(0)
                            .detach().cpu().to(torch.float32).numpy()
                            .astype(np.float64))  # (64, 4096)

        logits_all = out.logits[0].detach().cpu().to(torch.float32)  # (64, vocab)
        pt_logprobs = F.log_softmax(logits_all, dim=-1).numpy().astype(np.float32)

        out_path = os.path.join(REFS_DIR, f"ppl_window_{w:04d}.npz")
        np.savez_compressed(out_path,
                            pytorch_ref=pytorch_ref,
                            pytorch_pre_norm=pytorch_pre_norm,
                            pt_logprobs=pt_logprobs,
                            token_ids=token_ids_arr[w])

        del out, input_ids, logits_all, pytorch_ref, pytorch_pre_norm, pt_logprobs
        del pre_norm_capture
        torch.cuda.empty_cache()

        if (idx + 1) % 8 == 0 or idx == len(todo) - 1:
            elapsed = time.perf_counter() - t_cap
            rate = (idx + 1) / elapsed
            eta  = (len(todo) - idx - 1) / rate if rate > 0 else 0
            print(f"    window {w:3d}  [{idx+1}/{len(todo)}]  "
                  f"{elapsed:.0f}s elapsed  {rate:.2f} win/s  ETA {eta:.0f}s")

    t_wall = time.perf_counter() - t_cap
    print(f"  captured {len(todo)} windows in {t_wall:.1f}s ({t_wall/len(todo):.1f}s/win)")

    del model
    torch.cuda.empty_cache()
    print("  model freed")
    return len(todo)


def smoke_test_ppl(token_ids_arr, loss_mask_arr, num_windows, tok):
    import math
    print("\n=== PT-only PPL smoke test ===")
    sum_nll = 0.0
    n = 0
    scored_ids = []

    for w in range(num_windows):
        p = os.path.join(REFS_DIR, f"ppl_window_{w:04d}.npz")
        z = np.load(p)
        pt_lp = z["pt_logprobs"]   # (64, vocab)
        tids  = token_ids_arr[w]
        mask  = loss_mask_arr[w]
        for i in range(1, M):
            if not mask[i]:
                continue
            sum_nll += -float(pt_lp[i - 1, tids[i]])
            n += 1
            scored_ids.append(int(tids[i]))

    scored_text = tok.decode(scored_ids, skip_special_tokens=False)
    n_bytes = len(scored_text.encode("utf-8"))
    n_words = len(scored_text.split()) if scored_text.strip() else 1

    H = sum_nll / n
    ppl_tok  = math.exp(H)
    ppl_byte = math.exp(H * n / n_bytes)
    ppl_word = math.exp(H * n / n_words)

    print(f"  Scored tokens : {n:,}")
    print(f"  Scored bytes  : {n_bytes:,}")
    print(f"  Scored words  : {n_words:,}")
    print(f"  Mean NLL      : {H:.4f} nats/tok")
    print(f"  PT token-PPL  : {ppl_tok:.4f}")
    print(f"  PT byte-PPL   : {ppl_byte:.4f}")
    print(f"  PT word-PPL   : {ppl_word:.4f}")
    return ppl_tok, ppl_byte, ppl_word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",  type=int, default=0)
    ap.add_argument("--end",    type=int, default=256)
    ap.add_argument("--smoke",  action="store_true", help="Run PPL smoke test after capture")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    z = np.load(os.path.join(PPL_PREP, "windows.npz"))
    token_ids_arr = z["token_ids"]
    loss_mask_arr = z["loss_mask"]

    os.makedirs(REFS_DIR, exist_ok=True)
    capture_range(token_ids_arr, args.start, args.end)

    if args.smoke:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        smoke_test_ppl(token_ids_arr, loss_mask_arr, 256, tok)


if __name__ == "__main__":
    main()
