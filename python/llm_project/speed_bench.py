"""Fixed-`nt` speed benchmark to compare per-layer FHE compute time with
Cachemir's published table (Cachemir's bench uses fixed seq_len = 512
on A100 80 GB).

Pipeline:
  1. Load CKKS engine.
  2. Take MRPC idx=5's prompt (69 real tokens), pad with EOS to nt=512.
  3. Capture PyTorch reference once (HF model load + forward) in a
     subprocess so the 15-16 GB PyTorch GPU retention dies with the
     child — the FHE pass starts on a clean GPU.
  4. Run run_classifier_fhe at num_tokens=512, query_position = real
     last-real-token index (so the predicted Yes/No logits come from
     the real prompt's last position, not from a padding token).
  5. Report per-layer timing breakdown.

Usage:
  python speed_bench.py            # nt=512 (Cachemir-style)
  python speed_bench.py --nt 256   # any fixed length up to rope-table limit
  python speed_bench.py --idx 5    # which MRPC prompt to pad
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, os.path.join(_REPO, "build", "lib"))
sys.path.insert(0, _THIS_DIR)

PROBE = "/tmp/llama_probe_full"


def _pad_ids(token_ids, target_nt, pad_id):
    if len(token_ids) > target_nt:
        return list(token_ids[:target_nt])
    return list(token_ids) + [pad_id] * (target_nt - len(token_ids))


def _ptref_subprocess(idx, target_nt, out_path):
    """Run PT-ref capture in a child process so its ~15 GB PyTorch
    allocator state is reclaimed at exit."""
    script = f"""
import os, sys, numpy as np
sys.path.insert(0, '{os.path.join(_REPO, "build", "lib")}')
sys.path.insert(0, '{_THIS_DIR}')
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

tok = AutoTokenizer.from_pretrained('NousResearch/Meta-Llama-3.1-8B')
ds = load_dataset('nyu-mll/glue', 'mrpc')['validation']
row = ds[{idx}]
prompt = (
    'Are these two sentences paraphrases of each other?\\n'
    'Sentence 1: {{}}\\nSentence 2: {{}}\\nAnswer (Yes or No):'
).format(row['sentence1'], row['sentence2'])
real_ids = tok(prompt).input_ids
real_nt = len(real_ids)
pad_id = tok.eos_token_id
target_nt = {target_nt}
ids = real_ids + [pad_id] * (target_nt - real_nt) if real_nt < target_nt else real_ids[:target_nt]
print(f'real_nt={{real_nt}}  pad_to={{target_nt}}  pad_id={{pad_id}}', flush=True)

model = AutoModelForCausalLM.from_pretrained(
    'NousResearch/Meta-Llama-3.1-8B',
    torch_dtype=torch.float16, device_map='cuda:0')
model.eval()
print('model loaded', flush=True)

# No attention mask: FHE pipeline doesn't apply a mask either, so for
# apples-to-apples timing comparison both sides do attention over all
# nt positions including the padded tail. The Yes/No logits at the
# real_last_token position are still well-defined.
input_ids = torch.tensor([ids], device='cuda:0')
pre_norm_capture = {{}}
h = model.model.norm.register_forward_pre_hook(
    lambda m, i: (pre_norm_capture.update(x=i[0].clone()), None)[1])
with torch.no_grad():
    out = model(input_ids=input_ids, output_hidden_states=True)
h.remove()
pytorch_ref = np.stack([
    hs.squeeze(0).detach().cpu().to(torch.float32).numpy().astype(np.float64)
    for hs in out.hidden_states
], axis=0)
pytorch_pre_norm = pre_norm_capture['x'].squeeze(0).detach().cpu().to(
    torch.float32).numpy().astype(np.float64)

last_logits = out.logits[0, real_nt - 1].to(torch.float32).cpu().numpy()
meta = json.loads(open('{PROBE}/meta.json').read())
yes_pt = float(last_logits[meta['yes_token_id']])
no_pt = float(last_logits[meta['no_token_id']])
np.savez('{out_path}', ref=pytorch_ref, prenorm=pytorch_pre_norm,
         yes=np.float64(yes_pt), no=np.float64(no_pt),
         real_nt=np.int32(real_nt), pad_id=np.int32(pad_id))
print(f'saved ref shape={{pytorch_ref.shape}} prenorm={{pytorch_pre_norm.shape}}', flush=True)
print(f'yes_pt={{yes_pt:.4f}} no_pt={{no_pt:.4f}}', flush=True)
"""
    return subprocess.run([sys.executable, "-c", script], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=512)
    ap.add_argument("--idx", type=int, default=5)
    ap.add_argument("--rebuild-ptref", action="store_true")
    args = ap.parse_args()

    out_path = f"/tmp/mrpc_ptref_nt{args.nt}_idx{args.idx}.npz"
    if args.rebuild_ptref or not os.path.exists(out_path):
        print(f"[speed_bench] capturing PT-ref at nt={args.nt} in subprocess...",
              flush=True)
        t0 = time.perf_counter()
        _ptref_subprocess(args.idx, args.nt, out_path)
        print(f"[speed_bench] PT-ref captured in {time.perf_counter() - t0:.1f}s",
              flush=True)

    z = np.load(out_path)
    pytorch_ref = z["ref"]
    pytorch_pre_norm = z["prenorm"]
    yes_pt = float(z["yes"])
    no_pt = float(z["no"])
    real_nt = int(z["real_nt"])
    print(f"[speed_bench] loaded PT-ref: ref={pytorch_ref.shape} "
          f"prenorm={pytorch_pre_norm.shape} real_nt={real_nt}", flush=True)
    print(f"[speed_bench] PT logits at real-last-token: "
          f"yes={yes_pt:.4f} no={no_pt:.4f} → pt_pred="
          f"{'Yes' if yes_pt > no_pt else 'No'}", flush=True)

    # Load RoPE tables (must have at least nt positions).
    cos_all = np.load(f"{PROBE}/rope_cos.npy").astype(np.float64)
    sin_all = np.load(f"{PROBE}/rope_sin.npy").astype(np.float64)
    if cos_all.shape[0] < args.nt:
        raise RuntimeError(f"rope_cos.npy has {cos_all.shape[0]} positions, "
                            f"need >= {args.nt}. Re-run rope_extend.")

    # Engine.
    import pyPhantom as phantom  # noqa: F401
    from llama3_mrpc import build_user_steps_mrpc, setup_engine, run_classifier_fhe
    user_steps, step_categories = build_user_steps_mrpc()
    print(f"[speed_bench] building CKKS engine...", flush=True)
    engine = setup_engine(user_steps, step_categories=step_categories)

    # Disk cache root (rp_indep + wq).
    rp_disk = os.path.join(_REPO, "cache", "rp_indep")
    wq_root = os.path.join(_REPO, "cache", "wq", f"nt_{args.nt}")
    if not os.path.exists(os.path.join(wq_root, "MANIFEST.json")):
        print(f"[speed_bench] WARN: wq cache for nt={args.nt} not built. "
              f"Streaming will JIT-encode every layer (~5-7 s extra each). "
              f"To pre-build: python build_wq_disk_cache.py --nt {args.nt}",
              flush=True)
    if not os.path.exists(os.path.join(rp_disk, "MANIFEST.json")):
        raise RuntimeError(
            f"rp_indep disk cache missing at {rp_disk}. Run "
            f"build_disk_cache.py first.")

    print(f"[speed_bench] running run_classifier_fhe at nt={args.nt} ...",
          flush=True)
    t_total0 = time.perf_counter()
    yes_logit, no_logit = run_classifier_fhe(
        num_tokens=args.nt,
        query_position=real_nt - 1,
        pytorch_ref=pytorch_ref,
        pytorch_pre_norm=pytorch_pre_norm,
        cos_all=cos_all, sin_all=sin_all,
        label=f"nt{args.nt}_idx{args.idx}",
        debug_layer=None, max_layer=None, min_layer=None,
        rp_indep_cache={}, engine=engine,
        rp_indep_disk_root=rp_disk)
    t_total = time.perf_counter() - t_total0
    fhe_pred = "Yes" if yes_logit > no_logit else "No"
    pt_pred = "Yes" if yes_pt > no_pt else "No"
    print(f"\n=== speed_bench summary ===")
    print(f"  nt={args.nt}  real_nt={real_nt}  n_blocks={args.nt // 8}")
    print(f"  total wall: {t_total:.1f}s  (~{t_total / 32:.1f}s avg/layer)")
    print(f"  PT  : yes={yes_pt:.4f} no={no_pt:.4f} → {pt_pred}")
    print(f"  FHE : yes={yes_logit:.4f} no={no_logit:.4f} → {fhe_pred}")
    print(f"  agree: {pt_pred == fhe_pred}")


if __name__ == "__main__":
    main()
