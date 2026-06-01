"""Fixed-`nt` speed benchmark to compare per-layer FHE compute time
with Cachemir's published table (Cachemir's bench uses fixed
seq_len = 512 on A100 80 GB).

Pipeline:
  1. Build a REAL `nt`-token input by concatenating MRPC dev prompts
     (natural English). Every position holds a meaningful token.
  2. Capture PyTorch reference once (HF model load + forward) in a
     subprocess so the 15-16 GB PyTorch GPU retention dies with the
     child — the FHE pass starts on a clean GPU.
  3. Load CKKS engine.
  4. Run run_classifier_fhe with num_tokens=nt and query_position=nt-1
     (the LAST position). PyTorch's causal mask makes the query attend
     to all nt-1 previous positions; the FHE pipeline's "no mask"
     path is trivially equivalent because the query is at the end.
     All n_blocks = nt / T_MODEL attention blocks are exercised.
  5. Report per-layer timing breakdown.

Usage:
  python speed_bench.py            # nt=512 (Cachemir-style)
  python speed_bench.py --nt 256   # any fixed length up to rope-table limit
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


def _ptref_subprocess(target_nt, out_path):
    """Run PT-ref capture in a child process so its ~15 GB PyTorch
    allocator state is reclaimed at exit.

    Builds a REAL `target_nt`-token input by concatenating MRPC dev
    prompts (natural English) so every position holds a meaningful
    token. The query is placed at the LAST position (target_nt-1) so
    PyTorch's causal mask makes it attend to all target_nt-1 previous
    positions — the FHE pipeline's "no mask" path is then trivially
    equivalent (query is at the end, attends to everything before it).
    This is the Cachemir-style speed-bench setup.
    """
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

PROMPT_FMT = ('Are these two sentences paraphrases of each other?\\n'
              'Sentence 1: {{}}\\nSentence 2: {{}}\\nAnswer (Yes or No):')

target_nt = {target_nt}

# Build a long real input by concatenating MRPC prompts.
parts = []
total_tokens = 0
for idx in range(len(ds)):
    row = ds[idx]
    p = PROMPT_FMT.format(row['sentence1'], row['sentence2'])
    parts.append(p)
    total_tokens += len(tok(p).input_ids)
    if total_tokens > target_nt + 32:
        break
long_text = ' '.join(parts)
ids = tok(long_text).input_ids[:target_nt]
print(f'built {{len(ids)}}-token real input from '
      f'{{len(parts)}} MRPC prompts', flush=True)

model = AutoModelForCausalLM.from_pretrained(
    'NousResearch/Meta-Llama-3.1-8B',
    torch_dtype=torch.float16, device_map='cuda:0')
model.eval()
print('model loaded', flush=True)

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

# Query at the LAST position so causal mask is automatic (query attends
# to all previous positions). yes/no logits at that position are
# meaningless for paraphrase classification on the concatenated text,
# but the FHE forward pass exercises all target_nt attention positions,
# which is the speed metric we want.
last_logits = out.logits[0, target_nt - 1].to(torch.float32).cpu().numpy()
meta = json.loads(open('{PROBE}/meta.json').read())
yes_pt = float(last_logits[meta['yes_token_id']])
no_pt = float(last_logits[meta['no_token_id']])
np.savez('{out_path}', ref=pytorch_ref, prenorm=pytorch_pre_norm,
         yes=np.float64(yes_pt), no=np.float64(no_pt),
         real_nt=np.int32(target_nt))
print(f'saved ref shape={{pytorch_ref.shape}} prenorm={{pytorch_pre_norm.shape}}', flush=True)
print(f'yes_pt={{yes_pt:.4f}} no_pt={{no_pt:.4f}}', flush=True)
"""
    return subprocess.run([sys.executable, "-c", script], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=512)
    ap.add_argument("--rebuild-ptref", action="store_true")
    args = ap.parse_args()

    out_path = f"/tmp/speed_bench_ptref_nt{args.nt}.npz"
    if args.rebuild_ptref or not os.path.exists(out_path):
        print(f"[speed_bench] capturing PT-ref at nt={args.nt} (real "
              f"concatenated input, query at position nt-1) "
              f"in subprocess...", flush=True)
        t0 = time.perf_counter()
        _ptref_subprocess(args.nt, out_path)
        print(f"[speed_bench] PT-ref captured in {time.perf_counter() - t0:.1f}s",
              flush=True)

    z = np.load(out_path)
    pytorch_ref = z["ref"]
    pytorch_pre_norm = z["prenorm"]
    yes_pt = float(z["yes"])
    no_pt = float(z["no"])
    real_nt = int(z["real_nt"])
    print(f"[speed_bench] loaded PT-ref: ref={pytorch_ref.shape} "
          f"prenorm={pytorch_pre_norm.shape} nt={real_nt} (all real, query=nt-1)",
          flush=True)

    # Load RoPE tables (must have at least nt positions).
    cos_all = np.load(f"{PROBE}/rope_cos.npy").astype(np.float64)
    sin_all = np.load(f"{PROBE}/rope_sin.npy").astype(np.float64)
    if cos_all.shape[0] < args.nt:
        raise RuntimeError(f"rope_cos.npy has {cos_all.shape[0]} positions, "
                            f"need >= {args.nt}. Re-run rope_extend.")

    # Engine.
    import pyPhantom as phantom  # noqa: F401
    from fhe.llama3_mrpc import build_user_steps_mrpc, setup_engine, run_classifier_fhe
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
        cos_all_full=cos_all, sin_all_full=sin_all,
        label=f"speed_bench_nt{args.nt}",
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
