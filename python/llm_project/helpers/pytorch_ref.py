"""PyTorch reference capture helpers split out of llama3_mrpc.py."""
# design: doc/design/pytorch_ref.md#module-contents-and-reexport
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from helpers.llama3 import PROBE_FULL


def capture_pytorch_ref_with_model(model, tok, token_ids):
    """Run a forward pass on a pre-loaded model and return the same data as capture_pytorch_ref."""
    # design: doc/design/pytorch_ref.md#capture-with-model-contract
    import torch
    input_ids = torch.tensor([token_ids], device="cuda:0")
    pre_norm_capture = {}
    h = model.model.norm.register_forward_pre_hook(
        lambda m, i: (pre_norm_capture.update(x=i[0].clone()), None)[1])
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    h.remove()
    pytorch_ref = np.stack([
        h_.squeeze(0).detach().cpu().to(torch.float32).numpy().astype(np.float64)
        for h_ in out.hidden_states
    ], axis=0)
    pytorch_pre_norm = pre_norm_capture['x'].squeeze(0).detach().cpu().to(torch.float32).numpy().astype(np.float64)
    last_logits = out.logits[0, -1].to(torch.float32).cpu().numpy()
    meta = json.loads(open(f"{PROBE_FULL}/meta.json").read())
    yes_pt = float(last_logits[meta["yes_token_id"]])
    no_pt = float(last_logits[meta["no_token_id"]])
    return pytorch_ref, pytorch_pre_norm, yes_pt, no_pt


def capture_pytorch_ref(token_ids):
    """Run PyTorch LLaMA-3.1-8B forward on token_ids and capture all hidden states + the pre-final-norm last hidden state."""
    # design: doc/design/pytorch_ref.md#capture-pytorch-ref-returns
    import torch
    from transformers import AutoModelForCausalLM
    print(f"  Loading PyTorch model (fp16)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3.1-8B",
                                                  torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    ref, prenorm, yes_pt, no_pt = capture_pytorch_ref_with_model(model, None, token_ids)
    del model
    torch.cuda.empty_cache()
    return ref, prenorm, yes_pt, no_pt


def _cached_pytorch_ref(idx, truncate_to, token_ids):
    """Load cached PT reference for (idx, truncate_to) from disk if present; otherwise run capture_pytorch_ref and save to disk."""
    # design: doc/design/pytorch_ref.md#cached-ptref-disk-cache
    cache_path = f"/tmp/mrpc_ptref_idx{idx}_n{len(token_ids)}.npz"
    if __import__("os").path.exists(cache_path):
        print(f"  [cache hit] loading PT ref from {cache_path}")
        z = np.load(cache_path)
        return z["ref"], z["prenorm"], float(z["yes"]), float(z["no"])
    print(f"  [cache miss] running PT and saving to {cache_path}")
    ref, prenorm, yes_pt, no_pt = capture_pytorch_ref(token_ids)
    np.savez(cache_path, ref=ref, prenorm=prenorm,
             yes=np.float64(yes_pt), no=np.float64(no_pt))
    return ref, prenorm, yes_pt, no_pt
