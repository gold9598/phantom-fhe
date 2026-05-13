"""Test for blocks.lm_head.yes_no_logits_np against PyTorch (Stage 3b-e).

Verifies that our numpy LM head (final RMSNorm + 2 lm_head rows dot-product)
matches PyTorch's logits at the Yes/No token positions, given the same final
hidden state. Uses the qbrown4_bos.npy reference hidden states extracted in
Phase 2.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from blocks.lm_head import yes_no_logits_np

PROBE_FULL = "/tmp/llama_probe_full"


def main():
    # Load probe v2 artifacts
    final_norm_g = np.load(f"{PROBE_FULL}/final_norm_g.npy").astype(np.float64)
    lm_head_yesno = np.load(f"{PROBE_FULL}/lm_head_yesno.npy").astype(np.float64)
    meta = json.loads(open(f"{PROBE_FULL}/meta.json").read())
    yes_id = meta["yes_token_id"]
    no_id = meta["no_token_id"]
    rms_eps = meta["rms_norm_eps"]
    print(f"  final_norm_g shape: {final_norm_g.shape}")
    print(f"  lm_head_yesno shape: {lm_head_yesno.shape}")
    print(f"  yes_id={yes_id}, no_id={no_id}, rms_eps={rms_eps}")

    # PyTorch reference. qbrown4_bos.npy[32] is HF's hidden_states[-1] which is
    # POST-final-norm; we need PRE-norm for our pipeline (the FHE pipeline ends
    # at layer 31's residual2 output — pre-final-norm — so the LM head wiring
    # applies the final RMSNorm itself).
    pytorch_pre = np.load(f"{PROBE_FULL}/ref_acts/qbrown4_bos_prenorm.npy").astype(np.float64)
    P = 3
    y_final_pre = pytorch_pre[P]
    print(f"  y_final_pre shape: {y_final_pre.shape}, "
          f"||y_final_pre||={np.linalg.norm(y_final_pre):.4f}")

    # Numpy: Yes/No logits via our LM head (rmsnorm + dot)
    yes_np, no_np = yes_no_logits_np(y_final_pre, final_norm_g, lm_head_yesno, eps=rms_eps)
    print(f"  numpy LM head: yes_logit={yes_np:.4f}  no_logit={no_np:.4f}")

    # PyTorch: load model, run forward on the same 4-token prompt, extract
    # logits at last position (P=3) at Yes/No token IDs
    print(f"  Loading PyTorch model for reference...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3.1-8B",
                                                  torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    ids = [128000, 791, 4062, 14198]  # [BOS, "The", " quick", " brown"]
    input_ids = torch.tensor([ids], device="cuda:0")
    with torch.no_grad():
        out = model(input_ids=input_ids)
    last_logits = out.logits[0, -1].to(torch.float32).cpu().numpy()
    yes_torch = float(last_logits[yes_id])
    no_torch = float(last_logits[no_id])
    print(f"  pytorch logits: yes_logit={yes_torch:.4f}  no_logit={no_torch:.4f}")

    yes_diff = abs(yes_np - yes_torch)
    no_diff = abs(no_np - no_torch)
    print(f"  diff: yes={yes_diff:.3e}  no={no_diff:.3e}")
    ok = yes_diff < 0.05 and no_diff < 0.05
    print(f"  {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
