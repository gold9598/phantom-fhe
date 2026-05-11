"""Extract /tmp/llama_probe_full/ data from HF LLaMA-3.1-8B.

Required by llama3.py and mrpc_sweep.py. Run once on a fresh box before
the sweep:

  python setup_probe_data.py [--probe-dir /tmp/llama_probe_full]
                              [--rope-positions 128]
                              [--model NousResearch/Meta-Llama-3.1-8B]

Generates:
  layer_NN/{Wq,Wk,Wv,Wo,Wgate,Wup,Wdown,g1,g2}.npy   (32 layers)
  final_norm_g.npy
  lm_head_yesno.npy
  rope_cos.npy, rope_sin.npy   (cos/sin for positions 0..rope-positions)
  meta.json                    ({yes_token_id, no_token_id, rms_norm_eps})
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default="/tmp/llama_probe_full")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--rope-positions", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.probe_dir, exist_ok=True)
    print(f"Loading model {args.model}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu")
    cfg = model.config
    print(f"  hidden_size={cfg.hidden_size}  num_layers={cfg.num_hidden_layers}  "
          f"rms_norm_eps={cfg.rms_norm_eps}", flush=True)

    # Per-layer weight extraction (fp32 → fp64 done at load time by load_layer_weights).
    for L, layer in enumerate(model.model.layers):
        ld = os.path.join(args.probe_dir, f"layer_{L:02d}")
        os.makedirs(ld, exist_ok=True)
        np.save(f"{ld}/Wq.npy", layer.self_attn.q_proj.weight.detach().numpy())
        np.save(f"{ld}/Wk.npy", layer.self_attn.k_proj.weight.detach().numpy())
        np.save(f"{ld}/Wv.npy", layer.self_attn.v_proj.weight.detach().numpy())
        np.save(f"{ld}/Wo.npy", layer.self_attn.o_proj.weight.detach().numpy())
        np.save(f"{ld}/Wgate.npy", layer.mlp.gate_proj.weight.detach().numpy())
        np.save(f"{ld}/Wup.npy",   layer.mlp.up_proj.weight.detach().numpy())
        np.save(f"{ld}/Wdown.npy", layer.mlp.down_proj.weight.detach().numpy())
        np.save(f"{ld}/g1.npy", layer.input_layernorm.weight.detach().numpy())
        np.save(f"{ld}/g2.npy", layer.post_attention_layernorm.weight.detach().numpy())
        print(f"  layer {L:02d} saved", flush=True)

    np.save(f"{args.probe_dir}/final_norm_g.npy",
            model.model.norm.weight.detach().numpy())

    # Yes/No LM head rows. Llama-3 tokenizes " Yes" and " No" with leading space.
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]
    lm_head_yesno = np.stack([
        model.lm_head.weight[yes_id].detach().numpy(),
        model.lm_head.weight[no_id].detach().numpy(),
    ])
    np.save(f"{args.probe_dir}/lm_head_yesno.npy", lm_head_yesno)

    # RoPE tables (NTK-aware scaling per LLaMA-3.1 config).
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    base = float(getattr(cfg, "rope_theta", 500000.0))
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
    # llama-3.1 also applies an NTK-aware rescale via cfg.rope_scaling — match
    # transformers' default behavior. If rope_scaling is set, scale inv_freq.
    rs = getattr(cfg, "rope_scaling", None)
    if rs and rs.get("rope_type") == "llama3":
        factor = float(rs.get("factor", 8.0))
        low_freq = float(rs.get("low_freq_factor", 1.0))
        high_freq = float(rs.get("high_freq_factor", 4.0))
        old_max = float(rs.get("original_max_position_embeddings", 8192))
        # standard llama3 rope scaling — see HF modeling_llama.py rope_init_fn
        low_wavelen = old_max / low_freq
        high_wavelen = old_max / high_freq
        wavelens = 2 * np.pi / inv_freq
        scaled = inv_freq.copy()
        smooth = (old_max / wavelens - low_freq) / (high_freq - low_freq)
        is_high = wavelens < high_wavelen
        is_low = wavelens > low_wavelen
        scaled = np.where(is_low, inv_freq / factor, scaled)
        mid = (~is_low) & (~is_high)
        scaled = np.where(mid, (1 - smooth) * inv_freq / factor + smooth * inv_freq, scaled)
        inv_freq = scaled
    pos = np.arange(args.rope_positions, dtype=np.float64)
    freqs = np.einsum("i,j->ij", pos, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    np.save(f"{args.probe_dir}/rope_cos.npy", np.cos(emb))
    np.save(f"{args.probe_dir}/rope_sin.npy", np.sin(emb))

    meta = {
        "yes_token_id": int(yes_id),
        "no_token_id":  int(no_id),
        "rms_norm_eps": float(cfg.rms_norm_eps),
        "model": args.model,
        "rope_positions": int(args.rope_positions),
    }
    with open(f"{args.probe_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Probe data at {args.probe_dir}")
    print(f"  yes_id={yes_id} no_id={no_id} rms_eps={cfg.rms_norm_eps}")


if __name__ == "__main__":
    main()
