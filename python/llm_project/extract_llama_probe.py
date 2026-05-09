"""Phase 2: extract LLaMA-3.1-8B probe + capture per-example MRPC outputs.

Outputs to /tmp/llama_probe_full/:
  embed_tokens.npy           (vocab=128256, D_MODEL=4096) fp32
  lm_head.npy                (vocab=128256, D_MODEL=4096) fp32
  lm_head_yesno.npy          (2, D_MODEL) - just " Yes"/" No" rows
  final_norm_g.npy           (D_MODEL,)
  rope_cos.npy               (max_seq=2048, D_HEAD=128)
  rope_sin.npy               (max_seq=2048, D_HEAD=128)
  meta.json                  model config
  layer_{00..31}/{Wq, Wk, Wv, Wo, Wgate, Wup, Wdown, g1, g2}.npy
  mrpc_pytorch_raw.npz       per-example PyTorch outputs on MRPC dev
  mrpc_prompts.txt           prompts (one per example, separated by ---)
  ref_acts/ex{idx:02d}.npy   per-layer hidden states for 5 sample examples
                             shape (num_layers+1, seq_len, D_MODEL)
  ref_acts/ex{idx:02d}_meta.json  example metadata
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B"
PROBE_DIR = Path("/tmp/llama_probe_full")
MAX_SEQ_FOR_ROPE = 2048

PROMPT = (
    "Are these two sentences paraphrases of each other?\n"
    "Sentence 1: {s1}\n"
    "Sentence 2: {s2}\n"
    "Answer (Yes or No):"
)

REF_ACT_EXAMPLE_IDS = [0, 17, 50, 100, 250]


def _save(path, t):
    np.save(path, t.detach().cpu().to(torch.float32).numpy())


def extract_weights(model, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  embed_tokens / lm_head / final_norm ...")
    _save(save_dir / "embed_tokens.npy", model.model.embed_tokens.weight)
    _save(save_dir / "lm_head.npy",      model.lm_head.weight)
    _save(save_dir / "final_norm_g.npy", model.model.norm.weight)

    n_layers = len(model.model.layers)
    print(f"  per-layer weights ({n_layers} layers) ...")
    t0 = time.time()
    for i, layer in enumerate(model.model.layers):
        ldir = save_dir / f"layer_{i:02d}"
        ldir.mkdir(exist_ok=True)
        sa = layer.self_attn
        mlp = layer.mlp
        _save(ldir / "Wq.npy",    sa.q_proj.weight)
        _save(ldir / "Wk.npy",    sa.k_proj.weight)
        _save(ldir / "Wv.npy",    sa.v_proj.weight)
        _save(ldir / "Wo.npy",    sa.o_proj.weight)
        _save(ldir / "Wgate.npy", mlp.gate_proj.weight)
        _save(ldir / "Wup.npy",   mlp.up_proj.weight)
        _save(ldir / "Wdown.npy", mlp.down_proj.weight)
        _save(ldir / "g1.npy", layer.input_layernorm.weight)
        _save(ldir / "g2.npy", layer.post_attention_layernorm.weight)
        if (i + 1) % 8 == 0:
            print(f"    layer {i+1}/{n_layers} done ({time.time()-t0:.1f}s)")


def extract_rope(model, save_dir, max_seq=MAX_SEQ_FOR_ROPE):
    print(f"  RoPE cos/sin (max_seq={max_seq}) ...")
    rotary = model.model.rotary_emb
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    position_ids = torch.arange(max_seq, device=model.device).unsqueeze(0)
    dummy = torch.zeros((1, 1, head_dim), dtype=torch.float32, device=model.device)
    with torch.no_grad():
        cos, sin = rotary(dummy, position_ids)
    cos = cos.squeeze(0).detach().cpu().to(torch.float32).numpy()
    sin = sin.squeeze(0).detach().cpu().to(torch.float32).numpy()
    np.save(save_dir / "rope_cos.npy", cos)
    np.save(save_dir / "rope_sin.npy", sin)
    print(f"    rope_cos shape={cos.shape}, rope_sin shape={sin.shape}")


def write_meta(model, tokenizer, save_dir, yes_id, no_id):
    cfg = model.config
    meta = {
        "model_name": MODEL_NAME,
        "n_layers": cfg.num_hidden_layers,
        "d_model": cfg.hidden_size,
        "n_heads": cfg.num_attention_heads,
        "n_kv_heads": cfg.num_key_value_heads,
        "d_head": cfg.hidden_size // cfg.num_attention_heads,
        "d_hidden": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "rope_parameters": dict(getattr(cfg, "rope_parameters", {}) or {}),
        "max_position_embeddings": cfg.max_position_embeddings,
        "rms_norm_eps": cfg.rms_norm_eps,
        "torch_dtype": str(model.dtype),
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "max_seq_for_rope": MAX_SEQ_FOR_ROPE,
        "extracted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  meta: {meta}")
    return meta


def save_lm_head_yesno(model, save_dir, yes_id, no_id):
    w = model.lm_head.weight.detach().cpu().to(torch.float32).numpy()
    yesno = np.stack([w[yes_id], w[no_id]], axis=0)
    np.save(save_dir / "lm_head_yesno.npy", yesno)
    print(f"  lm_head_yesno shape={yesno.shape}")


def run_mrpc_capture(model, tokenizer, save_dir, yes_id, no_id):
    print(f"  MRPC dev pass + per-example capture ...")
    ds = load_dataset("nyu-mll/glue", "mrpc")["validation"]
    n = len(ds)
    yes_logits = np.zeros(n); no_logits = np.zeros(n)
    preds = np.zeros(n, dtype=np.int8); labels = np.zeros(n, dtype=np.int8)
    prompt_lengths = np.zeros(n, dtype=np.int32)
    prompts = []
    t0 = time.time()
    with torch.no_grad():
        for i, row in enumerate(ds):
            prompt = PROMPT.format(s1=row["sentence1"], s2=row["sentence2"])
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            out = model(**inputs)
            last = out.logits[0, -1]
            yes_logits[i] = float(last[yes_id].item())
            no_logits[i]  = float(last[no_id].item())
            preds[i]  = 1 if yes_logits[i] > no_logits[i] else 0
            labels[i] = int(row["label"])
            prompt_lengths[i] = inputs.input_ids.shape[1]
            prompts.append(prompt)
            if (i + 1) % 100 == 0:
                acc = float(np.mean(preds[:i+1] == labels[:i+1]))
                print(f"    [{i+1}/{n}] acc={acc:.4f} rate={(i+1)/(time.time()-t0):.2f} ex/s")
    np.savez(save_dir / "mrpc_pytorch_raw.npz",
             yes_logits=yes_logits, no_logits=no_logits,
             predictions=preds, labels=labels, prompt_lengths=prompt_lengths)
    (save_dir / "mrpc_prompts.txt").write_text("\n---\n".join(prompts))
    acc = float(np.mean(preds == labels))
    print(f"  mrpc raw acc={acc:.4f}  prompt_lens range=[{prompt_lengths.min()}, "
          f"{prompt_lengths.max()}] mean={prompt_lengths.mean():.1f}")


def capture_ref_activations(model, tokenizer, save_dir):
    print(f"  per-layer reference activations for {len(REF_ACT_EXAMPLE_IDS)} examples ...")
    ds = load_dataset("nyu-mll/glue", "mrpc")["validation"]
    ref_dir = save_dir / "ref_acts"
    ref_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for idx in REF_ACT_EXAMPLE_IDS:
            row = ds[idx]
            prompt = PROMPT.format(s1=row["sentence1"], s2=row["sentence2"])
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            out = model(**inputs, output_hidden_states=True)
            # tuple of (n_layers+1) tensors, each (1, seq_len, d_model)
            hs = np.stack([h.squeeze(0).detach().cpu().to(torch.float32).numpy()
                           for h in out.hidden_states], axis=0)
            np.save(ref_dir / f"ex{idx:03d}.npy", hs)
            meta = {
                "idx": idx, "label": int(row["label"]),
                "sentence1": row["sentence1"], "sentence2": row["sentence2"],
                "prompt_len": int(inputs.input_ids.shape[1]),
                "shape": list(hs.shape),
            }
            (ref_dir / f"ex{idx:03d}_meta.json").write_text(json.dumps(meta, indent=2))
            print(f"    ex{idx:03d}: shape={hs.shape}  label={row['label']}")


def main():
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Phase 2: extracting probe to {PROBE_DIR}")

    print(f"Loading tokenizer + model (fp16, cuda:0) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    yes_id = tokenizer(" Yes", add_special_tokens=False).input_ids[0]
    no_id  = tokenizer(" No",  add_special_tokens=False).input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda:0")
    model.eval()
    print(f"  load done in {time.time()-t0:.1f}s; yes_id={yes_id} no_id={no_id}")

    write_meta(model, tokenizer, PROBE_DIR, yes_id, no_id)
    save_lm_head_yesno(model, PROBE_DIR, yes_id, no_id)
    extract_rope(model, PROBE_DIR)
    extract_weights(model, PROBE_DIR)
    run_mrpc_capture(model, tokenizer, PROBE_DIR, yes_id, no_id)
    capture_ref_activations(model, tokenizer, PROBE_DIR)
    print(f"Done. Probe at {PROBE_DIR}")


if __name__ == "__main__":
    main()
