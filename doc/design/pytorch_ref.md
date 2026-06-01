# Design rationale: `pytorch_ref.py`

Design-rationale prose migrated out of
`python/llm_project/pytorch_ref.py` (PyTorch reference capture helpers for the
FHE/CKKS LLaMA-3.1-8B inference pipeline). The compacted source keeps the
one-line API summary of each function and points here for the WHY.

---

## module-contents-and-reexport

`pytorch_ref.py` contains:

- `capture_pytorch_ref_with_model` : run a forward on a pre-loaded model
- `capture_pytorch_ref`            : load LLaMA-3.1-8B + capture all hidden states
- `_cached_pytorch_ref`            : on-disk cache around capture_pytorch_ref

External callers (`mrpc_sweep`, `mrpc_sweep_parallel`, `precapture_ptref`, etc.)
keep their `from llama3_mrpc import capture_pytorch_ref*` paths via re-export.
That is why these helpers live in their own module yet must stay importable
under the original `llama3_mrpc` name.

---

## capture-with-model-contract

`capture_pytorch_ref_with_model` runs a forward pass on a pre-loaded model and
returns the same data as `capture_pytorch_ref`. The caller is responsible for
loading and deleting the model; this function does NOT load or free it.

Args:

- `model`: pre-loaded `AutoModelForCausalLM` on `cuda:0` (fp16, eval mode).
- `tok`:   unused; kept for call-site symmetry with `capture_pytorch_ref`.
- `token_ids`: `list[int]` token ids for the prompt.

Returns:

- `pytorch_ref`:      `(n_layers+1, num_tokens, D_MODEL)` ndarray float64
- `pytorch_pre_norm`: `(num_tokens, D_MODEL)` ndarray float64
- `yes_pt, no_pt`:    float logits at the last token position

---

## capture-pytorch-ref-returns

`capture_pytorch_ref` runs the PyTorch LLaMA-3.1-8B forward on `token_ids` and
captures all hidden states plus the pre-final-norm last hidden state. Returns:

- `pytorch_ref`:      `(n_layers+1, num_tokens, D_MODEL)` — post-final-norm at idx -1
- `pytorch_pre_norm`: `(num_tokens, D_MODEL)` — pre-final-norm last hidden state
- `yes_logit, no_logit`: PyTorch reference logits at the last token position

---

## cached-ptref-disk-cache

`_cached_pytorch_ref` loads the cached PT reference for `(idx, truncate_to)`
from disk if present; otherwise it runs `capture_pytorch_ref` and saves to disk.
Saves ~3 min of PT model load + forward when iterating on a specific layer's
FHE accuracy.
