# FHE LLaMA-3 PPL Evaluation Plan

> Goal: measure byte-level and word-level perplexity of the existing end-to-end
> FHE LLaMA-3.1-8B pipeline (the same pipeline producing the MRPC results in
> `results_summary.md`), so we can publish a single FHE-PPL number alongside
> PyTorch fp16 PPL on the same corpus + same tokenizer, with negligible
> additional pipeline engineering.
>
> The FHE forward IS the test. We never re-do FHE work; we just stop throwing
> away the per-position hidden state at the end of each layer-31 forward.
>
> Repo references throughout are to the modular reference tree at
> `/home/yongwoo-oh/q32-cleanup/python/llm_project/` (branch `cleanup-q32 @
> 8d7dd90`), which is byte-equivalent to the monolith at
> `/home/yongwoo-oh/phantom-fhe/python/llm_project/llama3_mrpc.py` (branch
> `quant-64bit @ 1dbf442`).

---

## 1. Corpus

### Primary: WikiText-2 raw (`wikitext-2-raw-v1`, `test` split)

- HuggingFace: `load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
  split="test")` (community mirror; the older `wikitext` namespace works too).
- Standard LM-eval citation: same corpus used by GPT-2, GPT-Neo, LLaMA-2,
  LLaMA-3 papers. Direct apples-to-apples with published HuggingFace eval
  numbers.
- Approximate sizes (raw test split, decompressed):
  - **Raw bytes**: ~1.27 MB (1,270,947 bytes).
  - **Words** (whitespace split, excluding empty lines): ~245k.
  - **LLaMA-3 BPE tokens** (concatenated, after `tokenizer(corpus_text)`):
    ~279k tokens (the LLaMA-3 tokenizer is a tiktoken-style BPE with vocab
    128256; tokens/byte ≈ 0.22, tokens/word ≈ 1.14).
- We use a **strided-window evaluation** (HF-standard, see HuggingFace "Perplexity
  of fixed-length models" tutorial). With max-context `M` and stride `S`:
  - sliding windows of size `M`; first `M-S` tokens of each non-initial window
    are context-only (loss masked); only the trailing `S` tokens contribute to
    the loss.
  - Number of windows ≈ `ceil((T - M) / S) + 1`.
- For FHE we cannot afford `M=2048` per window (each forward is ~370 s on
  int64). The proposal is **`M = 64`, `S = 32`** — a much smaller window
  than language-modeling convention, but the FHE pipeline's current
  attention path is correctness-validated at `num_tokens ≤ 128` against
  pytorch_ref (`results_summary.md` per-layer drift table is on 44-token MRPC
  prompts). `M=64,S=32` keeps us inside the validated regime and yields
  ~8,700 scoring tokens per window-pass over a sub-corpus.

### Secondary: Penn Treebank (`ptb_text_only`, `test` split)

- HuggingFace: `load_dataset("ptb_text_only", split="test")`.
- ~82k whitespace-tokens, ~451k bytes after concatenation.
- Faster, smaller; used as a cross-check that PPL trends agree across two
  corpora. Same byte/word formulas, same M/S.

### Out of scope

- WikiText-103 / C4: an order of magnitude more text than we can FHE-score in
  our budget. PT-only baselines on these are fine; FHE-side we stick to
  WikiText-2.
- PG-19: long-context corpus; doesn't fit our `M=64` window without invalidating
  the metric's comparability.

### Token budget (FHE-scored)

For each window we score the trailing `S=32` tokens.

- Pilot run: 32 windows = 1,024 scored tokens. ~3.3 h on int32 (32 × 370 s),
  ~6.0 h on int64.
- Full run: 256 windows = 8,192 scored tokens. ~26 h on int32, ~48 h on int64.
- (Stretch) 1,024 windows = ~32k tokens. Out of practical reach for int64.

---

## 2. PPL definitions (byte / word)

Let the corpus consist of a token sequence `t_1, …, t_N`. The cross-entropy
under the model is

```
  H = (1 / N) Σ_{i=1..N}  -log P(t_i | t_<i)
```

with `log = ln` (natural log). The classical token-level perplexity is

```
  PPL_token = exp(H)                              [unitless, per-token]
```

For corpora where the tokenizer differs across models, byte-PPL and
word-PPL renormalize H by the corpus length in bytes or words rather than
tokens, so models with different BPE inventories become comparable. Let
`B = total UTF-8 bytes in the corpus text` and
`W = total whitespace-words in the corpus text`. Then

```
  PPL_byte = exp( (N / B) · H )                   [equivalently exp(-1/B · Σ log P)]
  PPL_word = exp( (N / W) · H )                   [equivalently exp(-1/W · Σ log P)]
```

i.e. PPL_byte and PPL_word are simple unit conversions of PPL_token by
the ratios N/B and N/W respectively.

Two further conventions we lock down:

1. **Loss-masked first window prefix.** Following the HF stride-window
   recipe, the first `M-S` tokens of every non-initial window are
   context-only and contribute neither to N nor to the H sum (they are
   already counted by the previous window). The first window's full `M`
   tokens DO count.
2. **Token boundary alignment.** N, B, W are computed over the *scored*
   token region only (the set of `(window, position)` pairs whose
   log-probability we actually sum). B is the UTF-8 byte length of
   `tokenizer.decode(scored_tokens)`, W is the whitespace-word count of the
   same decoded substring. We DO NOT compute B/W on the raw corpus text and
   N on tokens separately — that would inflate B/W when scored tokens are a
   strict subset.

Sanity invariants to assert in the eval script:

- `N == sum_over_windows(scoring_token_count_in_window)`
- `N/B` and `N/W` are ~constant across runs (only depends on tokenizer +
  corpus), independent of pipeline correctness.
- `exp(H) == PPL_token` (no off-by-one between sum-log-P and the
  per-token reduction).

---

## 3. FHE measurement methodology

### 3.1 Where the next-token log-prob comes from

Read out at the SAME plaintext-FHE boundary the MRPC pipeline already uses
(`blocks/lm_head.py:37-52`). The hidden state at layer 31, post-residual2 is
*already* fully present in `y_ct` (the full `T_MODEL · D_MODEL` slot
ciphertext) and is *already* decrypted on the host every layer for the per-layer
drift log (`decoder_layer.py:653-656`):

```python
y_full = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, y_ct)),
                  dtype=np.float64)
...
y_p = y_full[::T_MODEL][:D_MODEL]   # CURRENT: extracts ONLY position P
```

For PPL we keep the same decrypt but slice ALL `num_tokens` positions instead
of just `P`. Layout from `llama3.py:46-62`: `D_MODEL=4096`, `T_MODEL=8`,
`NUM_SLOTS=32768`, slot[`i*T_MODEL + d`] = `x[i, d]` (stride-T_MODEL token
major). So:

```python
y_per_pos = np.stack(
    [y_full[i*T_MODEL : i*T_MODEL + D_MODEL] for i in range(num_tokens)],
    axis=0)                                        # (num_tokens, D_MODEL)
```

`y_per_pos` is byte-for-byte the all-position hidden state at the end of
layer 31 — the same tensor the MRPC pipeline computed but threw away every
example. **Zero change to the FHE pipeline**; we are only changing what the
host does with the decrypted output of the existing final layer.

### 3.2 LM-head readout strategy

**Recommendation: option (a) — full-vocab projection in PLAINTEXT, on the
host, after decrypt.**

Rationale (deciding against (b) "partial-vocab in FHE"):
- The full `lm_head` is `(VOCAB=128256, D_MODEL=4096)` fp32 ≈ 2.0 GB. It
  is a public model parameter (already extracted to
  `/tmp/llama_probe_full/lm_head.npy` by
  `extract_llama_probe.py:46-50`).
- The MRPC pipeline already runs the LM head's final-RMSNorm + matvec OUT
  of FHE for exactly this reason (see the docstring at
  `blocks/lm_head.py:1-26`: "doing the matvec in plaintext leaks
  nothing about the prompt. Only the final 2 scalar logits cross the trust
  boundary"). PPL needs `VOCAB` logits per position instead of 2 — same
  threat-model story, just larger numpy output.
- Partial-vocab-in-FHE would require either a top-K candidate set per
  position (requires the LM-head matvec in FHE to figure out which K, i.e.
  full-vocab anyway) or a fixed K from an external prior, which is what
  evaluation harnesses explicitly avoid because it biases PPL.
- Option (a)'s host cost is one (`num_tokens × D_MODEL`) @ (`VOCAB ×
  D_MODEL`).T matmul + a per-row log_softmax: ~10 ms per window at
  `num_tokens=64` in numpy fp64. Negligible vs the ~370 s FHE forward.

The readout (`blocks/lm_head.py` generalized):

```python
def all_position_logits_np(y_per_pos, final_norm_g, lm_head_full, eps):
    # y_per_pos: (num_tokens, D_MODEL)
    # lm_head_full: (VOCAB, D_MODEL) fp32 (load once)
    # final_norm_g: (D_MODEL,)
    rms = np.sqrt((y_per_pos**2).mean(axis=1, keepdims=True) + eps)
    y_norm = (y_per_pos / rms) * final_norm_g                  # (T, D_MODEL)
    logits = y_norm @ lm_head_full.T                           # (T, VOCAB)
    return logits.astype(np.float64)
```

Then per-position log-prob of the actual next token:

```python
def next_token_logprobs(logits, token_ids):
    # logits: (T, VOCAB), token_ids: list[int] length T (the window's tokens)
    # Standard LM convention: prediction at position i is for token_ids[i+1].
    # The window has T positions; the last position has no "next" to predict.
    # Drop logits[-1] and token_ids[0].
    shift_logits = logits[:-1]                                 # (T-1, VOCAB)
    shift_targets = np.asarray(token_ids[1:], dtype=np.int64)  # (T-1,)
    log_probs = shift_logits - np.log(np.sum(np.exp(shift_logits -
                shift_logits.max(axis=1, keepdims=True)), axis=1, keepdims=True)
                ) - shift_logits.max(axis=1, keepdims=True)
    return log_probs[np.arange(len(shift_targets)), shift_targets]   # (T-1,)
```

(Numerically stable log-softmax as written; in code use `scipy.special.logsumexp`
or `torch.nn.functional.log_softmax` for safety.)

For the strided window we then take only the last `S` of those `T-1`
log-probs as "scored" (the first `M-S-1` are context-only and were scored
by the previous window, except in the very first window where all `T-1`
are scored).

### 3.3 Per-token vs single-position cost

The MRPC pipeline already runs ONE FHE forward per sequence and reads ONE
position's logit. For PPL we run ONE FHE forward per WINDOW and read `S`
positions' log-probs from that single decrypt.

This means **one FHE forward yields `S=32` scored tokens**, not 1. The
amortized cost per scored token is `370 s / 32 ≈ 11.6 s` on int64 (vs the
MRPC numbers where the cost is `370 s` per scored token — a 32× efficiency
gain just from re-using the all-position output that the pipeline already
computes).

This single observation is what makes the project feasible.

Three small caveats to handle in code:

1. **K/V teacher-forcing is the validated regime.** The current pipeline
   (`decoder_layer.py:386-399`) builds `K_h`, `V_h`, and the RoPE-baked
   `Wq_baked` from `pytorch_ref[layer_idx]` (= the PyTorch reference's
   per-layer hidden state) every layer. The encrypted residual stream
   `x_ct` IS carried forward via AUTONOMOUS_FHE=1 (loop in
   `decoder_layer.py:330-345`), but K/V come from clean numpy. That's the
   "reference-guided" regime documented in CLAUDE memory
   (`project_mrpc_fhe_is_reference_guided.md`). For PPL we use the same
   regime — the FHE drift we measure is the residual-stream drift, exactly
   what `results_summary.md`'s per-layer drift table reports. Bridgeless
   K/V is a separate future project; not in scope here.

2. **PT reference must also score all positions.** `pytorch_ref` already
   captures all-position hidden states (`pytorch_ref.py:21-53`,
   `output_hidden_states=True`), so the necessary PT data is already on
   disk. Just keep the PT logits at every position, not only `[-1]`.

3. **Variable / fixed num_tokens.** The MRPC pipeline supports `--fixed-nt`
   (`mrpc_sweep.py:115-128`) to pad to a constant N. For PPL we want
   `num_tokens = M = 64` for every window — equivalent to `--fixed-nt 64`
   with the "real" token count also being 64 (no EOS padding). This means
   the existing fixed-nt code path is the right hook.

### 3.4 How many sequences / windows

Recommendation:
- **Pilot**: 32 windows from WikiText-2 test (covers ~2 kB of text). Used
  to validate the harness end-to-end and produce a first PPL point estimate
  (95% CI half-width on 1024 log-probs ≈ ±0.3 nats for typical LLaMA
  H ≈ 2.5 nats/token — coarse, but enough to confirm the pipeline isn't
  producing nonsense).
- **Production**: 256 windows = 8,192 scored tokens. 95% CI half-width
  ≈ ±0.1 nats → PPL CI ≈ ±10% on PPL ≈ 12. Adequate for a paper.
- **Stretch (only if int64 sweep finishes ahead of schedule)**: 512 windows
  = 16,384 scored tokens.

---

## 4. GPU time estimates

Per-layer warm time from `results_summary.md`:

| quant | warm s/seq (MRPC, mostly nt≈70) |
|------:|---------------------------------:|
| int8  | 147 |
| int16 | 141 |
| int32 | 216 |
| int64 | 369 |

For PPL at `num_tokens = M = 64`, runtime is mostly insensitive to N over
[44, 64] (the MRPC range has mean ~67 from `mrpc_pytorch_raw.npz`
prompt_lengths; the per-layer compute is dominated by D-dimensional matmuls
which are N-independent, and the only N-linear cost is QK^T + softmax,
which is small relative to MLPs). Conservative assumption: same warm s/seq.

Pilot (32 windows) ETAs:

| quant | wall time |
|------:|----------:|
| int8  | ~1.3 h |
| int16 | ~1.3 h |
| int32 | ~1.9 h |
| int64 | ~3.3 h |

Production (256 windows) ETAs:

| quant | wall time |
|------:|----------:|
| int8  | ~10.5 h |
| int16 | ~10.0 h |
| int32 | ~15.4 h |
| int64 | ~26.2 h |

The single-stream serial assumption matches `resume.sh`'s chunked driver
behavior (we are not introducing the parallel path here).

---

## 5. Code changes (concrete file list + sketches)

All new / modified paths below are on the modular reference tree
(`/home/yongwoo-oh/q32-cleanup/python/llm_project/`). The same edits apply
to the monolith (`/home/yongwoo-oh/phantom-fhe/python/llm_project/llama3_mrpc.py`)
by patching the corresponding regions inside the single file.

### 5.1 New: `python/llm_project/ppl_eval.py`

Driver, analogous to `mrpc_sweep.py` but reads all-position log-probs.
Key responsibilities:

- Load WikiText-2 raw test split + LLaMA-3 tokenizer once.
- Build the stride-window iterator: yields `(window_idx, token_ids[M],
  scoring_mask[M])` where the mask is True only for positions whose
  log-prob we will sum into H.
- Build the shared CKKS engine ONCE (`setup_engine(...)`,
  `llama3_mrpc.py:122`), mirror the MRPC sweep's engine-reuse pattern.
- Build `lm_head_full` once (load `/tmp/llama_probe_full/lm_head.npy`
  → `(128256, 4096)` fp32 → keep in RAM, ~2 GB).
- For each window:
  1. Run PyTorch reference via `capture_pytorch_ref(token_ids)`
     (`pytorch_ref.py:56-74`), get `pytorch_ref` (all 33 hidden states,
     all 64 positions) and `pytorch_pre_norm`.
  2. Call `run_classifier_fhe_all_positions(...)` (new wrapper, see §5.3) →
     `y_per_pos` of shape `(M, D_MODEL)`.
  3. Compute `all_position_logits_np(y_per_pos, final_norm_g,
     lm_head_full, eps)` → `(M, VOCAB)`.
  4. Compute PT logits the same way using
     `all_position_logits_np(pytorch_pre_norm, ...)` — note PT pre_norm is
     pre-final-norm, so we apply the SAME `final_norm_g` + matvec on host
     for a clean PT/FHE diff at the LM-head boundary.
  5. Compute scored next-token log-probs via §3.2's `next_token_logprobs`
     restricted to `scoring_mask`.
  6. Append to a CSV: `window_idx, position, token_id, target_id, ll_pt,
     ll_fhe, top1_pt, top1_fhe, agree_top1, time_sec`.
- After all windows, reduce: `H = -mean(ll)`, `PPL_token = exp(H)`,
  `B`, `W` from decoded scored region, `PPL_byte = exp(H * N/B)`,
  `PPL_word = exp(H * N/W)`. Same reduction for PT and FHE columns.
- CSV header similar to MRPC's at `mrpc_sweep.py:33-36`:

```python
CSV_HEADER = ["window_idx", "position", "token_id", "target_id",
              "ll_pt", "ll_fhe", "top1_pt", "top1_fhe",
              "agree_top1", "time_sec_per_window"]
```

### 5.2 New (small): `python/llm_project/blocks/lm_head_full.py`

Add `all_position_logits_np` and `next_token_logprobs` next to the
existing `yes_no_logits_np`. Re-use `rmsnorm_np` from
`blocks/lm_head.py:30-34`. This keeps the LM-head boundary in ONE place.

### 5.3 New tiny helper in `python/llm_project/llama3_mrpc.py`

Add `run_classifier_fhe_all_positions(...)` next to `run_classifier_fhe`.
~20 LOC. Sketch:

```python
def run_classifier_fhe_all_positions(
        num_tokens, pytorch_ref, pytorch_pre_norm,
        cos_all_full, sin_all_full, label="ppl_window",
        engine=None, preloaded_weights=None, precomputed_calib=None):
    """End-to-end FHE classifier returning ALL-POSITION hidden state
    at end of layer 31 (shape (num_tokens, D_MODEL)).

    Same body as run_classifier_fhe, but skips _run_lm_head and instead
    returns the full per-position decode of the last layer's y_ct.
    P_local is set to num_tokens-1 (preserves current calibration / RoPE
    matrix indexing; calibration is num_tokens-aware not P-aware in any
    way that matters here)."""
    P_local = num_tokens - 1
    cctx, NUM_DECODERS = _classifier_setup(
        num_tokens, P_local, pytorch_ref, pytorch_pre_norm,
        cos_all_full, sin_all_full, label,
        None, None, None,
        engine, preloaded_weights, precomputed_calib)
    layer_times = []
    y_ct_carry = None
    y_per_pos_last = None
    for layer_idx in range(NUM_DECODERS):
        y_p_fhe, y_ct_carry = run_decoder_layer(
            layer_idx, cctx, y_ct_carry, layer_times)
        if layer_idx == NUM_DECODERS - 1:
            # decoder_layer already decrypts y_ct (line 653) but throws away
            # all but slice [::T_MODEL][:D_MODEL]. Re-decrypt here to get the
            # full tensor. Cheaper alternative: have run_decoder_layer
            # return the full y_full when an opt-in flag is set; the
            # zero-touch version below decrypts twice on layer 31 only.
            y_full = np.array(cctx.encoder.decode_double_vector(
                cctx.ctx, cctx.sk.decrypt(cctx.ctx, y_ct_carry)),
                dtype=np.float64)
            T_MODEL = 8; D_MODEL = 4096
            y_per_pos_last = np.stack(
                [y_full[i*T_MODEL : i*T_MODEL + D_MODEL]
                 for i in range(num_tokens)], axis=0)
    return y_per_pos_last
```

(Optional micro-optimization: add an `expose_y_per_pos` flag to
`run_decoder_layer` so the layer-31 decrypt isn't done twice. Skipped in
the minimal proposal; +~300 ms/window on a ~370 s window is noise.)

### 5.4 Modifications to existing files

**`decoder_layer.py`** — None required for the minimal plan. (Optional
micro-opt above would add ~10 lines plumbing a flag through `ClassifierCtx`.)

**`llama3_mrpc.py`** — Add `run_classifier_fhe_all_positions` (§5.3) +
re-export `lm_head_full`'s helpers if convenient. The existing
`run_classifier_fhe` stays untouched (the MRPC sweep keeps working
byte-identically).

**`blocks/lm_head.py`** — No changes; new helpers live in
`blocks/lm_head_full.py` so we don't bloat the existing module.

**`mrpc_sweep.py`** — No changes. PPL is a separate driver.

### 5.5 Driver shell wrapper

`/home/yongwoo-oh/mrpc_campaign/run_ppl.sh` analogous to `resume.sh`,
chunked + resumable, invokes:

```bash
AUTONOMOUS_FHE=1 USE_BOOTSTRAP_17=1 \
python /home/yongwoo-oh/phantom-fhe/python/llm_project/ppl_eval.py \
    --corpus wikitext2 --window-size 64 --stride 32 \
    --num-windows 32 --csv /home/yongwoo-oh/mrpc_campaign/ppl_int32.csv
```

---

## 6. Verification protocol

Goal: be sure the FHE-PPL number is not poisoned by a bug in the new
all-position decode / new LM-head readout.

### 6.1 Tier-0 (cheap, ~minutes, CPU + GPU-PT only)

A pure-PyTorch reference run:

```python
# scripts/ppl_pt_reference.py
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Meta-Llama-3.1-8B", torch_dtype=torch.float16)
# strided-window with M=64, S=32 over WikiText-2 test
# numerator H_pt, denominators B,W as in §2
# emit PPL_token_pt, PPL_byte_pt, PPL_word_pt
```

Expected: PPL_token_pt for LLaMA-3.1-8B on WikiText-2 with full M=2048 is
6.4 (published HuggingFace number); with the truncated M=64 it will be
substantially HIGHER (less context → worse predictions). We expect M=64
PT PPL in the range **[20, 50]** based on standard sliding-window PPL
degradation curves for autoregressive LMs. The exact PT number IS our
reference; the FHE number is then judged against it.

### 6.2 Tier-1 (consistency of new LM-head readout)

Run our new `all_position_logits_np` on `pytorch_pre_norm` (which is the
PT pre-final-norm hidden state, already cached) and compare against the
HuggingFace model's logits at the same positions. Expected: max abs diff
< 1e-3 (fp16 vs fp64 arithmetic on the final RMSNorm + matvec). If this
fails, the bug is in our LM-head harness, not in FHE.

### 6.3 Tier-2 (FHE-vs-PT logit diff matches MRPC noise floor)

For each scored position, compute `Δ_log_p = ll_fhe - ll_pt`. From
`results_summary.md`:

- mean `|Δyes|` = 0.141, mean `|Δno|` = 0.108 across all 408 MRPC
  examples (int32, identical at int64 to 3 decimals).
- This is the |Δ| on a SINGLE logit; for the next-token log-prob
  `ll = logit_target - logsumexp(all logits)`, the dominant noise term is
  `Δ_logit_target` since `logsumexp` is contractive and target-independent
  noise averages out. So we EXPECT:

```
mean |Δ_ll| ≈ 0.14   (in nats)
mean |Δ_H|  ≈ 0.14 / sqrt(N_scored) → << 0.14 at N=8192
PPL_FHE / PPL_PT  ≈ exp(±0.14)  ≈ [0.87, 1.15]   at single-position
PPL_FHE / PPL_PT  ≈ exp(±0.0015) ≈ [0.998, 1.002] in aggregate at N=8192
```

i.e. the **FHE-PT PPL gap should be within ±1% on the production
(256-window) run**. If we see >10%, something is wrong with the new
all-position decode (e.g. T_MODEL slot layout mis-indexed, or LM-head
applied at the wrong chain index).

### 6.4 Tier-3 (top-1 agreement spot check)

For each scored position, log `top1_pt = argmax(logits_pt)` and `top1_fhe`.
Expected: top-1 agreement ≥ 95% (MRPC top-1 was 100% on a 2-class slice
where the margin was ~30× the noise; on full-vocab the margin between
top-1 and top-2 is much smaller, so 100% agreement is implausible — but
the relative entropy under FHE should be very close to PT). If <80%, the
ordering of logits is being scrambled, again pointing at a decode/layout
bug rather than a calibration drift.

### 6.5 Sanity asserts in the driver

- `len(y_per_pos) == num_tokens`.
- `np.isfinite(y_per_pos).all()`.
- `logits.shape == (num_tokens, 128256)`.
- `||y_per_pos[-1] - y_p_legacy_decode|| < 1e-9` where `y_p_legacy_decode`
  is the existing `y_full[::T_MODEL][:D_MODEL]` slice — proves the new
  all-position decode reduces to the existing single-position decode at
  `i = P_local`.

---

## 7. Run schedule (post-MRPC-campaign)

Current state (per `results_summary.md`, 2026-05-29):

- int64 sweep PAUSED at row 319/408. Resume time ~5 h (319 → 408 at warm
  ~370 s/example).
- int32 sweep COMPLETE.
- int8/int16: complete on accuracy/F1; raw CSVs lost to the 2026-05-26
  reboot but aggregate numbers preserved.

### 7.1 Critical path

```
NOW                  → int64 MRPC resume               5 h
+5 h                 → int32 PPL pilot (32 windows)    2 h    (validation run)
+7 h                 → review pilot, fix bugs           ad hoc
+ <fix delta>        → int32 PPL production            15.4 h
+ ~24 h              → review int32 PPL                ad hoc
+ <fix delta>        → int64 PPL production            26.2 h
+ ~50 h              → write paper section             ad hoc
```

Total wall: **~3 days end-to-end** assuming pilot passes.

### 7.2 Which quants to PPL-score

Recommend **int32 + int64 only**.
- int8 results lost the raw CSV; would need a re-MRPC-run to anchor the
  Δ-logit calibration that Tier-2 verification depends on.
- int16 is bit-identical to int32 in our table (mean |Δ| = 0.141 for both,
  to 3 decimals); the PPL difference will be in the noise.
- int32 = the headline number ("32-bit lossless plaintext quant, n=408
  MRPC 100% PT agreement, ~$PPL on WikiText-2"). int64 = the
  reference ("verified at higher precision; PPL within ε").

If time permits after int32+int64, add int16 production (10 h) for the
full-stack table.

### 7.3 Stop conditions

- Tier-2 fails (FHE PPL > 10% from PT) → STOP, investigate decode bug
  before scaling N.
- Pilot ETA on int32 > 4h → STOP, investigate (warm should be 2h).
- Out-of-memory on the LM-head matvec → switch to chunked matmul (`for
  v0 in range(0, VOCAB, 8192)`).

---

## 8. Open questions for the user

- **Stride-window M and S**: proposal is `M=64, S=32`. The MRPC pipeline's
  per-layer drift table (`results_summary.md`) is on prompts with mean
  ~67 tokens; staying inside that regime is the safe choice, but is the
  user willing to accept a *non-standard* PPL window size (most LLaMA PPL
  numbers in papers are `M=2048` or `M=4096`)? If yes, accept the
  proposal. If no, we need to validate the FHE pipeline at `M=512` or
  larger first (separate ~1-day project; the `--fixed-nt 512` code path
  exists in `mrpc_sweep.py` but was never PPL-validated, and CLAUDE memory
  notes "nt=512 loss is structural softmax NOT calibration").

- **Corpus**: WikiText-2 raw test as primary, PTB as secondary. Both
  are HF-loadable, both standard. Any preference for a different corpus
  (C4 small subset, custom domain text)?

- **Quants to score**: int32 + int64 (production), int16 optional. OK?

- **Should the LM head move INTO FHE later?** The current proposal does
  the full-vocab matvec in plaintext on the host (faithful to the existing
  MRPC pipeline's threat model — public LM head, only the resulting
  log-prob crosses the trust boundary). A future "everything in FHE" PPL
  number would require an in-FHE LM head + an in-FHE log_softmax (the
  latter is the structural-softmax problem CLAUDE memory references, at
  VOCAB=128256). Out of scope for this plan, but flag it so the user
  knows.

- **Where does the corpus selection live in the repo?** Proposed:
  cache `/tmp/wikitext2_test_tokenized.npz` analogous to
  `/tmp/mrpc_ptref_idx*.npz`. Confirm the HF_HUB_OFFLINE workflow
  documented in CLAUDE memory (`project_use17_mrpc_env_and_verification.md`)
  is OK for a one-time corpus download.

- **Driver: serial or parallel?** MRPC has both `mrpc_sweep.py` (serial)
  and `mrpc_sweep_parallel.py` (multi-worker). PPL pilot uses serial only;
  is the user OK skipping the parallel path? (The cost saving is small at
  256 windows.)
