# Design rationale: `decoder_layer.py`

Migrated FHE (CKKS) design-rationale for the single decoder-layer forward pass of the
LLaMA-3.1-8B inference pipeline (`python/llm_project/decoder_layer.py`).

The code module keeps only API-contract one-liners, section markers, and operationally
necessary NOTE/WARNING comments. Every multi-line "why" block — cryptographic reasoning,
constant justification, slot-layout math, failure-mode analysis, and tuning history — was
moved here and is referenced from the code with `# design: doc/design/decoder_layer.md#<anchor>`
pointer lines at the original indentation.

---

## module-overview

Single decoder-layer FHE forward + per-call shared-state dataclass, split out of
`llama3_mrpc.py` (Phase 3).

Public surface:

  - `ClassifierCtx`     : `@dataclass` bundling per-call state (engine handles,
                        bootstrap plan, weight-subset preload, RoPE tables,
                        LM head data, diagnostic toggles).
  - `run_decoder_layer` : run ONE decoder layer (rms1 -> attn -> resid -> rms2
                        -> mlp -> resid -> decrypt+log). Returns
                        `(y_p, y_ct_carry)`. `y_p` is the decrypted hidden at
                        the query position; `y_ct_carry` is the post-residual2
                        ciphertext (consumed by `AUTONOMOUS_FHE=1`; discarded
                        otherwise).

Behavior is byte-for-byte identical to the Phase-2 in-file helper
`_run_decoder_layer`: every diagnostic print, every `sk.decrypt` probe, every
bootstrap is preserved in the exact same order. `AUTONOMOUS_FHE` `y_ct` carry
semantics are unchanged.

---

## classifierctx-state

Per-call state shared across decoder layers + LM head.

Bundles every value the per-layer body reads from the orchestrator's
closure: call args (num_tokens, RoPE tables, references), engine handles
(ctx/encoder/sk/keys/fresh chain index), the bootstrap-placement plan,
the per-example weight-subset preload, and the LM head plaintext
weights. Lets `run_decoder_layer` take 4 args (layer_idx, cctx, y_ct_carry,
layer_times) instead of the original 18+.

---

## run-decoder-layer-contract

Run ONE decoder layer end-to-end. Mirrors the original
per-layer loop body BYTE-FOR-BYTE except that:

  - the loop var `layer_idx` and the cross-layer `_y_ct_carry` are passed
    in as args (and the new carry is returned)
  - `layer_times.append` mutates the caller-supplied list (the print
    sequence is unchanged)

Returns `(y_p, y_ct_for_carry)`. `y_p` is the decrypted hidden at the query
position; `y_ct_for_carry` is the post-residual2 ciphertext (only used
when `AUTONOMOUS_FHE=1` — otherwise the caller discards it).

---

## prefetch-next-layer

1-layer-ahead prefetch (latency-only; never a correctness dep).

The 32-layer warm run is I/O-bound: each layer cold-reads ~1.6 GB of
IRP SCP blobs + ~1.4 GB of fp64 weights from a 57 GB cache (> RAM →
page-cache thrash), adding ~4.6 s/layer of disk read on top of ~5.9 s
compute. Background-LOAD the NEXT layer's 5 IRP blobs + numpy weights
(encoder-FREE: mmap + scp_from_bytes + np.load only — the encoder's
expand happens later, per-matvec, ON THIS MAIN THREAD) so the next
layer's read overlaps this layer's GPU compute. The wrappers
(`*_plaintexts_cached` / `get_layer_weights`) await the pending future or
fall back to a synchronous load on a RAM miss, so a skipped/failed
prefetch only costs latency, never correctness. Cold MISS (no blob)
is left to the synchronous encode path — never threaded.

Prefetch constants mirror the actual call sites exactly:

  - Wq/Wo: d=D_TOTAL(=4096), baby_steps=16 (fhe_attention_dense_full)
  - MLP  : d_in=D_MODEL, d_out=16384, baby_steps=_BABY_STEPS_IRP_MLP_RECT=16

---

## no-numpy-mlp-prefetch

NOTE: no numpy prefetch for (Wgate, Wup, Wdown) — those are now
LAZY (passed as 0-arg loaders to the IRP cache wrappers). On a
warm run the SCP blobs hit and the loaders never fire, so a
background numpy prefetch would only re-introduce the ~1.4 GB/
layer of WASTED I/O this change exists to eliminate. The
IRP-SCP prefetch above (`prefetch_layer`) is what overlaps the
real reads with compute.

---

## evict-older-layers

Trim RAM entries for layers older than the current one (the LRU bound
already caps memory; this keeps only current + next layer resident).

---

## probe-dump-layer-tag

DIAGNOSTIC ONLY: tag stage dumps with the current layer so the
offline rel-RMS harness can pick the right file. No-op when
`PROBE_DECRYPT_STAGES` is unset (verbose is also False here normally).

---

## per-layer-weights-subset

Per-layer weights (the {Wq,Wk,Wv,g1,g2} subset preloaded above).
The dense path reloads the big R_P-indep matrices
(Wo/Wgate/Wup/Wdown) per-layer inside the attention / MLP blocks.

---

## per-layer-calibration

Per-layer rmsnorm + bootstrap calibration (num_tokens-aware).
When `precomputed_calib` is supplied (parallel sweep), skip the
per-example shadow forward pass entirely — calib was precomputed
once at startup using a representative example, which also lets
the worker preload drop the big Wo/Wgate/Wup/Wdown matrices
(~45 GB across 32 layers) since the per-example hot path only
touches Wq/Wk/Wv/g1/g2 directly.

---

## calib-disk-cache

Disk-cached calibration: `load_layer_weights()` pulls the full
~1.4 GB weight dict (Wo/Wgate/Wup/Wdown) purely to run the numpy
shadow forward, then discards it. The (z1,z2,max_abs) output is
deterministic in (x_btd, layer, num_tokens, query_position), so
cache it — warm runs skip both the weight load and the forward.

`layer_weights[layer_idx]` is the subset; reload the full dict
per-layer and drop after calib so the heap doesn't grow to
60 GB across the 32-layer loop.

---

## silu-domain-margin

Tightened `silu_domain` margin 1.2 → 1.05 to narrow the Chebyshev fit
range. Narrower domain → smaller polynomial coefficients → less
CKKS noise amplification through Clenshaw recurrence. Safe because
`max_abs_calib` already includes `BOOT_CALIB_MARGIN`=1.5× over actual
numpy-predicted max; the additional 1.05× covers FHE noise on gate.

---

## silu-degree-search

Use NORMALIZED monomial fit when an adaptive degree <= 20 meets
the error threshold (~1e-3); falls back to the deg=32 Chebyshev
Clenshaw path otherwise. Clenshaw adds 2 extra bootstraps + ~30
ct-ct multiplies (~840ms/layer), so prefer `eval_polynomial` when
the simpler path's accuracy is comparable.

Test degrees up to 20 (PS depth 5; +1 for normalization = 6 levels).
deg=24 with normalized coeffs has c_top ~ 8e4 (encoded ~9e16, within
prime 2^60 ≈ 1.15e18 but apparently triggers a slow path in Phantom's
`eval_polynomial` — observed to hang on L31 silu). deg=28+ even worse.
Higher degrees would also push PS depth to 6, busting chain budget.

**GPU-CONFIRMED (2026-06, quant-32bit idx-0): cap-20 is a chain-budget ceiling; mechanism corrected.**
Raising the cap to 24 or 27 busts the chain at **L1's post-MLP bootstrap**:
`ValueError: bootstrap: input at user_level 15 (== max_user_level 15) requires scaling`
— NOT the eval_polynomial hang theorised above. L0 picks deg≤20 and runs clean; L1
picks deg>20, which consumes **one extra multiplicative level** in Phantom's actual
`eval_polynomial`, pushing `mlp_out` to ul15 (== max) with no level left for the
post-MLP bootstrap's pre-scale. The textbook PS-depth formula `ceil(log2(deg+1))`
reports depth-5 for ALL of deg 20–27 and is blind to this +1 — so the "PS depth 6"
above was the wrong mechanism but the right conclusion. cap-24 fails identically to cap-27.

The precision gain is **real but blocked**. Numpy fit (bits = −log2 Linf, post-SCALE
quantization): silu_max=3.5 → deg-27 = 27.97 b vs deg-14 = 14.87 b (+13.1); silu_max=4.0
→ deg-27 = 24.74 b (+11.7). c_top stays small at these (small-domain) layers (16.4 at
silu_max=3.5; the "~8e4" above is a wide-domain figure). To unlock deg>20 you must FREE
one multiplicative level in the MLP segment (gate/up → silu → Wdown → mlp_out leaves
exactly 1 level of slack that deg-20 fills): bootstrap mlp_out one level earlier, trim a
level in gate/up/Wdown, or raise max_user_level — a coupled change needing its own FHE
validation, and it competes with bootstrap-reduction for the same MLP-segment levels.
Until then the adaptive cap stays at **20** (current code).

---

## silu-clenshaw-dispatch

Opt 2: dispatch `silu_clenshaw` only when the adaptive winner is
still over the error budget. The deg=32 Chebyshev BASIS path
(Clenshaw) bounds intermediates by max|t_k| ~ silu_max — needed
when the normalized poly fit can't hit ~1e-3 Linf at deg <= 20.
Threshold = 5e-3 (matches the error budget the existing pipeline
tolerates at deg=32 Clenshaw on wide silu domains).

Force Clenshaw at high-magnitude layers (L=30/31 in LLaMA-3.1-8B,
silu_max ≥ 6 there). With NSQ=6, accumulated softmax-path drift
can push some gate slot past the ±1.2·silu_max cushion at those
layers — deg-20 monomial extrapolation past domain is catastrophic
(silu(1.2D=9.5)≈-60 vs true 9.5), causing the L=30 cascade blowup
to 150k+ observed in NSQ=6 sweeps. Clenshaw with deg-32 Chebyshev
basis bounds intermediates by max|t_k| and stays bounded outside
the fit domain. Cost: +2 bootstraps + ~840ms on the dispatched
layers (only 2 of 32) → negligible at the layer-sweep scale.

---

## clenshaw-deg-32

Clenshaw deg=32 is sufficient; deg=48 gives no measurable
accuracy gain (verified idx=6: identical max|err| at L=30/31,
confirming silu fit error ≪ accumulated CKKS noise floor).

---

## encrypt-inputs-multi

Encrypt inputs (multi-ct K, V). K/V/c_per_head always derive
from the clean numpy x_btd (= pytorch_ref[layer_idx]); only the
encrypted query residual x_ct is carried in autonomous mode.

---

## probe-dump-calib

DIAGNOSTIC ONLY (`PROBE_DECRYPT_STAGES=1`): dump the EXACT live
c_per_head + safety_scale the FHE pipeline uses, so the offline
harness compares decrypted intermediates against the SAME centering
the ciphertext actually got. No-op when the flag is unset.

---

## autonomous-fhe-carry

Mirror reverted 625ea9c: discard the freshly-encrypted x_ct
(from pytorch_ref[layer_idx]) and feed the previous layer's
output ciphertext forward instead. 625ea9c called
`engine.bootstrap_inplace(y_ct)` directly to refresh it to a
fresh level; the SK-free equivalent here is bootstrap
with the same x_in calibration the pipeline already trusts
for x_ct (line below mirrors the boot_before["rms1"] site).
This restores y_ct to the freshest chain index / SCALE that
`encrypt_layer_inputs_multi` produces; the stride-T_MODEL slot
layout is already preserved by the decoder pipeline (same
layout that `y_full[::T_MODEL][:D_MODEL]` decodes).

---

## dense-attention-block

Dense token-major attention (THE compute path — IRP deleted).
QK^T + softmax + score·V + BSGS Wo, all FHE.
`fhe_attention_dense_full` runs the complete dense block
internally; its `_f_out` is the layer's attention output feeding
residual1. Teacher-forced Q/K/V mirror `encrypt_layer_inputs_multi`
exactly (same numpy x_btd / weights / rope). Wo / (Wv via w) are
R_P-independent; Wo is NOT in the per-example hot subset and the
shared cache may hold only the 5-key subset (`w["Wo"]` -> KeyError),
so load Wo directly off disk via the subset loader (one
(4096,4096) fp64 array). Layout bridge: dense BSGS Wo output is
replicated-block period-D_TOTAL; the residual stream / x_ct is
stride-T_MODEL (slot[i*T_MODEL]==x[i], cf.
`encrypt_layer_inputs_multi`). Re-encode into stride-T_MODEL and
re-encrypt at fresh_ci — the SAME teacher-forcing layout bridge
the per-layer pipeline already applies to its inputs each layer.
The `Layer {L}` rel-RMS vs pytorch_ref[L+1] below is the
validation metric.

---

## bridge2-bridgeless-attn

BRIDGE 2 (bridgeless): o_ct is already stride-T_MODEL, bootstrap-
refreshed — use directly for residual1, no decrypt→re-encrypt.

---

## irp-rect-mlp-overview

IRP-rect MLP (Cachemir §4.1 rect host).
K_sq = d²/N = 512 per square sub-IRP, K_total = K_sq*α = 2048
SCPs per matmul — 8× fewer than dense BSGS's 16384. NO bridges:
stride-T_MODEL == IRP layout at d=D_MODEL (both slot[i*8]=x[i]),
so x_mid_norm enters and mlp_out exits in the same shape the
surrounding rmsnorm+residual already use.

`_BABY_STEPS_IRP_MLP_RECT = 16`  # M=16, G=32 for K_sq=512 (~sqrt)
`_D_PAD_OUT_MLP = 16384`  # D_HIDDEN=14336 padded to pow-2 multiple of D_MODEL (α=4)

---

## mlp-weights-lazy

Wgate/Wup/Wdown are R_P-independent and may not be in w's hot
subset (same situation as Wo). They are LAZY: the ~1.4 GB fp64 load +
zero-pad below is wrapped in 0-arg callables and passed straight to
the IRP cache wrappers. On a WARM run the SCP blobs hit on disk, the
wrappers return the cached SCPs WITHOUT calling the loader, and the
big numpy load/pad never fires — dropping ~1.4 GB/layer of wasted I/O
so the prefetch can finally hide the IRP-only reads. On a COLD MISS
the three loaders share ONE memoized subset load (so a triple-miss
reads the weights off disk once, not 3×).

Convention: `irp_matvec_rect_host` computes `y = x @ M`.
For Wgate/Wup we want gate = x @ Wgate.T → M = Wgate.T padded to
(D_MODEL, D_PAD_OUT) with trailing columns zero. Wdown.T padded to
(D_PAD_OUT, D_MODEL) with trailing rows zero.

---

## mlp-subset-cold-miss

COLD-MISS ONLY: load (Wgate, Wup, Wdown) once, memoized so the
three lazy loaders below share a single ~1.4 GB disk read.

---

## encode-irp-rect-scps

Encode IRP-rect SCPs. Wgate/Wup are WIDE (d_in=D_MODEL < d_out=D_PAD_OUT).
They are complex output-FOLDED (K/2 SCPs each): the folded matvec
emits a complex ct (real=out[:d/2], imag=out[d/2:]) at the folded dim
d_out_fold = D_PAD_OUT/2, split + interleave-recombined back to a real
interleaved-layout ct below. Wdown is TALL (d_in=D_PAD_OUT >
d_out=D_MODEL), UNFOLDED, with rows permuted to absorb the gate/up
interleave layout (interleave_output_order, applied in the cache
wrapper) → mlp_out comes out NATURAL order, no un-permute.

---

## bridge3-bridgeless-wdown

BRIDGE 3 (bridgeless): UNFOLDED row-permuted Wdown. Row permute at
full d_out=D_MODEL → natural stride-T_MODEL output, no SK bridge.

---

## lazy-level-gate-up

Lazy-level: drop the IRP-Wgate/Wup input to a deep chain so these two
rotation-heavy wide rect matvecs (D_MODEL×D_HIDDEN, K=2048 SCPs each —
the biggest matvecs in the model) run at few RNS limbs (cheap).
Headroom audit: gate AND up are refreshed together by ONE
`merge_bootstrap` right before silu (both land at user_level ~1, fresh),
so neither output has a deep downstream constraint. The matvec
consumes 2 levels (sub_mask + rescale), so targeting input user_level
11 leaves both outputs at user_level 13 going into the merge. After
the merge silu consumes ~7 levels (bootstrap pre-scale + Clenshaw) and
up stays fresh, so the post-silu multiply aligns to silu (~ul 7→8)
leaving h_ct at user_level ~9 — shallow enough that Wdown's lazy-level
mod_switch handles the rest without a dedicated h-boot. Both matvecs
consume the same input, so mod_switch a single shared deep copy.

---

## rect-irp-mask-convention

Rect-IRP mask convention: wide path needs
only sub_mask_pt; tall path needs BOTH sub_mask_pt (at chain+1)
AND input_mask_pt (at chain, encoded at d=d_out as square mask).
FOLDED wide rect path: the fold halves d_out (16384 → 8192) but the
path stays wide (d_in=D_MODEL=4096 < d_out_fold=8192). ct_in goes
directly into the matvec with mask_pt=sub_mask; the mask op fires
INSIDE the matvec at ct_in's chain (the lazy-leveled
`_x_mid_norm_deep`). The fold input layout == the unfolded wide input
layout (slot[i*t]=x[i], t=N/D_MODEL), so `_x_mid_norm_deep` enters
unchanged.

---

## folded-interleaved-matvec

Folded wide matvec → complex ct → split → interleave-recombine
→ real ct in interleaved (stride t_fold/2) layout. The fold +
extract adds ~+1 level vs the unfolded matvec; interleave_recombine
is 0 levels (1 rot + 1 add).

---

## silu-gate-layout-invariant

silu(gate): slot-wise, layout-invariant.
At high-magnitude layers (L30/31, silu_max>6) the harness
sets silu_t_coeffs/silu_D and the BOUNDED Chebyshev-basis
Clenshaw path is taken. The deg<=20 monomial silu_coeffs
catastrophically extrapolates past the ±1.2·silu_max fit
domain (silu(9.5)≈-60 vs ≈9.5), so this path MUST honor
silu_t_coeffs/silu_D too.

---

## merge-bootstrap-gate-up

Merge gate+up refresh into ONE bootstrap (same cost as one). This
makes `_up_ct` fresh too, so the post-silu multiply lands h_ct shallow
enough to skip the h-boot. max_abs must bound BOTH gate and up; both
bounds are taken raw (no double BOOT_CALIB_MARGIN) to match the
previous gate-boot which used max_abs=silu_max (raw gate).

---

## h-mul-chain-align

h = silu(gate) * up (chain alignment as before).

---

## h-boot-eliminated

h-boot eliminated: merge_bootstrap above made `_up_ct` fresh, so the
post-silu multiply (silu @ ~7 * up @ fresh) aligns to ~7, leaving
`_h_ct` shallow enough that Wdown's lazy-level mod_switch handles the
rest without a dedicated bootstrap.

---

## wdown-tall-matvec

Wdown (bridgeless unfolded tall rect IRP matvec, K=2048 SCPs).
BRIDGE 3: row-permuted unfolded matvec emits natural stride-T_MODEL
output → bootstrap refreshes chain (no SK bridge).
Tall path masks at FULL dims (d_in=D_PAD_OUT, d_out=D_MODEL):
sub_mask_pt (rect at chain+1), input_mask_pt (square at d=D_MODEL).
Lazy-level: unfolded tall = input_mask (+1) + sub_mask (+1) + rescale
(+1) = 3 levels. Target input user_level 11 → mlp_out at ~14 (< 15).

---

## mlp-out-bootstrap-mean-center

bootstrap replaces SK bridge. Mean-center before boot.

---

## residual2

residual2 (both operands natural stride-T_MODEL; residual aligns chain).

---

## decrypt-accuracy-check

Decrypt for accuracy check (vs pre-norm reference for L=31, vs pytorch_ref[L+1] for others).

---

## autonomous-fhe-carry-out

Carry the post-residual2 output ciphertext into the next
layer's x_ct (bootstrapped at the encrypt site above). The
decrypt above is logging-only here (true drift measurement),
exactly as in reverted 625ea9c.
