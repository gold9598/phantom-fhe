# Design rationale: `engine_setup.py`

Design-rationale notes migrated out of
`python/llm_project/engine_setup.py` (engine / user-step setup + per-layer
numpy calibration, split out of `llama3_mrpc.py`). The code module retains
one-line summaries and operational markers; every block below carries the
hard-won cryptographic-engineering reasoning that the compact module points
to via `# design: doc/design/engine_setup.md#<anchor>`.

This is part of an FHE (CKKS) LLaMA-3.1-8B inference pipeline.

---

## module-contents

The module contains:

  - `_make_rms_params_local`   : nested rmsnorm_params builder
  - `_real_nt`                 : real (non-padded) token count helper
  - `_sim_pre_finsmx_mean`     : numpy-side softmax pre-finalize mean simulator
  - `compute_layer_calib_n`    : numpy shadow forward, calibrates bootstrap max_abs
  - `build_user_steps_mrpc`    : galois rotation step set + step_categories
  - `setup_engine`             : ckks_engine construction with per-step targets

---

## fixed-nt-causal-invariant

Fixed-nt causal-correctness invariant (rationale for `_real_nt`).

FHE attention has no causal mask: correctness comes purely from the
query only ever seeing real (non-padded) keys. In variable-nt mode the
prompt is exactly real_nt tokens with the query at the last position
(query_position == num_tokens - 1), so num_tokens IS the real token
count and there is nothing to clip. In --fixed-nt N mode the prompt is
padded with EOS keys AFTER the query (query_position = real_nt - 1,
num_tokens = N > real_nt); those pad keys must be excluded from every
score / softmax / calibration reduction or the query non-causally
attends future EOS tokens and the layer output diverges (L0 ~1e45).

`real_nt = query_position + 1` is the count of real tokens up to and
including the query and is the single quantity every num_tokens-direct
attention/softmax/calibration site must use. It is derivable at every
call site from the already-threaded query_position, so the fix needs
no new parameter or env flag and is the DEFAULT behavior. When the
prompt is not padded (variable-nt), real_nt == num_tokens and every
clip below is a no-op: the variable-nt path is byte-identical.

(The `_real_nt` docstring body: Real token count (incl. query). Pad slots
[real_nt, num_tokens) are excluded from all reductions. No-op
(== num_tokens) for variable-nt.)

---

## exp-cheb-coeffs-provenance

Degree-8 polynomial coefficients for exp on [-2, 2], extracted from
`softmax.cu:21-31` (`EXP_CHEB_COEFFS_DEG4_R2`). Used by
`_sim_pre_finsmx_mean`; stored at module level to avoid repeated
array construction.

---

## sim-pre-finsmx-mean-realnt

`_sim_pre_finsmx_mean` docstring body: Empirical mean of Stage-B softmax
output over ALL NUM_SLOTS slots.

Plaintext simulation (pure NumPy, no GPU). Mirrors
`/tmp/sim_pre_finsmx_mean.py`.

Args:

  - `scores_post_C_ht`: np.ndarray shape (N_HEADS, num_tokens) — real
    per-head post-sub_C scores from the current layer.
  - `num_tokens`: int — number of populated tokens.
  - `real_nt`: int or None — real (non-padded) token count. At fixed-nt
    the block_sizes clip masks pad-token score slots to 0 in the
    FHE pipeline, so columns [real_nt, num_tokens) behave like
    unpopulated (score=0) slots. None / == num_tokens → no-op
    (variable-nt: byte-identical to the original).

---

## ps-exp-init-token-count

ps_exp_init / damps use the SAME token count the FHE pipeline uses
(real_nt). The damping schedule's f_sq*d = target_mag cancellation
only holds when t_factor and damps share one token count, so these
MUST move together (see `softmax_damping_schedule`).

(Marks the `--- ps_exp_init on populated slots ---` section.)

---

## pad-columns-masked

Only the first real_nt score columns are populated; pad columns are
masked to 0 by the block_sizes clip and contribute v0 like every
other unpopulated slot.

---

## calib-realnt-clip

`compute_layer_calib_n` real_nt rationale.

Real (non-padded) token count. The FHE pipeline's block_sizes clip
masks pad-token K/V slots [real_nt, num_tokens) to 0 at stage-A, so
the calibration MUST compute scores / c_per_head / softmax_safety /
pre_finsmx_mean over the SAME first real_nt keys — otherwise the
padded-key scores (huge at L0: residual magnitude largest there)
pollute c_per_head, ps_exp saturates, and the layer blows up to
~1e45. Variable-nt: real_nt == num_tokens → identical slicing,
byte-identical result.

---

## exclude-eos-pad-kv

Exclude EOS-pad keys/values: the query (at P_q = real_nt - 1) only
ever attends real tokens [0, real_nt). No-op for variable-nt.

---

## softmax-safety-goldschmidt-noise

Softmax safety scale rationale (the `SOFTMAX_TARGET = 1.5` bound).

Softmax safety scale: post-damped per-head sum is approximately
TARGET_MAG (0.45) * sum_t exp(score_post_C[h, t]). Goldschmidt
`softmax_correct` converges for a∈(0, 2) but its CKKS noise floor
scales roughly as |a|^iters per iteration multiplication — at a≈1.5
the floor is ~570× larger than at a≈0.6. Block-0 (attention sink)
carries the largest weights → inherits this amplified Goldschmidt
residual, causing the observed 7.3× per-block noise asymmetry at L=10.
weights=e/a is scale-invariant, so we aggressively scale to land at
~0.6 instead of ~1.5.

---

## build-user-steps-overview

`build_user_steps_mrpc` docstring body: Galois rotation steps for the
dense token-major MRPC pipeline.

The IRP machinery is gone; the dense path is the SOLE compute path.
Every step below is a rotation a dense FHE kernel actually performs
(verified: the dense pipeline rotates exactly this set, a subset of
the rotations the deleted IRP step builders used to provision, so the
galois-key set is unchanged in coverage):

      rms_steps      rmsnorm_required_steps_stride_t  -> stride-t rmsnorm
                       {8,16,...,16384}
      bsgs_steps     phantom.bsgs_required_steps(64)  -> Wq/Wo/MLP BSGS
                       (bsgs_matmul_preencoded, d_pad in {D_TOTAL,D_PAD_MLP},
                        baby_steps in {64,128}; required steps {1,2,...,64})
      qkt_steps      qkt_required_steps(D_HEAD)       -> phantom.compute_qkt
                       inner-sum over d_head {1,2,...,64}
      smx_steps      softmax_required_steps(P,D_TOTAL)-> dense softmax
                       sum_reduce_stride(stride=D_TOTAL,count=P) {4096,8192,16384}
      score_v_steps  score_v_required_steps(D_HEAD,   -> phantom.score_times_v
                       D_TOTAL,P)  {-1,..,-64, 4096,8192,16384}
      bcast_steps    broadcast_required_steps(N_HEADS)-> optional per-head
                       softmax fold (env DENSE_SMX_BCAST) {-1,..,-16}

Returns (user_steps, step_categories); step_categories buckets steps
by the chain depth they fire at for setup_engine's per-step galois
target-chain assignment.

---

## p-frames-layout

Full positions per ciphertext frame (dense token-major layout):
NUM_SLOTS / D_TOTAL frames; the dense softmax sum_reduce_stride folds
these P frames at stride D_TOTAL.

---

## bcast-provision-unconditional

The DENSE_SMX_BCAST per-head fold is OFF by default; provision its
keys unconditionally so toggling the env var never needs an engine
rebuild (5 negative single-step keys, already a subset of score_v's
negative broadcast set — zero extra distinct galois elements).

---

## irp-attn-rotations

IRP §4.1 rotations (used by Wq + Wo) + §5.1 compute_qkt_irp rotations.
Wq and Wo share the same IRP rotation pattern at d=D_TOTAL=4096
baby_steps=16 → set-union collapses them. sdpa_irp_required_steps is a
safe superset covering QK^T + softmax + score_v rotations for future
downstream IRP extensions.

---

## irp-rect-mlp-rotations

IRP-rect MLP rotations (Cachemir §4.1 rect). MLP is now IRP-rect:
Wgate/Wup wide (d_in=D_MODEL=4096, d_out=D_PAD_OUT=16384) and Wdown
tall (d_in=D_PAD_OUT=16384, d_out=D_MODEL=4096). baby_steps=16 shared
with the attention IRP, so the square sub-IRP rotations set-union
collapse; only the α-stride rect rotations (q*t_prime / (N-q*t_prime))
are new. Provisioned unconditionally — IRP MLP is the sole compute path.

---

## irp-mlp-gateup-fold

MLP gate/up are now complex output-FOLDED at d_out_fold = D_PAD_OUT/2.
The folded matvec runs at the folded wide dims (d_in=D_MODEL,
d_out=8192), then extract_real_imag_pair (conj) + interleave_recombine
(right-rotate by t_fold/2). All of these collapse into the existing
wide/tall sets (verified empty diff), but provision them explicitly so
the keys survive a future dim change. NOTE: the conjugation step (0) is
auto-generated by phantom across all chains (merge_bootstrap + bootstrap
rely on it); it MUST NOT be added to user_rotation_steps or it gets a
single shallow target chain and breaks bootstrap's conjugation.

---

## irp-mlp-wdown-fold

MLP Wdown is now complex output-FOLDED too (d_out 4096 → d_out_fold 2048,
the biggest remaining tall fold). The folded TALL matvec runs the rect
machinery at d_out_fold = D_MODEL/2 (alpha doubles 4→8), needing finer
input-alignment / reduce rotations than the unfolded (D_PAD_OUT, D_MODEL)
tall path. The output is split by extract_real_imag_pair (conj step 0,
auto-generated — NOT added) and bridged out (decrypt + numpy recombine +
re-encrypt), so no interleave-recombine rotation here.

---

## irp-wq-wo-square-fold

Wq + Wo are now complex output-FOLDED SQUARE matvecs (d=D_TOTAL,
d_out_fold = D_TOTAL/2, K=512→256 SCPs each). The folded square matvec
runs the TALL-rect machinery at (d_in=D_TOTAL, d_out=D_TOTAL/2, alpha=2),
then extract_real_imag_pair (conj step 0, auto-generated — NOT added) and
an output SK bridge (decrypt + numpy recombine + re-encrypt). The rect
steps here are already a subset of the square IRP set (verified empty
diff), but provision them explicitly so the keys survive a dim change.

---

## chain-depth-buckets

Chain-depth buckets (dense pipeline trace, restarts at 16 each
bootstrap): BSGS Wq + compute_qkt inner-sum fire post-bootstrap at
chain 16 (qkt-class); the dense-softmax sum_reduce_stride fires
post-mask at chain 17 (finalize-class); score_times_v's broadcast
fires post-softmax at chain 23 (score_v-class). rms steps fire at 16.

score_v's positive accumulate steps coincide with smx sum_reduce
({4096,8192,16384}); the broadcast (negative) steps are score_v-only.

---

## setup-engine-target-chain

`setup_engine` docstring body: Build engine with per-step Galois target
chain assignment (Stage 3b-f-4).

Mirrors the optimization in `llama3.py` main(): each step's target_chain
is set to the SHALLOWEST chain at which it actually fires in the pipeline.
Smaller-target keys are larger; larger-target keys are smaller. Empirically
on this 5090 build the savings are storage-only (per-layer compute time
is unchanged from uniform target_chain=16) — phantom rotations cost scales
with the ciphertext's chain, not the key's coverage size. Kept for memory
correctness and parity with main()'s structure.

Pipeline chain trace (between bootstraps each stage restarts at 16):

      rms steps fire at chain 16 (sum_reduce inside rmsnorm)              -> 16
      qkt_q_preprocess {-1,-2,-4} fires at chain 16 (post-Wq bootstrap)   -> 16
      finalize_softmax sum_reduce {1,2,4} fires at chain 17 (post mask)   -> 17
      cross-block doubling {-1,-2,-4} fires at chain 17 (post mask)       -> 17
        (collides with qkt_q_preprocess on same galois elt; min wins -> 16)
      score_v broadcast {-T_MODEL*2^s} fires at chain 23 (post softmax)   -> 23
      IRP-only steps fire at chain 26 (USER_LEVEL_IRP_ATTN=10)            -> 26

Galois-element collisions are resolved with min-target-wins: the engine
generates one key per distinct galois element, so two steps mapping to
the same element must share the smaller (= shallowest-chain) target,
otherwise a key sized for a deep chain would be silently used at a
shallower chain and cause out-of-bounds reads.

---

## freshest-chain-invariant

FRESHEST_CHAIN=16 is invariant for both legacy (NSL=14) and use17
(NSL=16) under evalmod_r=3. Verified post-engine-construction
via `engine.freshest_chain_index()`.

---

## packed-softmax-chains

Packed-score softmax: same chain depths as the per-block
equivalents. sum_reduce + broadcast fire post-stage-A bootstrap
(chain 17 = TARGET_FINALIZE); unpack rotates the post-Goldschmidt
weights (chain 23 = TARGET_SCORE_V). Empty sets when packed
softmax is disabled.

---

## bootstrap-17-levels

Opt-in to the BootstrapTo17Levels chain layout (Lapis-shape, the_lib's
prime *counts* but Lapis prime *sizes*). Gives max_user_level = NSL-1
= 16 at NSL=17; useful for NUM_SQUARINGS=6 if the resulting bootstrap
working memory fits the GPU.
