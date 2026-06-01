"""End-to-end MRPC single-example FHE forward via multi-ct K/V cache.

# design: doc/design/llama3_mrpc.md#module-overview
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")

# ---- Phase 1/2/3 re-exports ------------------------------------------------
# design: doc/design/llama3_mrpc.md#reexport-diagnostics
from helpers.diagnostics import (
    _malloc_trim, _probe,
    _PROBE_DECRYPT_STAGES, _PROBE_DUMP_DIR, _PROBE_DUMP_LAYER,
)
# design: doc/design/llama3_mrpc.md#reexport-engine-setup
from fhe.engine_setup import (
    _make_rms_params_local, _real_nt, _sim_pre_finsmx_mean,
    compute_layer_calib_n, build_user_steps_mrpc, setup_engine,
)
# design: doc/design/llama3_mrpc.md#reexport-fhe-attention-dense
from fhe.fhe_attention_dense import (
    K_CACHE_SCALE, _DENSE_WQ_BABY_STEPS,
    encrypt_layer_inputs_multi, fhe_attention_dense_full,
    _LazyLayerWeights, _LAZY_FULL_WEIGHT_CACHE, _LAZY_FULL_WEIGHT_LOCK,
)
# design: doc/design/llama3_mrpc.md#reexport-pytorch-ref
from helpers.pytorch_ref import (
    capture_pytorch_ref_with_model, capture_pytorch_ref, _cached_pytorch_ref,
)
# design: doc/design/llama3_mrpc.md#reexport-decoder-layer
from fhe.decoder_layer import ClassifierCtx, run_decoder_layer
# ----------------------------------------------------------------------------

from blocks.bootstrap_placement import (
    build_layers_from_table, find_optimal_placement,
)
from blocks.lm_head import yes_no_logits_np
from helpers.llama3 import (
    NUM_SCALE_LEVELS,
    USER_LEVEL_IRP_ATTN,
    PROBE, PROBE_FULL,
    BOOT_CALIB_MARGIN,
    rope_matrix_np,
    load_layer_weights,
)


def _classifier_setup(num_tokens, query_position, pytorch_ref, pytorch_pre_norm,
                       cos_all_full, sin_all_full, label,
                       debug_layer, max_layer, min_layer,
                       engine, preloaded_weights, precomputed_calib):
    """Build the engine context + bootstrap plan + weight preload.
    Returns (ClassifierCtx, NUM_DECODERS).
    # design: doc/design/llama3_mrpc.md#classifier-setup-contract
    """
    print(f"=== run_classifier_fhe: {label}, NUM_TOKENS={num_tokens}, P={query_position} ===")
    P_local = query_position
    cos_all = cos_all_full[:num_tokens]
    sin_all = sin_all_full[:num_tokens]
    R_P = rope_matrix_np(cos_all[P_local], sin_all[P_local])

    final_norm_g = np.load(f"{PROBE_FULL}/final_norm_g.npy").astype(np.float64)
    lm_head_yesno = np.load(f"{PROBE_FULL}/lm_head_yesno.npy").astype(np.float64)
    meta = json.loads(open(f"{PROBE_FULL}/meta.json").read())

    # design: doc/design/llama3_mrpc.md#preloaded-weights-rationale
    def _get_layer_w(layer_idx):
        if preloaded_weights is not None:
            return _LazyLayerWeights(
                layer_idx, preloaded_weights[layer_idx],
                _LAZY_FULL_WEIGHT_CACHE, _LAZY_FULL_WEIGHT_LOCK)
        return load_layer_weights(layer_idx)

    # design: doc/design/llama3_mrpc.md#engine-reuse-rationale
    if engine is None:
        user_steps, step_categories = build_user_steps_mrpc()
        print(f"User steps ({len(user_steps)}): first 10 = {user_steps[:10]}")
        engine = setup_engine(user_steps, step_categories=step_categories)
    ctx = engine.context()
    encoder = engine.encoder()
    sk = engine.secret_key()
    relin_key = engine.relin_key()
    galois_key = engine.galois_key()
    fresh_ci = engine.freshest_chain_index()
    # design: doc/design/llama3_mrpc.md#freshest-chain-assert
    assert fresh_ci == 16, (
        f"engine.freshest_chain_index()={fresh_ci} != 16 — "
        "build_user_steps_mrpc targets need to be updated.")

    # design: doc/design/llama3_mrpc.md#irp-masks-removed

    # ---- Bootstrap placement
    # design: doc/design/llama3_mrpc.md#bootstrap-placement
    NSL_MAX = NUM_SCALE_LEVELS - 1
    T_BOOT_MS = 182.0
    OUTPUT_LEVEL_AFTER_IRP = USER_LEVEL_IRP_ATTN + 2
    placement_table = [
        ("rms1",      7,  29.4, True,  None,    True),
        ("attention", 0, 521.0, True,  NSL_MAX, False),
        ("residual1", 0,   1.0, True,  None,    False),
        ("rms2",      7,  27.4, True,  None,    True),
        ("mlp",       0, 624.1, True,  OUTPUT_LEVEL_AFTER_IRP, False),
        ("residual2", 0,   1.0, True,  None,    False),
    ]
    layers_for_dag = build_layers_from_table(placement_table)
    plan = find_optimal_placement(layers_for_dag, NSL_MAX, T_BOOT_MS)
    boot_before = {plan.layers[s.layer_idx].name: s.bootstrap_before for s in plan.steps}

    # ---- Per-layer FHE forward
    NUM_DECODERS = 32
    # design: doc/design/llama3_mrpc.md#autonomous-fhe-residual-stream
    _autonomous_fhe = os.environ.get("AUTONOMOUS_FHE") == "1"
    if _autonomous_fhe:
        print("  [AUTONOMOUS_FHE] carrying y_ct forward (K/V from ref)")
    # design: doc/design/llama3_mrpc.md#weight-subset-preload
    t_wq_encode0 = time.perf_counter()
    layer_weights = {}  # layer_idx -> {Wq,Wk,Wv,g1,g2} subset
    for _li in range(NUM_DECODERS):
        if min_layer is not None and _li < min_layer:
            continue
        if max_layer is not None and _li > max_layer:
            break
        _w = _get_layer_w(_li)
        layer_weights[_li] = {
            k: _w[k] for k in ("Wq", "Wk", "Wv", "g1", "g2") if k in _w
        }
        del _w
    t_wq_encode = time.perf_counter() - t_wq_encode0
    print(f"[weight-subset preload: {t_wq_encode:.1f}s]")

    cctx = ClassifierCtx(
        num_tokens=num_tokens, P_local=P_local,
        pytorch_ref=pytorch_ref, pytorch_pre_norm=pytorch_pre_norm,
        cos_all=cos_all, sin_all=sin_all, R_P=R_P,
        debug_layer=debug_layer, max_layer=max_layer, min_layer=min_layer,
        precomputed_calib=precomputed_calib,
        engine=engine, ctx=ctx, encoder=encoder, sk=sk,
        relin_key=relin_key, galois_key=galois_key, fresh_ci=fresh_ci,
        boot_before=boot_before, layer_weights=layer_weights,
        autonomous_fhe=_autonomous_fhe,
        final_norm_g=final_norm_g, lm_head_yesno=lm_head_yesno, meta=meta,
    )
    return cctx, NUM_DECODERS


def _run_lm_head(cctx, y_p_fhe, layer_times):
    """Final RMS norm + Yes/No logit projection + summary print.
    # design: doc/design/llama3_mrpc.md#run-lm-head-contract
    """
    # ---- LM head (host-side)
    yes_logit, no_logit = yes_no_logits_np(y_p_fhe, cctx.final_norm_g, cctx.lm_head_yesno,
                                              eps=cctx.meta["rms_norm_eps"])
    print(f"\n--- LM head: FHE yes_logit={yes_logit:.4f}  no_logit={no_logit:.4f} ---")
    print(f"--- Total layer time: {sum(layer_times)/1000:.1f}s "
          f"(avg {sum(layer_times)/len(layer_times):.0f}ms/layer) ---")
    return yes_logit, no_logit


def run_classifier_fhe(num_tokens, query_position, pytorch_ref, pytorch_pre_norm,
                         cos_all_full, sin_all_full, label="prompt",
                         debug_layer=None, max_layer=None, min_layer=None,
                         rp_indep_cache=None, engine=None,
                         shared_wq_cache=None, shared_wq_cache_events=None,
                         shared_wq_cache_lock=None,
                         preloaded_weights=None,
                         precomputed_calib=None,
                         rp_indep_disk_root=None):
    """End-to-end FHE classifier: 32 decoder layers + LM head -> Yes/No logits.

    # design: doc/design/llama3_mrpc.md#run-classifier-fhe-orchestrator
    """
    cctx, NUM_DECODERS = _classifier_setup(
        num_tokens, query_position, pytorch_ref, pytorch_pre_norm,
        cos_all_full, sin_all_full, label,
        debug_layer, max_layer, min_layer,
        engine, preloaded_weights, precomputed_calib)

    print(f"\nRunning {NUM_DECODERS} decoder layers...")
    layer_times = []
    y_p_fhe = None  # final hidden state at P (post-residual2 of last layer)
    _y_ct_carry = None  # bootstrapped y_ct from previous layer -> next x_ct

    for layer_idx in range(NUM_DECODERS):
        if min_layer is not None and layer_idx < min_layer:
            continue
        if max_layer is not None and layer_idx > max_layer:
            print(f"  early exit after layer {max_layer}")
            break
        y_p_fhe, _y_ct_carry = run_decoder_layer(
            layer_idx, cctx, _y_ct_carry, layer_times)

    return _run_lm_head(cctx, y_p_fhe, layer_times)


DEBUG_LAYER = None
MAX_LAYER = None
MIN_LAYER = None


def run_mrpc_example(idx, truncate_to=None):
    """Tokenize MRPC dev example #idx, run FHE pipeline, compare to PyTorch.
    # design: doc/design/llama3_mrpc.md#run-mrpc-example-contract
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    print(f"--- run_mrpc_example idx={idx} truncate_to={truncate_to} ---")
    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
    ds = load_dataset("nyu-mll/glue", "mrpc")["validation"]
    row = ds[idx]
    PROMPT_FMT = ("Are these two sentences paraphrases of each other?\n"
                  "Sentence 1: {s1}\nSentence 2: {s2}\n"
                  "Answer (Yes or No):")
    prompt = PROMPT_FMT.format(s1=row["sentence1"], s2=row["sentence2"])
    token_ids = tok(prompt).input_ids
    if truncate_to is not None and truncate_to < len(token_ids):
        token_ids = token_ids[:truncate_to]
    num_tokens = len(token_ids)
    P_local = num_tokens - 1
    print(f"  num_tokens={num_tokens}  label={row['label']}")
    print(f"  s1={row['sentence1']!r}")
    print(f"  s2={row['sentence2']!r}")

    # PyTorch reference (cached on disk)
    pytorch_ref, pytorch_pre_norm, yes_pt, no_pt = _cached_pytorch_ref(
        idx, truncate_to, token_ids)
    print(f"  PT  yes_logit={yes_pt:.4f}  no_logit={no_pt:.4f}")
    pt_pred = "Yes" if yes_pt > no_pt else "No"

    # RoPE tables for the prompt's positions
    cos_all_full = np.load(f"{PROBE_FULL}/rope_cos.npy").astype(np.float64)
    sin_all_full = np.load(f"{PROBE_FULL}/rope_sin.npy").astype(np.float64)

    yes_logit, no_logit = run_classifier_fhe(
        num_tokens, P_local, pytorch_ref, pytorch_pre_norm,
        cos_all_full, sin_all_full, label=f"mrpc_{idx}",
        debug_layer=DEBUG_LAYER, max_layer=MAX_LAYER, min_layer=MIN_LAYER)
    fhe_pred = "Yes" if yes_logit > no_logit else "No"
    print(f"\n=== Stage 3b-f-2 result ===")
    print(f"  FHE yes={yes_logit:.4f}  no={no_logit:.4f}  pred={fhe_pred}")
    print(f"  PT  yes={yes_pt:.4f}  no={no_pt:.4f}  pred={pt_pred}")
    print(f"  diff yes={abs(yes_logit-yes_pt):.3e}  no={abs(no_logit-no_pt):.3e}")
    print(f"  prediction agrees: {fhe_pred == pt_pred}")


def main_4tok():
    """Stage 3b-f-1 sanity: 4-token "[BOS] The quick brown" via the same pipeline.
    # design: doc/design/llama3_mrpc.md#main-4tok-contract
    """
    print("=== main_4tok: 4-token sanity ===")
    cos_all_full = np.load(f"{PROBE}/rope_cos.npy").astype(np.float64)
    sin_all_full = np.load(f"{PROBE}/rope_sin.npy").astype(np.float64)
    pytorch_ref = np.load(f"{PROBE_FULL}/ref_acts/qbrown4_bos.npy").astype(np.float64)
    pytorch_pre_norm = np.load(f"{PROBE_FULL}/ref_acts/qbrown4_bos_prenorm.npy").astype(np.float64)
    yes_logit, no_logit = run_classifier_fhe(
        num_tokens=4, query_position=3,
        pytorch_ref=pytorch_ref, pytorch_pre_norm=pytorch_pre_norm,
        cos_all_full=cos_all_full, sin_all_full=sin_all_full,
        label="qbrown4")
    print(f"\n=== main_4tok result ===")
    print(f"  FHE yes={yes_logit:.4f}  no={no_logit:.4f}  "
          f"pred={'Yes' if yes_logit > no_logit else 'No'}")
    yes_pt_ref, no_pt_ref = 0.9551, 3.2324
    print(f"  PT  yes={yes_pt_ref:.4f}  no={no_pt_ref:.4f}  "
          f"pred={'Yes' if yes_pt_ref > no_pt_ref else 'No'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mrpc", choices=["mrpc", "qbrown4"])
    ap.add_argument("--idx", type=int, default=359,
                    help="MRPC dev example index (default: 359 — 44-token shortest)")
    ap.add_argument("--debug-layer", type=int, default=None,
                    help="Print stage probes for this layer index")
    ap.add_argument("--max-layer", type=int, default=None,
                    help="Stop after this layer index (early exit)")
    ap.add_argument("--min-layer", type=int, default=None,
                    help="Skip layers before this index (uses PT ref as input)")
    ap.add_argument("--truncate-to", type=int, default=None,
                    help="Truncate the MRPC prompt to first N tokens (for num_tokens sweep)")
    args = ap.parse_args()
    DEBUG_LAYER = args.debug_layer
    MAX_LAYER = args.max_layer
    MIN_LAYER = args.min_layer
    if args.mode == "qbrown4":
        main_4tok()
    else:
        run_mrpc_example(args.idx, truncate_to=args.truncate_to)
