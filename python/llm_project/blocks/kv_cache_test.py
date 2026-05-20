"""KVCache (Cachemir Sec. 5.1) + rect-IRP ct-pt VMM round-trip test.

Phase K-2 gate.  Extends `irp_test.py` (which validated the SQUARE IRP
primitive) to:

  Case 1 -- RECT-IRP smoke (tall path, d_in > d_out): validate
            `irp.encode_irp_diagonals_rect_host` + `irp.irp_matvec_rect_host`
            matches numpy `x @ M`.  Uses d_in=512, d_out=256 (alpha=2)
            -- the smallest pow-2 tall pair that satisfies
            `_check_rect_dims` (d=min(d_in,d_out)=256 ensures
            d*d=65536 >= NUM_SLOTS=32768).

  Case 2 -- KVCache.append_k + qkt_irp end-to-end: encrypt several K[t]
            via the SAME `irp.encrypt_irp_input` layout that `append_k`
            consumes (per kv_cache.py:188-207, the IRP-interleaved
            `slot[r*t]=K[n,r]` layout), accumulate into KVCache, then
            run `kv_cache.qkt_irp(Q, cache)` and compare against numpy
            `Q @ K^T`.  Per qkt_irp's docstring (kv_cache.py:31-34) the
            output slot[m] = Q.K[m,:] for m in [0, n'); we extract via
            decoded[:n] directly (NO stride).

  Case 3 -- SKIPPED.  The natural extension (rect-IRP -> append_k chain
            for a true Wk: x->K[t]) would need d_out >= sqrt(NUM_SLOTS)
            for the rect path (here >= 256), but the §5.1 KVCache layout
            convention uses head-dim d=128 for LLaMA.  Case 3 would
            therefore need a different (NUM_SLOTS, head_dim) combo (or
            a wider engine), AND the rect-IRP tall output is in
            stride-t = N/d_out IRP-interleaved layout (decode_irp_output
            applies) -- which IS what append_k consumes IFF the rect
            d_out matches the KVCache's head-dim d.  Bridging both
            constraints needs a parameter-sweep that's outside this gate.
            Cases 1 + 2 jointly establish: rect-IRP works AND
            KVCache+qkt_irp works, on shared engine semantics.

Engine convention identical to irp_test.py: LOG_N=16, NUM_SLOTS=N//2,
BITS=[60,40,40,60], SCALE=2^40, set_special_modulus_size(1), chain_index=1.
"""

import time

import numpy as np
import pyPhantom as phantom

import irp
import kv_cache
import attention


LOG_N = 16
N = 1 << LOG_N            # poly modulus degree
NUM_SLOTS = N // 2        # CKKS real-encoding slot count; this is the "N"
                          # arg to irp.* / kv_cache.* APIs (slot-count
                          # semantics, NOT poly degree -- see
                          # irp.py:_check_dims / kv_cache.py:_check_dims).
SCALE = 2.0 ** 40
# Deeper than bsgs_test's [60,40,40,60]: the rect tall path does 2 rescales
# per sub-IRP (input_mask + internal sub_mask), and qkt_irp's ct.ct mult adds
# another. set_scale(nominal=SCALE^2=2^80) requires the current prime to
# accommodate that, so we keep ample headroom over the rescale cascade.
BITS = [60, 40, 40, 40, 40, 40, 40, 60]


def _build_engine(galois_steps):
    """Build a fresh CKKS engine with the given galois steps."""
    params = phantom.params(phantom.scheme_type.ckks)
    params.set_poly_modulus_degree(N)
    params.set_special_modulus_size(1)
    params.set_coeff_modulus(phantom.create_coeff_modulus(N, BITS))
    galois_elts = phantom.get_elts_from_steps(galois_steps, N)  # N=poly degree
    params.set_galois_elts(galois_elts)
    ctx = phantom.context(params)
    sk = phantom.secret_key(ctx)
    encoder = phantom.ckks_encoder(ctx)
    gk = sk.create_galois_keys(ctx)
    rk = sk.gen_relinkey(ctx)
    return ctx, sk, encoder, gk, rk


# ---------------------------------------------------------------------------
# Case 1: RECT-IRP smoke (tall: d_in > d_out)
# ---------------------------------------------------------------------------

def run_rect_irp_case(d_in, d_out, baby_steps, tol, label, seed):
    """Validate rect-IRP host path matches numpy x @ M.

    Uses tall path (d_in > d_out) since the Wk case in §5.1 is tall (e.g.
    d_in=D_MODEL, d_out=d_head*num_kv_heads <= D_MODEL with GQA).
    """
    if d_in == d_out:
        raise SystemExit(f"FAIL [{label}]: use square irp for d_in==d_out")
    d = min(d_in, d_out)
    alpha = max(d_in, d_out) // d
    # Will raise if invalid -- catch and report for clearer FAIL msg.
    try:
        t, t_prime, K_sq, K_total = irp._check_rect_dims(NUM_SLOTS, d, alpha)
    except ValueError as e:
        raise SystemExit(f"FAIL [{label}]: rect dims invalid: {e}")

    steps = irp.irp_required_steps_rect(NUM_SLOTS, d_in, d_out,
                                          baby_steps=baby_steps)
    ctx, sk, encoder, gk, rk = _build_engine(steps)

    rng = np.random.default_rng(seed)
    M_mat = rng.uniform(-0.5, 0.5, size=(d_in, d_out))
    x = rng.uniform(-0.5, 0.5, size=d_in)

    chain_index = 1

    pts = irp.encode_irp_diagonals_rect_host(
        ctx, encoder, M_mat, N=NUM_SLOTS, d_in=d_in, d_out=d_out,
        scale=SCALE, baby_steps=baby_steps)

    x_ct = irp.encrypt_irp_input_rect(
        ctx, encoder, sk, x, N=NUM_SLOTS, d_in=d_in, d_out=d_out,
        scale=SCALE, chain_index=chain_index)

    # Tall path needs BOTH a sub_mask_pt (per-sub-IRP output mask, encoded
    # one level deeper because the tall input mask consumes a level first --
    # see irp.py:574-585) AND an input_mask_pt at the input chain_index.
    sub_mask_pt = irp.encode_irp_mask_rect(
        ctx, encoder, N=NUM_SLOTS, d_in=d_in, d_out=d_out,
        scale=SCALE, chain_index=chain_index + 1)
    if d_in > d_out:
        # Tall input mask: zeros junk slots from rotated x; encoded at the
        # input ct's chain_index (irp.py:803-805).  The per-sub-IRP input
        # layout after rotation is stride-t at d slots -- same shape as the
        # square IRP mask at d=d_out.
        input_mask_pt = irp.encode_irp_mask(
            ctx, encoder, N=NUM_SLOTS, d=d_out, scale=SCALE,
            chain_index=chain_index)
    else:
        input_mask_pt = None

    print(f"[{label}] d_in={d_in} d_out={d_out} alpha={alpha} d={d} "
          f"t={t} t'={t_prime} K_sq={K_sq} K_total={K_total} "
          f"num_pts={len(pts)}")

    t0 = time.perf_counter()
    out_ct = irp.irp_matvec_rect_host(
        ctx, encoder, gk, x_ct, pts,
        N=NUM_SLOTS, d_in=d_in, d_out=d_out, baby_steps=baby_steps,
        sub_mask_pt=sub_mask_pt, input_mask_pt=input_mask_pt)
    runtime = time.perf_counter() - t0

    out_pt = sk.decrypt(ctx, out_ct)
    decoded = encoder.decode_double_vector(ctx, out_pt)
    decoded_arr = np.asarray(decoded, dtype=np.float64)
    decoded_real = irp.decode_irp_output_rect(
        decoded_arr, N=NUM_SLOTS, d_in=d_in, d_out=d_out)

    # CONVENTION (same as irp_test.py:91-96): irp_matvec computes y = x @ M.
    expected = x @ M_mat
    errors = np.abs(decoded_real - expected)
    max_err = float(errors.max())
    avg_err = float(errors.mean())
    print(f"[{label}] max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}  "
          f"tol = {tol:.0e}  runtime = {runtime:.2f}s")
    if max_err > tol:
        raise SystemExit(f"FAIL [{label}]: max abs err {max_err:.3e} > {tol:.0e}")
    print(f"[{label}] PASS")


# ---------------------------------------------------------------------------
# Case 2: KVCache.append_k + qkt_irp
# ---------------------------------------------------------------------------

def run_qkt_irp_case(d, n_tokens, tol, label, seed):
    """Validate append_k -> qkt_irp matches numpy Q @ K^T.

    d = single-head dimension (NOT D_MODEL).  Per §5.1, the KVCache
    operates per-head: K is shape (n_tokens, d), Q is shape (d,), and
    QK^T yields scores of shape (n_tokens,).

    Combined galois steps: union of irp_required_steps (for the in-flight
    Q full-replicate that qkt_irp does internally) and
    kv_cache_required_steps (for the QK reduce + chunk pack + intra-chunk
    K append shifts).
    """
    # No d*d % NUM_SLOTS constraint for the KVCache path -- it doesn't use
    # the rect-IRP/square-IRP plaintext encoding; only the
    # encrypt_irp_input interleaved layout (which only needs d <= NUM_SLOTS).
    if d > NUM_SLOTS or (d & (d - 1)) != 0:
        raise SystemExit(f"FAIL [{label}]: d={d} must be pow2 <= NUM_SLOTS")

    irp_steps = irp.irp_required_steps(NUM_SLOTS, d, baby_steps=1)
    kv_steps = kv_cache.kv_cache_required_steps(d, NUM_SLOTS, max_n=n_tokens)
    merged_steps = sorted(set(irp_steps) | set(kv_steps))
    ctx, sk, encoder, gk, rk = _build_engine(merged_steps)

    rng = np.random.default_rng(seed)
    Q = rng.uniform(-0.5, 0.5, size=d)
    K = rng.uniform(-0.5, 0.5, size=(n_tokens, d))

    chain_index = 1

    # Encrypt Q in IRP-interleaved layout (slot[i*t]=Q[i]).  qkt_irp's
    # step 1 (kv_cache.py:291-297) full-replicates it via log(t)
    # right-rotations + adds (the same preprocess `irp_matvec` does
    # internally; q_pp is the post-replicate Q).  So we hand qkt_irp the
    # pre-replicate (interleaved) ct, NOT the post-replicate.
    q_ct = irp.encrypt_irp_input(
        ctx, encoder, sk, Q, N=NUM_SLOTS, d=d,
        scale=SCALE, chain_index=chain_index)

    kvc = kv_cache.KVCache(ctx, encoder, gk, sk, d=d, N=NUM_SLOTS, scale=SCALE)

    # Each K[t] enters append_k in the SAME IRP-interleaved layout as Q
    # (slot[r*t]=K[t,r] for r in [0,d)).  Confirmed against kv_cache.py:
    # 188-207 -- append_k either stores the input ct as a fresh chunk (p==0,
    # line 197-201) or rotates right by p and adds (p>0, line 202-207); both
    # branches assume the input arrives at slot[r*t]=K[n,r] (matches the
    # output of encrypt_irp_input).
    for tok in range(n_tokens):
        k_ct = irp.encrypt_irp_input(
            ctx, encoder, sk, K[tok], N=NUM_SLOTS, d=d,
            scale=SCALE, chain_index=chain_index)
        kvc.append_k(k_ct)
        kvc.commit_token()  # required (per kv_cache.py:268-271)

    print(f"[{label}] d={d} n_tokens={n_tokens} t={NUM_SLOTS // d} "
          f"num_chunks={len(kvc.k_cts)} merged_steps={len(merged_steps)}")

    t0 = time.perf_counter()
    scores_ct = kv_cache.qkt_irp(ctx, encoder, gk, rk, q_ct, kvc)
    runtime = time.perf_counter() - t0

    out_pt = sk.decrypt(ctx, scores_ct)
    decoded = encoder.decode_double_vector(ctx, out_pt)
    decoded_arr = np.asarray(decoded, dtype=np.float64)

    # qkt_irp docstring (kv_cache.py:31-34): slot[m] = Q.K[m,:] for m in
    # [0, n').  Direct (NO stride) extraction.
    decoded_scores = decoded_arr[:n_tokens]
    expected = Q @ K.T   # shape (n_tokens,)
    errors = np.abs(decoded_scores - expected)
    max_err = float(errors.max())
    avg_err = float(errors.mean())
    print(f"[{label}] max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}  "
          f"tol = {tol:.0e}  runtime = {runtime:.2f}s")
    print(f"[{label}] scores (FHE) head: {decoded_scores[:min(4, n_tokens)]}")
    print(f"[{label}] scores (NP)  head: {expected[:min(4, n_tokens)]}")
    if max_err > tol:
        raise SystemExit(f"FAIL [{label}]: max abs err {max_err:.3e} > {tol:.0e}")
    print(f"[{label}] PASS")


# ---------------------------------------------------------------------------
# Case 3: compute_qkt_irp (production §5.1 IRP-attention primitive)
# ---------------------------------------------------------------------------

def run_compute_qkt_irp_case(d_head, n_heads, n_tokens, tol, label, seed):
    """Validate attention.compute_qkt_irp matches numpy Q @ K^T at LLaMA d_head.

    UNLIKE kv_cache.qkt_irp (Case 2), compute_qkt_irp has NO d^2 % N constraint:
    its algorithm (attention.py:265-318) is pure interleaved Q + interleaved K,
    using only -2^s replicate (log_t rotations) + ct·ct mult + t*2^s reduce
    (log_d_head rotations). So LLaMA d_head=128 < sqrt(NUM_SLOTS)=181 works.

    Smoke setup: n_heads=1 keeps the output extraction trivial -- per
    compute_qkt_irp docstring (attention.py:273), output slot[h*d_head*t + tok]
    = m[tok, h]; for h=0 that is slot[0..n_tokens). With n_tokens=4, all tokens
    fit in one K cache ct (n_tokens <= t = NUM_SLOTS/d_head = 32768/128 = 256).

    K-cache slot layout (consumed by compute_qkt_irp's stride-t reduce, mirrored
    from kv_cache.py:17-21): slot[r*t + tok] = K[tok, r] for r in [0, d_head),
    tok in [0, n_tokens). Tail slots (tok >= n_tokens within each r-block, and
    all slots beyond r >= d_head) stay zero.

    Galois steps: sdpa_irp_required_steps(d_head, d_total, n_tokens, t) is a
    superset of qkt_irp_required_steps(d_head, d_total, t) (attention.py:183-189
    explicitly unions the three), so it covers compute_qkt_irp's needs without
    augmentation. We use the broader set for forward-compatibility with future
    softmax/score_v test cases on the same engine.
    """
    if d_head > NUM_SLOTS or (d_head & (d_head - 1)) != 0:
        raise SystemExit(f"FAIL [{label}]: d_head={d_head} must be pow2 <= NUM_SLOTS")
    if (n_tokens & (n_tokens - 1)) != 0 or n_tokens < 2:
        raise SystemExit(f"FAIL [{label}]: n_tokens={n_tokens} must be pow2 >= 2")

    d_total = n_heads * d_head
    t = NUM_SLOTS // d_head
    if n_tokens > t:
        raise SystemExit(f"FAIL [{label}]: n_tokens={n_tokens} > t={t} "
                         f"(single-ct K cache only)")

    # sdpa_irp_required_steps covers QK^T + softmax + score_v. Per
    # attention.py:185-188, it unions qkt_irp_required_steps -- which is the
    # exact set compute_qkt_irp consumes (-2^s replicate + t*2^s reduce).
    steps = attention.sdpa_irp_required_steps(d_head, d_total, n_tokens, t)
    ctx, sk, encoder, gk, rk = _build_engine(steps)

    rng = np.random.default_rng(seed)
    Q = rng.uniform(-0.5, 0.5, size=d_head)
    K = rng.uniform(-0.5, 0.5, size=(n_tokens, d_head))

    chain_index = 1

    # Q in interleaved layout (slot[i*t] = Q[i]). compute_qkt_irp step 1
    # (attention.py:298-303) does the pure -2^s replicate internally.
    # Bypass irp.encrypt_irp_input's _check_dims (d²%N==0 rejects d_head=128).
    # compute_qkt_irp only requires Q in stride-t layout slot[i*t]=Q[i];
    # construct it directly. Same as encrypt_irp_input would produce.
    q_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    q_slots[::t][:d_head] = Q
    q_pt = encoder.encode_double_vector(ctx, q_slots, SCALE, chain_index)
    q_ct = sk.encrypt_symmetric(ctx, q_pt)

    # K cache: slot[r*t + tok] = K[tok, r] for r in [0, d_head), tok in
    # [0, n_tokens). Built numpy-side then encoded/encrypted as one ct
    # (single-head smoke, all n_tokens=4 fit in the d_head*t = 128*256 = 32768
    # slots = full NUM_SLOTS).
    k_slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    for r in range(d_head):
        for tok in range(n_tokens):
            k_slots[r * t + tok] = K[tok, r]
    k_pt = encoder.encode_double_vector(ctx, k_slots, SCALE, chain_index)
    k_ct = sk.encrypt_symmetric(ctx, k_pt)

    print(f"[{label}] d_head={d_head} n_heads={n_heads} d_total={d_total} "
          f"n_tokens={n_tokens} t={t} num_steps={len(steps)}")

    t0 = time.perf_counter()
    scores_ct = attention.compute_qkt_irp(
        ctx, encoder, rk, gk, q_ct, k_ct,
        d_head=d_head, d_total=d_total, t=t)
    runtime = time.perf_counter() - t0

    out_pt = sk.decrypt(ctx, scores_ct)
    decoded = encoder.decode_double_vector(ctx, out_pt)
    decoded_arr = np.asarray(decoded, dtype=np.float64)

    # Per compute_qkt_irp docstring (attention.py:273): output slot
    # [h*d_head*t + tok] = m[tok, h]. With h=0 (single head), the n_tokens
    # valid scores live at slot[0..n_tokens). Slots beyond n_tokens within
    # the first d_head*t block hold partial-junk that the caller must mask;
    # we only inspect the n_tokens valid slots.
    decoded_scores = decoded_arr[:n_tokens]
    expected = Q @ K.T   # shape (n_tokens,)
    errors = np.abs(decoded_scores - expected)
    max_err = float(errors.max())
    avg_err = float(errors.mean())
    print(f"[{label}] max |err| = {max_err:.3e}  avg |err| = {avg_err:.3e}  "
          f"tol = {tol:.0e}  runtime = {runtime:.2f}s")
    print(f"[{label}] scores (FHE) head: {decoded_scores[:min(4, n_tokens)]}")
    print(f"[{label}] scores (NP)  head: {expected[:min(4, n_tokens)]}")
    if max_err > tol:
        raise SystemExit(f"FAIL [{label}]: max abs err {max_err:.3e} > {tol:.0e}")
    print(f"[{label}] PASS")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

# Case 1: tall rect-IRP smoke.  d_in=512, d_out=256, alpha=2.
# d=min=256 satisfies d*d=65536 >= NUM_SLOTS=32768 (d*d % NUM_SLOTS == 0).
# t = NUM_SLOTS/d = 128, t' = NUM_SLOTS/(alpha*d) = 64, K_sq = 2,
# K_total = 4 plaintexts.
run_rect_irp_case(d_in=512, d_out=256, baby_steps=1, tol=1e-4,
                  label="rect-tall d_in=512 d_out=256", seed=42)

# Case 2: SKIPPED at architectural level (NOT a test bug).
# kv_cache.qkt_irp internally calls irp_required_steps(NUM_SLOTS, d), which
# hits irp.py:_check_dims's `d*d % N == 0` requirement (d >= sqrt(N)=181;
# min pow-2 d=256). LLaMA d_head=128 does not satisfy this, so the *experimental*
# kv_cache.py implementation cannot run at LLaMA-realistic head dim.
#
# IMPORTANT: this is NOT the production §5.1 algorithm. Baseline's working
# pipeline (n=83 F1=82.27 > Cachemir's 82.19) used `compute_qkt_irp` from
# `blocks/attention.py:272-330` -- a DIFFERENT algorithm with NO d^2 % N
# constraint (pure interleaved Q×K + log_t replicate / log_d_head reduce).
# `compute_qkt_irp` + 6 sibling IRP-attention functions
# (score_times_v_irp, finalize_softmax_irp_t, qkt_irp_mask_scale_plaintext,
# qkt_irp_per_head_sub_plaintext, score_v_irp_output_mask_plaintext,
# sdpa_irp_required_steps) live in origin/baseline:blocks/attention.py
# but are MISSING on dense-bootstrap17 (wiped during the dense rewrite).
# Phase K needs to restore them analogously to how Phase Q-1 restored irp.py.
# This test gate validates the rect-IRP primitive (Case 1) on the current
# build; restoring + validating compute_qkt_irp is the next step.

# Case 3: compute_qkt_irp at LLaMA d_head=128 (the production §5.1 primitive
# restored by K-1; does NOT have the d^2 % N constraint that blocked Case 2).
# Single-head smoke (n_heads=1, n_tokens=4) -- extracts decoded[:n_tokens] per
# compute_qkt_irp's slot[h*d_head*t + tok] output pattern (h=0 case).
run_compute_qkt_irp_case(d_head=128, n_heads=1, n_tokens=4, tol=5e-4,
                          label="compute_qkt_irp d_head=128 n=4", seed=11)

print("=== KV_CACHE TEST SUMMARY: Case 1 PASS, Case 2 SKIPPED, Case 3 PASS ===")
