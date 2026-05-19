"""Correctness gate for the dense token-major K/V packer (Stage 1, pure numpy).

Source contract: src/attention.cu:21-25,59-95 + src/linear.cu:17-24.

Tests:
1. next_pow2
2. Q pack/unpack bit-exact round-trip (broadcast across P token frames)
3. K/V pack/unpack bit-exact round-trip + exact-0 pad region
4. dense_qkt slot-level simulation matches einsum within 1e-12
5. dense_score_v slot-level simulation matches einsum within 1e-12
6. GQA head mapping: query head h -> kv head h // n_kv_groups
7. Multi-shard: sharding over P=8 with real_nt > P (shard-split and cross-shard add)
8. Per-head softmax denominator cross-shard sum (Stage 3 oracle)

Runs standalone (python3 test_kv_layout_dense.py) — no FHE, no GPU.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks.kv_layout_dense import (
    next_pow2,
    positions_per_ct,
    pack_q_dense,
    unpack_q_dense,
    pack_kv_dense_shards,
    unpack_kv_dense_shards,
    pack_scores_shard,
    unpack_scores_shard,
    dense_qkt_shard,
    dense_qkt,
    dense_score_v_shard,
    dense_score_v,
    _inner_sum_slots,
    _broadcast_within_heads,
    _accumulate_over_positions,
)

# LLaMA-3.1-8B dims (llama3.py:70-75)
N_HEADS    = 32
N_KV_HEADS = 8
N_KV_GROUPS = N_HEADS // N_KV_HEADS   # 4
D_HEAD     = 128
D_TOTAL    = N_HEADS * D_HEAD          # 4096
NUM_SLOTS  = 32768
# positions_per_ct at real scale: min(nt_pad, 32768//4096) = min(nt_pad, 8)
P_FULL     = NUM_SLOTS // D_TOTAL      # 8

REAL_NT_CASES = [1, 8, 44, 60, 64, 128, 512]


def _gqa_expand(X_t_kvh_d):
    """Reference GQA expansion = np.repeat(..., N_KV_GROUPS, axis=1)."""
    return np.repeat(X_t_kvh_d, N_KV_GROUPS, axis=1)


def _softmax_ref(scores_t_h):
    """Numerically stable softmax over tokens for each head."""
    s = scores_t_h - scores_t_h.max(axis=0, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# 1. next_pow2
# ---------------------------------------------------------------------------
def test_next_pow2():
    cases = {0: 1, 1: 1, 2: 2, 3: 4, 8: 8, 44: 64, 60: 64, 64: 64,
             128: 128, 129: 256, 512: 512, 513: 1024}
    for n, want in cases.items():
        got = next_pow2(n)
        assert got == want, f"next_pow2({n})={got}, want {want}"
    print("[ok] test_next_pow2")


# ---------------------------------------------------------------------------
# 2. Q round-trip: pack broadcasts Q across all P frames; unpack reads frame 0
# ---------------------------------------------------------------------------
def test_q_roundtrip_bitexact():
    rng = np.random.default_rng(11)
    Q = rng.standard_normal((N_HEADS, D_HEAD))
    for P in [1, 4, 8]:
        slots = pack_q_dense(Q, P)
        assert slots.shape == (P * D_TOTAL,), \
            f"q_slots shape {slots.shape}, expected ({P * D_TOTAL},)"
        # every token frame must carry the same Q
        for tok in range(P):
            frame = slots[tok * D_TOTAL:(tok + 1) * D_TOTAL].reshape(N_HEADS, D_HEAD)
            assert np.array_equal(frame, Q), f"Q frame {tok} not bit-exact copy"
        Q_back = unpack_q_dense(slots, N_HEADS, D_HEAD)
        assert np.array_equal(Q, Q_back), "unpack_q_dense not bit-exact"
    print("[ok] test_q_roundtrip_bitexact")


# ---------------------------------------------------------------------------
# 3. K/V shard round-trip: bit-exact + pad = exact 0.0
# ---------------------------------------------------------------------------
def test_kv_roundtrip_bitexact_and_pad_zero():
    rng = np.random.default_rng(22)
    for real_nt in REAL_NT_CASES:
        P = min(next_pow2(real_nt), P_FULL)
        K = rng.standard_normal((real_nt, N_KV_HEADS, D_HEAD))
        V = rng.standard_normal((real_nt, N_KV_HEADS, D_HEAD))
        k_shards, v_shards = pack_kv_dense_shards(K, V, real_nt, P, N_HEADS)

        n_shards = math.ceil(real_nt / P)
        assert len(k_shards) == n_shards, f"n_shards mismatch real_nt={real_nt}"
        for b, (ks, vs) in enumerate(zip(k_shards, v_shards)):
            assert ks.shape == (P * D_TOTAL,), f"shard {b} shape {ks.shape}"
            assert vs.shape == (P * D_TOTAL,)
            # check pad slots inside this shard are exactly 0
            for tok_local in range(P):
                tok_abs = b * P + tok_local
                if tok_abs >= real_nt:
                    frame_k = ks[tok_local * D_TOTAL:(tok_local + 1) * D_TOTAL]
                    frame_v = vs[tok_local * D_TOTAL:(tok_local + 1) * D_TOTAL]
                    assert np.array_equal(frame_k, np.zeros_like(frame_k)), \
                        f"K pad nonzero shard={b} tok_local={tok_local}"
                    assert np.array_equal(frame_v, np.zeros_like(frame_v)), \
                        f"V pad nonzero shard={b} tok_local={tok_local}"

        # round-trip == GQA-expanded inputs, bit-exact
        K_back, V_back = unpack_kv_dense_shards(k_shards, v_shards,
                                                  real_nt, P, N_HEADS, D_HEAD)
        K_exp = _gqa_expand(K)
        V_exp = _gqa_expand(V)
        assert np.array_equal(K_back, K_exp), f"K round-trip real_nt={real_nt}"
        assert np.array_equal(V_back, V_exp), f"V round-trip real_nt={real_nt}"
        print(f"[ok] kv roundtrip+pad real_nt={real_nt} P={P} n_shards={n_shards}")
    print("[ok] test_kv_roundtrip_bitexact_and_pad_zero")


# ---------------------------------------------------------------------------
# 4. dense_qkt: slot-level inner_sum simulation matches einsum
#    PROVES GEOMETRY: _inner_sum_slots operates on contiguous d_head slots
# ---------------------------------------------------------------------------
def test_dense_qkt_matches_einsum():
    """dense_qkt (slot-level inner_sum) == einsum('hd,thd->th',Q,K_exp)/sqrt(H)."""
    rng = np.random.default_rng(44)
    inv_sqrt_d = 1.0 / np.sqrt(D_HEAD)
    for real_nt in REAL_NT_CASES:
        P = min(next_pow2(real_nt), P_FULL)
        Q = rng.standard_normal((N_HEADS, D_HEAD))
        K = rng.standard_normal((real_nt, N_KV_HEADS, D_HEAD))
        V = np.zeros_like(K)
        k_shards, _ = pack_kv_dense_shards(K, V, real_nt, P, N_HEADS)
        q_shards = [pack_q_dense(Q, P) for _ in k_shards]

        got = dense_qkt(q_shards, k_shards, N_HEADS, D_HEAD, real_nt, P, inv_sqrt_d)

        K_exp = _gqa_expand(K)  # [real_nt, N_HEADS, D_HEAD]
        ref = np.einsum('hd,thd->th', Q, K_exp) * inv_sqrt_d  # [real_nt, N_HEADS]

        assert got.shape == (real_nt, N_HEADS)
        err = np.max(np.abs(got - ref))
        assert err < 1e-12, f"dense_qkt err={err:.3e} real_nt={real_nt}"
        print(f"[ok] dense_qkt real_nt={real_nt} P={P} maxerr={err:.2e}")
    print("[ok] test_dense_qkt_matches_einsum")


# ---------------------------------------------------------------------------
# 5. dense_score_v: slot-level broadcast+accumulate simulation matches einsum
#    PROVES GEOMETRY: _broadcast_within_heads and _accumulate_over_positions
#    operate on correct contiguous d_head and d_total-stride patterns
# ---------------------------------------------------------------------------
def test_dense_score_v_matches_einsum():
    """dense_score_v (slot-level simulation) == einsum('th,thd->hd', W, V_exp)."""
    rng = np.random.default_rng(55)
    for real_nt in REAL_NT_CASES:
        P = min(next_pow2(real_nt), P_FULL)
        n_shards = math.ceil(real_nt / P)
        W = rng.standard_normal((real_nt, N_HEADS))
        K = np.zeros((real_nt, N_KV_HEADS, D_HEAD))
        V = rng.standard_normal((real_nt, N_KV_HEADS, D_HEAD))
        _, v_shards = pack_kv_dense_shards(K, V, real_nt, P, N_HEADS)
        # build score_shards from W
        score_shards = [
            pack_scores_shard(W, b * P, P, N_HEADS, D_HEAD)
            for b in range(n_shards)
        ]

        got = dense_score_v(score_shards, v_shards, N_HEADS, D_HEAD, P)

        V_exp = _gqa_expand(V)  # [real_nt, N_HEADS, D_HEAD]
        ref = np.einsum('th,thd->hd', W, V_exp)  # [N_HEADS, D_HEAD]

        assert got.shape == (N_HEADS, D_HEAD)
        err = np.max(np.abs(got - ref))
        assert err < 1e-12, f"dense_score_v err={err:.3e} real_nt={real_nt}"
        print(f"[ok] dense_score_v real_nt={real_nt} P={P} maxerr={err:.2e}")
    print("[ok] test_dense_score_v_matches_einsum")


# ---------------------------------------------------------------------------
# 6. GQA head mapping: query head h -> kv head h // n_kv_groups
#    Uses fingerprinted K so recovered score reveals exactly which kv head
# ---------------------------------------------------------------------------
def test_gqa_head_mapping():
    real_nt = 8
    P = min(next_pow2(real_nt), P_FULL)
    # K[tok, kvh, j] = (kvh+1)*1000 (constant per kv head, all j and tok)
    K = np.zeros((real_nt, N_KV_HEADS, D_HEAD))
    for kvh in range(N_KV_HEADS):
        K[:, kvh, :] = (kvh + 1) * 1000.0
    V = np.zeros_like(K)
    Q = np.ones((N_HEADS, D_HEAD))  # all-ones -> dot = D_HEAD * kv_head_value
    k_shards, _ = pack_kv_dense_shards(K, V, real_nt, P, N_HEADS)
    q_shards = [pack_q_dense(Q, P) for _ in k_shards]

    # inv_sqrt_d=1.0 to disable scaling for exact identification
    scores = dense_qkt(q_shards, k_shards, N_HEADS, D_HEAD, real_nt, P, inv_sqrt_d=1.0)

    for h in range(N_HEADS):
        # scores[tok, h] = D_HEAD * (expected_kvh + 1) * 1000
        recovered_kvh = int(round(scores[0, h] / D_HEAD / 1000.0)) - 1
        expected_kvh = h // N_KV_GROUPS
        assert recovered_kvh == expected_kvh, (
            f"query head {h}: read kv head {recovered_kvh}, expected {expected_kvh}"
        )
    # cross-check vs canonical np.repeat
    K_exp = _gqa_expand(K)
    for h in range(N_HEADS):
        assert np.array_equal(K_exp[:, h, :], K[:, h // N_KV_GROUPS, :])
    print(f"[ok] test_gqa_head_mapping (h -> h//{N_KV_GROUPS})")


# ---------------------------------------------------------------------------
# 7. Multi-shard verification: real_nt > P forces multiple shards
#    Checks shard-split, per-shard reduction, and cross-shard accumulation
# ---------------------------------------------------------------------------
def test_multi_shard_consistency():
    """Verify sharding with real_nt=44 (P=8 -> 6 shards) and real_nt=128 (16 shards)."""
    rng = np.random.default_rng(77)
    for real_nt in [44, 128, 512]:
        P = P_FULL  # 8 (the actual FHE positions_per_ct at D=4096, N=32768)
        n_shards = math.ceil(real_nt / P)
        Q = rng.standard_normal((N_HEADS, D_HEAD))
        K = rng.standard_normal((real_nt, N_KV_HEADS, D_HEAD))
        V = rng.standard_normal((real_nt, N_KV_HEADS, D_HEAD))
        k_shards, v_shards = pack_kv_dense_shards(K, V, real_nt, P, N_HEADS)
        assert len(k_shards) == n_shards

        # QKT: scores from all shards assembled
        q_shards = [pack_q_dense(Q, P) for _ in k_shards]
        scores = dense_qkt(q_shards, k_shards, N_HEADS, D_HEAD, real_nt, P)
        K_exp = _gqa_expand(K)
        ref_scores = np.einsum('hd,thd->th', Q, K_exp) / np.sqrt(D_HEAD)
        err_q = np.max(np.abs(scores - ref_scores))
        assert err_q < 1e-12, f"multi-shard qkt err={err_q:.3e} real_nt={real_nt}"

        # softmax weights
        W = _softmax_ref(scores)  # [real_nt, N_HEADS]

        # score_times_v: per-shard accumulate + cross-shard add
        score_shards = [
            pack_scores_shard(W, b * P, P, N_HEADS, D_HEAD)
            for b in range(n_shards)
        ]
        got_out = dense_score_v(score_shards, v_shards, N_HEADS, D_HEAD, P)
        V_exp = _gqa_expand(V)
        ref_out = np.einsum('th,thd->hd', W, V_exp)
        err_v = np.max(np.abs(got_out - ref_out))
        assert err_v < 1e-12, f"multi-shard score_v err={err_v:.3e} real_nt={real_nt}"
        print(f"[ok] multi-shard real_nt={real_nt} n_shards={n_shards} "
              f"qkt_err={err_q:.2e} sv_err={err_v:.2e}")
    print("[ok] test_multi_shard_consistency")


# ---------------------------------------------------------------------------
# 8. Per-head softmax denominator: cross-shard sum oracle (Stage 3)
#    sum_tok exp(score[tok,h]) accumulates the same way as score_times_v
# ---------------------------------------------------------------------------
def test_softmax_denom_cross_shard_sum():
    """Stage 3 oracle: denominator = sum_shards sum_{tok in shard} exp(s[tok,h])."""
    rng = np.random.default_rng(88)
    for real_nt in [8, 44, 128]:
        P = P_FULL
        n_shards = math.ceil(real_nt / P)
        # arbitrary scores (post-QKT, pre-softmax)
        scores_raw = rng.standard_normal((real_nt, N_HEADS))
        exp_scores = np.exp(scores_raw - scores_raw.max(axis=0))  # [real_nt, N_HEADS]

        # Reference: straight sum per head
        ref_denom = exp_scores.sum(axis=0)  # [N_HEADS]

        # Oracle: accumulate per-shard partial sums then cross-shard add
        total_denom = np.zeros(N_HEADS, dtype=np.float64)
        for b in range(n_shards):
            tok_start = b * P
            tok_end = min(tok_start + P, real_nt)
            partial = exp_scores[tok_start:tok_end, :].sum(axis=0)
            total_denom += partial

        err = np.max(np.abs(total_denom - ref_denom))
        assert err < 1e-12, f"denom cross-shard sum err={err:.3e} real_nt={real_nt}"
        print(f"[ok] softmax_denom real_nt={real_nt} n_shards={n_shards} err={err:.2e}")
    print("[ok] test_softmax_denom_cross_shard_sum")


# ---------------------------------------------------------------------------
# Geometry micro-tests: verify the primitive slot-ops directly
# ---------------------------------------------------------------------------
def test_inner_sum_geometry():
    """_inner_sum_slots at base slot = sum of d_head contiguous values."""
    H = 4
    # construct: slot[h*H + j] = value_h for all j
    D = 3 * H  # 3 heads
    slots = np.zeros(D, dtype=np.float64)
    expected = []
    for h in range(3):
        vals = [float(h * 10 + j) for j in range(H)]
        slots[h * H:(h + 1) * H] = vals
        expected.append(sum(vals))
    after = _inner_sum_slots(slots, H)
    for h in range(3):
        base = h * H
        got = after[base]
        assert abs(got - expected[h]) < 1e-12, \
            f"inner_sum head {h}: got {got}, expected {expected[h]}"
    print("[ok] test_inner_sum_geometry")


def test_broadcast_within_heads_geometry():
    """_broadcast_within_heads copies base-slot value across d_head block."""
    H = 8
    D = 2 * H
    P = 3
    slots = np.zeros(P * D, dtype=np.float64)
    # place fingerprint at base slots only
    fingerprints = {}
    for tok in range(P):
        for h in range(2):
            base = tok * D + h * H
            v = float(tok * 100 + h * 10)
            slots[base] = v
            fingerprints[(tok, h)] = v
    after = _broadcast_within_heads(slots, H)
    for tok in range(P):
        for h in range(2):
            v = fingerprints[(tok, h)]
            for j in range(H):
                got = after[tok * D + h * H + j]
                assert abs(got - v) < 1e-12, \
                    f"broadcast tok={tok} h={h} j={j}: got {got}, expected {v}"
    print("[ok] test_broadcast_within_heads_geometry")


def test_accumulate_geometry():
    """_accumulate_over_positions folds P frames into frame 0."""
    H = 4
    D = 2 * H
    P = 4
    # slot[tok*D + h*H + j] = (tok+1) * 1.0 (same for all h,j within a frame)
    slots = np.zeros(P * D, dtype=np.float64)
    for tok in range(P):
        slots[tok * D:(tok + 1) * D] = float(tok + 1)
    after = _accumulate_over_positions(slots, P, D)
    # frame 0 should hold sum_{tok=0}^{P-1}(tok+1) = 1+2+3+4 = 10
    expected_sum = sum(range(1, P + 1))
    for h in range(2):
        for j in range(H):
            got = after[h * H + j]  # tok=0 frame
            assert abs(got - expected_sum) < 1e-12, \
                f"accumulate h={h} j={j}: got {got}, expected {expected_sum}"
    print("[ok] test_accumulate_geometry")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    fns = [
        test_next_pow2,
        test_q_roundtrip_bitexact,
        test_kv_roundtrip_bitexact_and_pad_zero,
        test_inner_sum_geometry,
        test_broadcast_within_heads_geometry,
        test_accumulate_geometry,
        test_dense_qkt_matches_einsum,
        test_dense_score_v_matches_einsum,
        test_gqa_head_mapping,
        test_multi_shard_consistency,
        test_softmax_denom_cross_shard_sum,
    ]
    for fn in fns:
        fn()
    print(f"\nALL DENSE-LAYOUT TESTS PASSED ({len(fns)})")


if __name__ == "__main__":
    _run_all()
