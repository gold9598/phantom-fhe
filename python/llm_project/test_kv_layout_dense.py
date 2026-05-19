"""Correctness gate for the dense K/V packer (Stage 1, pure numpy — no FHE).

Validates blocks/kv_layout_dense.py as the executable spec the later FHE
stages are checked against:

* pack -> unpack round-trip is bit-exact (np.array_equal) for non-pow2
  real_nt with nt_pad = next_pow2.
* the [real_nt, nt_pad) pad region is exactly 0.0.
* dense_qkt matches einsum('hd,thd->th', Q, K) / sqrt(d_head) within 1e-12.
* dense_score_v matches einsum('th,thd->hd', W, V) within 1e-12.
* GQA: query head h reads kv head h // n_kv_groups (repo block-repeat).

Runs under pytest or standalone (`python test_kv_layout_dense.py`).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks.kv_layout_dense import (
    next_pow2,
    pack_q_dense,
    unpack_q_dense,
    pack_kv_dense,
    unpack_kv_dense,
    pack_scores_dense,
    unpack_scores_dense,
    dense_qkt,
    dense_score_v,
)

# LLaMA-3.1-8B dims (llama3.py:70-75). Use the real GQA ratio (4).
N_HEADS = 32
N_KV_HEADS = 8
N_KV_GROUPS = N_HEADS // N_KV_HEADS  # 4
D_HEAD = 128

REAL_NT_CASES = [1, 8, 44, 60, 64, 128, 512]  # includes non-pow2: 44, 60


def _gqa_expand(X_t_kvh_d):
    """Reference GQA expansion = the repo's np.repeat(..., axis=1)."""
    return np.repeat(X_t_kvh_d, N_KV_GROUPS, axis=1)


def test_next_pow2():
    cases = {0: 1, 1: 1, 2: 2, 3: 4, 8: 8, 44: 64, 60: 64, 64: 64,
             128: 128, 129: 256, 512: 512, 513: 1024}
    for n, want in cases.items():
        got = next_pow2(n)
        assert got == want, f"next_pow2({n})={got}, want {want}"
    print("[ok] test_next_pow2")


def test_q_roundtrip_bitexact():
    rng = np.random.default_rng(11)
    Q = rng.standard_normal((N_HEADS, D_HEAD))
    slots = pack_q_dense(Q)
    assert slots.shape == (N_HEADS * D_HEAD,)
    Q_back = unpack_q_dense(slots, N_HEADS, D_HEAD)
    assert np.array_equal(Q, Q_back), "pack_q/unpack_q not bit-exact"
    print("[ok] test_q_roundtrip_bitexact")


def test_kv_roundtrip_bitexact_and_pad_zero():
    rng = np.random.default_rng(22)
    for real_nt in REAL_NT_CASES:
        nt_pad = next_pow2(real_nt)
        K = rng.standard_normal((real_nt, N_KV_HEADS, D_HEAD))
        V = rng.standard_normal((real_nt, N_KV_HEADS, D_HEAD))
        k_slots, v_slots = pack_kv_dense(K, V, real_nt, nt_pad, N_HEADS)

        exp_len = N_HEADS * D_HEAD * nt_pad
        assert k_slots.shape == (exp_len,), (real_nt, k_slots.shape, exp_len)
        assert v_slots.shape == (exp_len,)

        # round-trip == GQA-expanded inputs, bit-exact
        K_back, V_back = unpack_kv_dense(k_slots, v_slots, real_nt, nt_pad,
                                         N_HEADS, D_HEAD)
        K_exp = _gqa_expand(K)
        V_exp = _gqa_expand(V)
        assert np.array_equal(K_back, K_exp), f"K round-trip real_nt={real_nt}"
        assert np.array_equal(V_back, V_exp), f"V round-trip real_nt={real_nt}"

        # pad region [real_nt, nt_pad) is EXACTLY 0.0 for every (h, j) block
        if nt_pad > real_nt:
            head_stride = D_HEAD * nt_pad
            for h in range(N_HEADS):
                for j in range(D_HEAD):
                    b = h * head_stride + j * nt_pad
                    ktail = k_slots[b + real_nt:b + nt_pad]
                    vtail = v_slots[b + real_nt:b + nt_pad]
                    assert np.array_equal(ktail, np.zeros_like(ktail)), \
                        f"K pad nonzero real_nt={real_nt} h={h} j={j}"
                    assert np.array_equal(vtail, np.zeros_like(vtail)), \
                        f"V pad nonzero real_nt={real_nt} h={h} j={j}"
        print(f"[ok] kv roundtrip+pad real_nt={real_nt} nt_pad={nt_pad}")
    print("[ok] test_kv_roundtrip_bitexact_and_pad_zero")


def test_scores_roundtrip_bitexact_and_pad_zero():
    rng = np.random.default_rng(33)
    for real_nt in REAL_NT_CASES:
        nt_pad = next_pow2(real_nt)
        S = rng.standard_normal((real_nt, N_HEADS))
        slots = pack_scores_dense(S, real_nt, nt_pad, N_HEADS)
        assert slots.shape == (N_HEADS * nt_pad,)
        S_back = unpack_scores_dense(slots, real_nt, nt_pad, N_HEADS)
        assert np.array_equal(S, S_back), f"scores round-trip real_nt={real_nt}"
        if nt_pad > real_nt:
            for h in range(N_HEADS):
                tail = slots[h * nt_pad + real_nt:h * nt_pad + nt_pad]
                assert np.array_equal(tail, np.zeros_like(tail)), \
                    f"score pad nonzero real_nt={real_nt} h={h}"
    print("[ok] test_scores_roundtrip_bitexact_and_pad_zero")


def test_dense_qkt_matches_einsum():
    """dense_qkt == einsum('hd,thd->th', Q, K_expanded) / sqrt(d_head)."""
    rng = np.random.default_rng(44)
    inv_sqrt_d = 1.0 / np.sqrt(D_HEAD)
    for real_nt in REAL_NT_CASES:
        nt_pad = next_pow2(real_nt)
        Q = rng.standard_normal((N_HEADS, D_HEAD))
        K = rng.standard_normal((real_nt, N_KV_HEADS, D_HEAD))
        V = np.zeros_like(K)
        q_slots = pack_q_dense(Q)
        k_slots, _ = pack_kv_dense(K, V, real_nt, nt_pad, N_HEADS)

        got = dense_qkt(q_slots, k_slots, N_HEADS, D_HEAD, real_nt, nt_pad)

        K_exp = _gqa_expand(K)  # [real_nt, n_heads, d_head]
        ref = np.einsum('hd,thd->th', Q, K_exp) * inv_sqrt_d  # [real_nt, n_heads]

        err = np.max(np.abs(got - ref))
        assert got.shape == (real_nt, N_HEADS)
        assert err < 1e-12, f"dense_qkt err={err:.3e} real_nt={real_nt}"
        print(f"[ok] dense_qkt real_nt={real_nt} maxerr={err:.2e}")
    print("[ok] test_dense_qkt_matches_einsum")


def test_dense_score_v_matches_einsum():
    """dense_score_v == einsum('th,thd->hd', W, V_expanded)."""
    rng = np.random.default_rng(55)
    for real_nt in REAL_NT_CASES:
        nt_pad = next_pow2(real_nt)
        W = rng.standard_normal((real_nt, N_HEADS))      # softmax-ish weights
        K = np.zeros((real_nt, N_KV_HEADS, D_HEAD))
        V = rng.standard_normal((real_nt, N_KV_HEADS, D_HEAD))
        _, v_slots = pack_kv_dense(K, V, real_nt, nt_pad, N_HEADS)
        w_slots = pack_scores_dense(W, real_nt, nt_pad, N_HEADS)

        got = dense_score_v(w_slots, v_slots, N_HEADS, D_HEAD, real_nt, nt_pad)

        V_exp = _gqa_expand(V)  # [real_nt, n_heads, d_head]
        ref = np.einsum('th,thd->hd', W, V_exp)  # [n_heads, d_head]

        err = np.max(np.abs(got - ref))
        assert got.shape == (N_HEADS, D_HEAD)
        assert err < 1e-12, f"dense_score_v err={err:.3e} real_nt={real_nt}"
        print(f"[ok] dense_score_v real_nt={real_nt} maxerr={err:.2e}")
    print("[ok] test_dense_score_v_matches_einsum")


def test_gqa_head_mapping():
    """Query head h must read kv head h // n_kv_groups (block-repeat).

    Build K with a head-id fingerprint so each query head's recovered
    scores reveal exactly which kv head it pulled from.
    """
    real_nt, nt_pad = 8, 8
    # K[tok, kvh, j] = (kvh+1)*1000  -> constant per kv head
    K = np.zeros((real_nt, N_KV_HEADS, D_HEAD))
    for kvh in range(N_KV_HEADS):
        K[:, kvh, :] = (kvh + 1) * 1000.0
    V = np.zeros_like(K)
    # Q all-ones so sum_j Q*K = d_head * (kvh+1)*1000 before scale
    Q = np.ones((N_HEADS, D_HEAD))
    q_slots = pack_q_dense(Q)
    k_slots, _ = pack_kv_dense(K, V, real_nt, nt_pad, N_HEADS)

    scores = dense_qkt(q_slots, k_slots, N_HEADS, D_HEAD, real_nt, nt_pad,
                       inv_sqrt_d=1.0)  # disable scaling for exact id read
    # scores[tok, h] / d_head = (expected_kv_head + 1) * 1000
    for h in range(N_HEADS):
        recovered_kvh = int(round(scores[0, h] / D_HEAD / 1000.0)) - 1
        expected_kvh = h // N_KV_GROUPS
        assert recovered_kvh == expected_kvh, (
            f"query head {h}: read kv head {recovered_kvh}, "
            f"expected {expected_kvh}"
        )
    # cross-check against the repo's canonical np.repeat expansion
    K_exp = _gqa_expand(K)
    for h in range(N_HEADS):
        assert np.array_equal(K_exp[:, h, :], K[:, h // N_KV_GROUPS, :])
    print("[ok] test_gqa_head_mapping (h -> h//%d)" % N_KV_GROUPS)


def _run_all():
    fns = [
        test_next_pow2,
        test_q_roundtrip_bitexact,
        test_kv_roundtrip_bitexact_and_pad_zero,
        test_scores_roundtrip_bitexact_and_pad_zero,
        test_dense_qkt_matches_einsum,
        test_dense_score_v_matches_einsum,
        test_gqa_head_mapping,
    ]
    for fn in fns:
        fn()
    print("\nALL DENSE-LAYOUT TESTS PASSED (%d)" % len(fns))


if __name__ == "__main__":
    _run_all()
