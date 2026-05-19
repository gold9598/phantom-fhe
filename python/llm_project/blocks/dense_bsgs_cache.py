"""Disk-persistent cache for dense BSGS pre-encoded diagonals.

Env-gated by DENSE_BSGS_DISK_CACHE (default OFF -> the wrappers call
phantom.pre_encode_bsgs_diagonals directly: byte-identical to the
pre-cache dense pipeline; ZERO behavior change when unset).

When DENSE_BSGS_DISK_CACHE=1:
  miss -> phantom.pre_encode_bsgs_diagonals(...) (the ~44 s/layer encode),
          then persist {"diagonals": [SCP,...], "d_pad", "baby_steps"} to
          disk via blocks.scp_disk_cache (the existing atomic SCP infra,
          identical to the deleted IRP cache).
  hit  -> load the SCP list + (d_pad, baby_steps), reconstruct the opaque
          phantom.bsgs_diagonals via the binding ctor
          phantom.bsgs_diagonals(scps, d_pad, baby_steps) (giant_steps is
          recomputed C++-side as d_pad // baby_steps, exactly src/bsgs.cu).
          The ~44 s encode is replaced by an SCP-bytes load.

Cache root: <repo>/cache/dense_bsgs/ (repo-root cache/ is gitignored;
multi-GB — NEVER git-add). One subdir per cache key.

Cache key (per call site, see llama3_mrpc.py):
  Wq_baked : wq_L{layer}_q{P_local}_d{d_pad}_b{baby}_s{scale_tag}
             R_P (rope@query) is baked into Wq_baked -> the key MUST
             include P_local/query_position (the wq-cache-key bug lesson:
             a stale R_P-independent Wq key silently poisons fixed-nt).
  Wo / gate / up / down : {tag}_L{layer}_d{d_pad}_b{baby}_s{scale_tag}
             R_P-independent -> layer-keyed only.
"""
from __future__ import annotations

import os

import numpy as np
import pyPhantom as phantom

from blocks import scp_disk_cache


def _enabled() -> bool:
    return os.environ.get("DENSE_BSGS_DISK_CACHE", "0") == "1"


def _cache_root() -> str:
    # blocks/ -> python/llm_project/ -> python/ -> phantom-fhe/  (repo root)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(repo_root, "cache", "dense_bsgs")


def _scale_tag(scale: float) -> str:
    # Stable, filesystem-safe scale tag (e.g. 2^40 -> "1p1e12"-ish);
    # repr round-trips exactly, hex of the float bits is unambiguous.
    import struct
    return struct.pack("<d", float(scale)).hex()


def _key_dir(key: str) -> str:
    return os.path.join(_cache_root(), key)


def _diags_to_payload(diags) -> dict:
    """Opaque phantom.bsgs_diagonals -> JSON/SCP-serializable dict."""
    return {
        "diagonals": list(diags.diagonals),
        "d_pad": int(diags.d_pad),
        "baby_steps": int(diags.baby_steps),
    }


def _payload_to_diags(payload: dict):
    """Reconstruct phantom.bsgs_diagonals from a loaded payload via the
    binding ctor (giant_steps recomputed C++-side as d_pad // baby_steps)."""
    return phantom.bsgs_diagonals(
        payload["diagonals"],
        int(payload["d_pad"]),
        int(payload["baby_steps"]),
    )


def _cached(key: str, encode_fn):
    """Generic: return cached bsgs_diagonals for `key`, else encode_fn()
    (which must return a phantom.bsgs_diagonals) and persist it.

    DENSE_BSGS_DISK_CACHE unset -> just encode_fn() (no disk touch).
    """
    if not _enabled():
        return encode_fn()
    path = _key_dir(key)
    if scp_disk_cache.has_cache(path):
        payload = scp_disk_cache.load_scp_dict_from_disk(path)
        return _payload_to_diags(payload)
    diags = encode_fn()
    scp_disk_cache.save_scp_dict_to_disk(_diags_to_payload(diags), path)
    return diags


# ---------------------------------------------------------------------------
# Public wrappers — drop-in for the dense call sites in llama3_mrpc.py
# ---------------------------------------------------------------------------

def wq_diags_cached(ctx, encoder, Wq_baked: np.ndarray, d_pad: int,
                    baby_steps: int, scale: float, layer_idx: int,
                    P_local: int):
    """Wq_baked (R_P-baked) -> BSGS diagonals. Key includes P_local."""
    num_rows, num_cols = Wq_baked.shape
    matrix_flat = np.ascontiguousarray(
        Wq_baked, dtype=np.float64).ravel().tolist()
    key = (f"wq_L{int(layer_idx)}_q{int(P_local)}_d{int(d_pad)}"
           f"_b{int(baby_steps)}_s{_scale_tag(scale)}")
    return _cached(key, lambda: phantom.pre_encode_bsgs_diagonals(
        ctx, encoder, matrix_flat, num_rows, num_cols,
        d_pad, baby_steps, scale))


def matrix_diags_cached(ctx, encoder, W: np.ndarray, num_rows: int,
                         num_cols: int, d_pad: int, baby_steps: int,
                         scale: float, tag: str, layer_idx: int):
    """R_P-independent matrix (Wo / gate / up / down) -> BSGS diagonals.
    Layer-keyed only (no P_local — these never depend on R_P)."""
    matrix_flat = np.ascontiguousarray(
        W, dtype=np.float64).ravel().tolist()
    key = (f"{tag}_L{int(layer_idx)}_d{int(d_pad)}"
           f"_b{int(baby_steps)}_s{_scale_tag(scale)}")
    return _cached(key, lambda: phantom.pre_encode_bsgs_diagonals(
        ctx, encoder, matrix_flat, num_rows, num_cols,
        d_pad, baby_steps, scale))
