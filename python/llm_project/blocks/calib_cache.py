"""Always-on disk cache for per-layer bootstrap calibration (no env flag).

Mirrors irp_cache.py: the cached call is the ONLY path per the project's
no-flag-gated-fallback policy. Cold runs populate the cache, warm runs load
from disk and skip both the ~1.4 GB load_layer_weights() and the numpy
shadow-forward in compute_layer_calib_n().

Cache root: <repo>/cache/calib/
Key schema:
  calib_L{layer_idx}_nt{num_tokens}_q{query_position}_x{hash16}
where hash16 is the first 16 hex chars of sha1(x_btd float64 bytes). The
input hash guards against a stale cache if the prompt / activation changes.

On-disk format: JSON with full float precision (repr round-trips exactly):
  {"z1": float, "z2": float, "max_abs": {str: float, ...}}
max_abs values are plain Python floats, so json (float() == repr round-trip)
reproduces them byte-identically — required because a single mistyped key or
truncated float silently shifts a bootstrap calibration and drifts rel-RMS.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

import numpy as np


def _cache_root() -> str:
    # blocks/ -> python/llm_project/ -> python/ -> phantom-fhe/ (repo root)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    root = os.path.join(repo_root, "cache", "calib")
    os.makedirs(root, exist_ok=True)
    return root


def _x_hash(x_btd) -> str:
    """Stable 16-hex hash of x_btd, independent of array memory layout."""
    arr = np.ascontiguousarray(x_btd, dtype=np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def _calib_path(x_btd, layer_idx, num_tokens, query_position) -> str:
    key = (f"calib_L{int(layer_idx)}_nt{int(num_tokens)}"
           f"_q{int(query_position)}_x{_x_hash(x_btd)}")
    return os.path.join(_cache_root(), key + ".json")


def _save(path: str, z1: float, z2: float, max_abs: dict) -> None:
    payload = {
        "z1": float(z1),
        "z2": float(z2),
        "max_abs": {str(k): float(v) for k, v in max_abs.items()},
    }
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".calib_tmp_", dir=parent)
    try:
        with os.fdopen(fd, "w") as f:
            # repr-precision floats: json.dump uses float.__repr__ which is
            # the shortest string that round-trips to the same float64.
            json.dump(payload, f)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _load(path: str):
    with open(path, "r") as f:
        payload = json.load(f)
    z1 = float(payload["z1"])
    z2 = float(payload["z2"])
    max_abs = {str(k): float(v) for k, v in payload["max_abs"].items()}
    return z1, z2, max_abs


def calib_cached(x_btd, layer_idx, num_tokens, query_position, compute_fn):
    """Return (z1, z2, max_abs) for the given layer/input.

    Warm: load from <repo>/cache/calib/. Cold/miss: call compute_fn() (which
    loads full layer weights + runs compute_layer_calib_n), save, return.
    """
    path = _calib_path(x_btd, layer_idx, num_tokens, query_position)
    if os.path.isfile(path):
        return _load(path)
    z1, z2, max_abs = compute_fn()
    _save(path, z1, z2, max_abs)
    return z1, z2, max_abs
