"""Disk persistence for SingleChainPlaintext IRP caches.

Used by mrpc_sweep_parallel.py to survive process restarts:

  rp_indep_cache (~36 GB) -- built once via encode_layer_rp_indep_irps
    across 32 layers; takes ~7.5 min from cold. Persists to
    /tmp/phantom_irp_cache/rp_indep/.

  shared_wq_cache (~160 GB at full sweep) -- built lazily per
    num_tokens during the 408-MRPC sweep. Persists each
    (num_tokens, layer_idx) entry to /tmp/phantom_irp_cache/wq/nt_{N}/.

On-disk layout (one directory per cache, atomic via tempfile+rename):

  <root>/
    index.json          -- version tag + structure metadata
    <key>.bin           -- one file per SCP holding raw int16 bytes

The index.json records:
  {
    "version": 1,
    "N": 65536,
    "entries": [
      {"path": "lay00__diag_wo__000.bin", "scale": 1.34e34, ...},
      ...
    ],
    "tree": <nested dict of paths>
  }

"tree" mirrors the in-memory dict structure with SCP leaves replaced by
{"__scp__": "<entry_idx>"}. Lists, None, and ints/tuples are preserved.

Tuples are encoded as {"__tuple__": [...]} since JSON has no native tuple.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any

import pyPhantom as phantom


# v4: per-SCP .bin files hold block-floating-point int8 payloads ([N int8][N/B
# fp32]) and the index carries coeff_scale + is_int8_bfp + block_size. Bumped
# from v2 (int16) / v1 (int64) so stale caches are rejected and cold-re-encoded.
CACHE_VERSION = 4


def _is_scp(obj: Any) -> bool:
    """phantom.single_chain_plaintext check that doesn't rely on isinstance
    against a pybind11 class (which works, but this is more forgiving)."""
    return hasattr(obj, "coeffs_bytes") and hasattr(obj, "get_scale") and \
           hasattr(obj, "N")


def _walk_collect(obj: Any, entries: list, path_prefix: str) -> Any:
    """Recursively walk obj. Replace SCP leaves with placeholder dicts and
    append (path, scale, N, bytes) tuples to `entries`. Returns a tree of
    plain JSON-encodable values (lists, dicts, tuples-as-dicts, ints, etc.)."""
    if _is_scp(obj):
        entry_idx = len(entries)
        fname = f"{path_prefix}__{entry_idx:06d}.bin"
        entries.append({
            "fname": fname,
            "scale": float(obj.get_scale()),
            "coeff_scale": float(obj.get_coeff_scale()),
            "block_size": int(obj.get_block_size()),
            "N": int(obj.N),
            "bytes": obj.coeffs_bytes(),
        })
        return {"__scp__": entry_idx}
    if obj is None:
        return None
    if isinstance(obj, bool) or isinstance(obj, int) or isinstance(obj, float) \
       or isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return [_walk_collect(v, entries, f"{path_prefix}_{i}")
                for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return {"__tuple__": [_walk_collect(v, entries, f"{path_prefix}_{i}")
                              for i, v in enumerate(obj)]}
    if isinstance(obj, dict):
        return {str(k): _walk_collect(v, entries, f"{path_prefix}_{k}")
                for k, v in obj.items()}
    raise TypeError(
        f"save_scp_dict_to_disk: unsupported type {type(obj).__name__} "
        f"at {path_prefix!r}; supported: dict, list, tuple, SCP, None, "
        f"bool, int, float, str.")


def _walk_rebuild(tree: Any, entries_meta: list, root: str) -> Any:
    """Inverse of _walk_collect. Loads SCP bytes from <root>/<fname> and
    rebuilds the original dict/list/tuple tree."""
    if tree is None or isinstance(tree, (bool, int, float, str)):
        return tree
    if isinstance(tree, dict):
        if "__scp__" in tree:
            meta = entries_meta[tree["__scp__"]]
            path = os.path.join(root, meta["fname"])
            with open(path, "rb") as f:
                data = f.read()
            return phantom.scp_from_bytes(
                data, meta["scale"], meta["N"],
                meta.get("coeff_scale", meta["scale"]),
                meta.get("block_size", 0))
        if "__tuple__" in tree:
            return tuple(_walk_rebuild(v, entries_meta, root)
                         for v in tree["__tuple__"])
        # Plain dict: JSON keys are strings; attempt int() promotion so the
        # rebuilt cache keys (layer_idx, num_tokens) match in-memory types.
        out = {}
        for k, v in tree.items():
            try:
                kk = int(k)
            except (TypeError, ValueError):
                kk = k
            out[kk] = _walk_rebuild(v, entries_meta, root)
        return out
    if isinstance(tree, list):
        return [_walk_rebuild(v, entries_meta, root) for v in tree]
    raise TypeError(f"_walk_rebuild: unexpected type {type(tree).__name__}")


def save_scp_dict_to_disk(scp_dict: Any, path: str) -> None:
    """Serialize an arbitrary dict-of-SCPs (or dict-of-dicts) to a directory.

    Format: index.json (structure + scales + sizes) plus one .bin per SCP
    (raw int16 bytes). Atomic via tempfile + rename: the destination
    directory is fully populated under a sibling temp dir, then renamed
    over `path`. If `path` already exists, it is replaced.
    """
    entries = []
    tree = _walk_collect(scp_dict, entries, path_prefix="entry")

    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=".scp_cache_tmp_", dir=parent)
    try:
        for e in entries:
            fp = os.path.join(tmp_dir, e["fname"])
            with open(fp, "wb") as f:
                f.write(e["bytes"])
        # Build index.json with bytes stripped (keep only metadata). coeff_scale
        # + block_size are required to reconstruct int16 (scale_2) and int8
        # block-FP SCPs respectively.
        index = {
            "version": CACHE_VERSION,
            "entries": [{"fname": e["fname"], "scale": e["scale"],
                         "coeff_scale": e["coeff_scale"],
                         "block_size": e["block_size"], "N": e["N"]}
                        for e in entries],
            "tree": tree,
        }
        with open(os.path.join(tmp_dir, "index.json"), "w") as f:
            json.dump(index, f)
        # Atomic publish.
        if os.path.exists(path):
            # Remove the existing target then rename. (os.rename onto a
            # non-empty dir is not portable; shutil.rmtree first is fine.)
            shutil.rmtree(path)
        os.rename(tmp_dir, path)
        tmp_dir = None
    finally:
        if tmp_dir is not None and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def load_scp_dict_from_disk(path: str) -> Any:
    """Inverse of save_scp_dict_to_disk. Returns the same dict structure
    with SCPs rebuilt via phantom.scp_from_bytes. Raises FileNotFoundError
    if `path` does not exist; raises RuntimeError on version mismatch."""
    idx_path = os.path.join(path, "index.json")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"SCP disk cache index missing: {idx_path}")
    with open(idx_path) as f:
        index = json.load(f)
    if index.get("version") != CACHE_VERSION:
        raise RuntimeError(
            f"SCP disk cache version mismatch at {path}: "
            f"got {index.get('version')}, expected {CACHE_VERSION}")
    return _walk_rebuild(index["tree"], index["entries"], path)


def has_cache(path: str) -> bool:
    """True if `path/index.json` exists and the version matches."""
    idx_path = os.path.join(path, "index.json")
    if not os.path.exists(idx_path):
        return False
    try:
        with open(idx_path) as f:
            index = json.load(f)
        return index.get("version") == CACHE_VERSION
    except (json.JSONDecodeError, OSError):
        return False
