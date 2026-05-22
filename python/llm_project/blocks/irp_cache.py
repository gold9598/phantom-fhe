"""Always-on disk cache for IRP-encoded plaintexts (no env flag).

Mirrors dense_bsgs_cache.py but is unconditional per the project's
no-flag-gated-fallback policy. The cached call is the ONLY path; cold
runs populate the cache, warm runs load from disk.

Cache root: <repo>/cache/irp_diagonals/
Key schema:
  Wq    : wq_L{layer}_q{P_local}_d{d}_b{baby_steps}_s{scale_tag}
  Wo    : wo_L{layer}_d{d}_b{baby_steps}_s{scale_tag}
  Wgate : gate_L{layer}_di{d_in}_do{d_out}_b{baby_steps}_s{scale_tag}
  Wup   : up_L{layer}_di{d_in}_do{d_out}_b{baby_steps}_s{scale_tag}
  Wdown : down_L{layer}_di{d_in}_do{d_out}_b{baby_steps}_s{scale_tag}

On-disk format (version 4, one blob file per weight):
  8-byte magic  : b'IRPCV2\n\x00'
  4-byte uint32 : header_len (little-endian)
  header_len bytes : UTF-8 JSON
        {"version":4,"count":K,"scale":f,"coeff_scale":f,
         "is_int16":b,"is_int32":b,"N":n}
  K * (N * w) bytes : raw SCP coefficients back-to-back, where the per-coeff
        width w is 2 (int16), 4 (int32) or 8 (int64) per the storage flags.

Cold run: encodes K SCPs, writes ONE blob file via tmp+rename.
Warm run: opens ONE file, mmap's it, slices K contiguous N*w-byte
chunks, calls phantom.scp_from_bytes per chunk — typically 200-300 MB/s
off NVMe with zero per-SCP open overhead.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from blocks import irp as _irp

import pyPhantom as phantom


_MAGIC = b'IRPCV2\n\x00'
# v4 (quant-32bit): SCP coeffs stored at the adaptive width recorded by the
# is_int16/is_int32 header flags (N*2 / N*4 / N*8 B) with a per-blob coeff_scale.
# Bumped from v3 (int16-only @ 2^24) so stale int16 blobs are rejected by the
# version check and the cache cold-re-encodes at the int32 (2^32) scale.
_CACHE_VERSION = 4


def _scale_tag(scale: float) -> str:
    return struct.pack("<d", float(scale)).hex()


# ============================ RAM LRU cache + prefetch =====================
# Process-level LRU cache of loaded list-of-SCPs keyed by blob key (the same
# string the wrappers compute). A background ThreadPoolExecutor loads blobs
# (mmap + scp_from_bytes, encoder-FREE) one layer ahead so the next layer's
# weight LOAD overlaps the current layer's GPU compute. The encode (cold-miss)
# path uses the shared PhantomCKKSEncoder and therefore MUST stay on the main
# thread — the prefetcher only ever submits _load_blob for HITS (_has_blob).
#
# Eviction: keep ~3 layers resident (5 entries/layer → ~15). A layer's 5 blobs
# total ~1.6 GB, so ~3 layers ≈ ~5 GB host RAM, bounded.
_RAM_CACHE: "OrderedDict[str, object]" = OrderedDict()  # key -> list[SCP] | Future
_RAM_LOCK = threading.RLock()
_RAM_MAX_ENTRIES = 15  # ~3 layers (wq/wo/gate/up/down each)
# 2 workers: the SCP-loading is I/O + cheap parse, bounded so we never starve
# the main thread's CPU during compute.
_PREFETCH_POOL = ThreadPoolExecutor(max_workers=2,
                                    thread_name_prefix="irp_prefetch")


def _ram_evict_locked() -> None:
    """Drop oldest entries beyond the LRU bound. Caller holds _RAM_LOCK."""
    while len(_RAM_CACHE) > _RAM_MAX_ENTRIES:
        _RAM_CACHE.popitem(last=False)  # FIFO/LRU: oldest first


def _ram_get_or_load(key: str, path: str) -> list:
    """Return the loaded SCP list for `key`, awaiting a pending prefetch
    future or doing a synchronous disk load on a RAM miss. Never encodes.
    Always returns a concrete list (resolves futures)."""
    with _RAM_LOCK:
        entry = _RAM_CACHE.get(key)
        if entry is not None:
            # Touch for LRU recency.
            _RAM_CACHE.move_to_end(key)
    if entry is not None:
        if hasattr(entry, "result"):  # a pending/finished Future
            pts = entry.result()       # blocks until the bg load completes
        else:
            pts = entry
        # Replace any resolved future with the concrete list (idempotent).
        with _RAM_LOCK:
            _RAM_CACHE[key] = pts
            _RAM_CACHE.move_to_end(key)
            _ram_evict_locked()
        return pts
    # RAM miss: synchronous disk load on the main thread, then cache.
    pts = _load_blob(path)
    with _RAM_LOCK:
        _RAM_CACHE[key] = pts
        _RAM_CACHE.move_to_end(key)
        _ram_evict_locked()
    return pts


def _prefetch_one(key: str) -> None:
    """If the blob for `key` exists on disk and is not already resident/pending,
    submit a background _load_blob and store the future in the RAM cache.
    Encoder-free → thread-safe. A cold MISS is left to the synchronous wrapper."""
    path = _blob_path(key)
    if not _has_blob(path):
        return  # cold miss: encode path is main-thread-only; do not prefetch
    with _RAM_LOCK:
        if key in _RAM_CACHE:
            _RAM_CACHE.move_to_end(key)
            return  # already resident or pending
        fut = _PREFETCH_POOL.submit(_load_blob, path)
        _RAM_CACHE[key] = fut
        _RAM_CACHE.move_to_end(key)
        _ram_evict_locked()


def _layer_keys(layer_idx, P_local, d, mlp_d_in, mlp_d_out,
                scale, baby_steps_attn, baby_steps_mlp):
    """The 5 blob keys for a layer, via the same builders the wrappers use.
    Single source of truth so prefetch keys can never drift from cache keys."""
    return [
        _wq_key(layer_idx, P_local, d, baby_steps_attn, scale),
        _wo_key(layer_idx, d, baby_steps_attn, scale),
        _gate_key(layer_idx, mlp_d_in, mlp_d_out, baby_steps_mlp, scale),
        _up_key(layer_idx, mlp_d_in, mlp_d_out, baby_steps_mlp, scale),
        _down_key(layer_idx, mlp_d_out, mlp_d_in, baby_steps_mlp, scale),
    ]


def prefetch_layer(layer_idx, P_local, d, mlp_d_in, mlp_d_out,
                   scale, baby_steps_attn, baby_steps_mlp):
    """Background-load all 5 of `layer_idx`'s IRP blobs (HIT-only) so the next
    layer's weight LOAD overlaps the current layer's compute. Pure latency
    optimization: every blob is also fetchable synchronously by the wrappers,
    so a skipped/failed prefetch never affects correctness.

    NOTE the down key swaps (mlp_d_out, mlp_d_in): the down blob is keyed
    di{d_in=16384}_do{d_out=4096}, i.e. the TALL matvec's d_in/d_out — see
    down_plaintexts_cached. _down_key takes (di, do) in that order, so we pass
    (mlp_d_out=16384, mlp_d_in=4096) here. Both are constant per run."""
    try:
        for key in _layer_keys(layer_idx, P_local, d, mlp_d_in, mlp_d_out,
                               scale, baby_steps_attn, baby_steps_mlp):
            _prefetch_one(key)
    except Exception:
        # Prefetch is best-effort; never let it break the run.
        pass


def evict_layers_before(layer_idx, P_local, d, mlp_d_in, mlp_d_out,
                        scale, baby_steps_attn, baby_steps_mlp):
    """Explicitly drop RAM entries for layers strictly older than `layer_idx`.
    The LRU bound already caps memory; this is a belt-and-suspenders trim that
    keeps only the current + next layer's blobs resident when called with the
    just-finished layer index + 1 semantics from the loop."""
    keep = set()
    for li in (layer_idx, layer_idx + 1):
        keep.update(_layer_keys(li, P_local, d, mlp_d_in, mlp_d_out,
                                scale, baby_steps_attn, baby_steps_mlp))
    with _RAM_LOCK:
        for k in list(_RAM_CACHE.keys()):
            if k not in keep:
                _RAM_CACHE.pop(k, None)


# ===================== numpy per-layer weight prefetch ====================
# The big R_P-independent matrices (Wo / Wgate / Wup / Wdown) are read off disk
# as fp64 numpy arrays per layer (~1.4 GB/layer) inside the attention/MLP
# blocks. numpy load = pure disk + memcpy (no encoder), so it is thread-safe and
# can be prefetched one layer ahead alongside the IRP SCP blobs.
#
# Keyed by (layer_idx, frozenset(keys)). The loader callable is supplied by the
# caller (load_layer_weights_subset) to keep this module decoupled. The wrapper
# get_layer_weights() awaits a pending future or loads synchronously on a miss.
_NP_CACHE: "OrderedDict[tuple, object]" = OrderedDict()  # (li, fkeys) -> dict | Future
_NP_LOCK = threading.RLock()
_NP_MAX_ENTRIES = 6  # ~2-3 layers of (Wo) + (Wgate/Wup/Wdown) entries


def _np_evict_locked() -> None:
    while len(_NP_CACHE) > _NP_MAX_ENTRIES:
        _NP_CACHE.popitem(last=False)


def _np_cache_key(layer_idx, keys):
    return (int(layer_idx), frozenset(keys))


def get_layer_weights(layer_idx, keys, loader):
    """Return the numpy weight subset dict for (layer_idx, keys), awaiting a
    pending prefetch future or loading synchronously via `loader(layer_idx,
    keys)` on a miss. `loader` is pure disk/numpy (encoder-free)."""
    ck = _np_cache_key(layer_idx, keys)
    with _NP_LOCK:
        entry = _NP_CACHE.get(ck)
        if entry is not None:
            _NP_CACHE.move_to_end(ck)
    if entry is not None:
        res = entry.result() if hasattr(entry, "result") else entry
        with _NP_LOCK:
            _NP_CACHE[ck] = res
            _NP_CACHE.move_to_end(ck)
            _np_evict_locked()
        return res
    res = loader(layer_idx, list(keys))
    with _NP_LOCK:
        _NP_CACHE[ck] = res
        _NP_CACHE.move_to_end(ck)
        _np_evict_locked()
    return res


def prefetch_layer_weights(layer_idx, keys, loader):
    """Background-load the numpy weight subset for (layer_idx, keys) one layer
    ahead. Best-effort: get_layer_weights still loads synchronously on a miss."""
    try:
        ck = _np_cache_key(layer_idx, keys)
        with _NP_LOCK:
            if ck in _NP_CACHE:
                _NP_CACHE.move_to_end(ck)
                return
            fut = _PREFETCH_POOL.submit(loader, layer_idx, list(keys))
            _NP_CACHE[ck] = fut
            _NP_CACHE.move_to_end(ck)
            _np_evict_locked()
    except Exception:
        pass


def _cache_root() -> str:
    # blocks/ -> python/llm_project/ -> python/ -> phantom-fhe/ (repo root)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    root = os.path.join(repo_root, "cache", "irp_diagonals")
    os.makedirs(root, exist_ok=True)
    return root


def _blob_path(key: str) -> str:
    return os.path.join(_cache_root(), key + ".irpcv2")


def _save_blob(path: str, plaintexts: list) -> None:
    """Write K SCPs to a single blob file atomically (tmp + rename)."""
    if not plaintexts:
        raise ValueError("_save_blob: empty plaintexts list")
    K = len(plaintexts)
    scale = float(plaintexts[0].get_scale())
    coeff_scale = float(plaintexts[0].get_coeff_scale())
    is_int16 = bool(plaintexts[0].get_is_int16())
    is_int32 = bool(plaintexts[0].get_is_int32())
    N = int(plaintexts[0].N)
    # int16 (2 B) / int32 (4 B) quantized vs int64 (8 B) full-scale.
    scp_bytes = N * (2 if is_int16 else 4 if is_int32 else 8)

    header_bytes = json.dumps(
        {"version": _CACHE_VERSION, "count": K, "scale": scale,
         "coeff_scale": coeff_scale, "is_int16": is_int16,
         "is_int32": is_int32, "N": N}
    ).encode("utf-8")
    header_len = len(header_bytes)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".irpcv2_tmp_", dir=parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<I", header_len))
            f.write(header_bytes)
            for scp in plaintexts:
                f.write(scp.coeffs_bytes())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _load_blob(path: str) -> list:
    """Load K SCPs from a blob file using mmap for OS-level read-ahead."""
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            magic = mm[:8]
            if magic != _MAGIC:
                raise RuntimeError(
                    f"IRP cache magic mismatch at {path}: got {magic!r}")
            header_len = struct.unpack_from("<I", mm, 8)[0]
            header_raw = mm[12:12 + header_len]
            if hasattr(header_raw, 'tobytes'):
                header_raw = header_raw.tobytes()
            header = json.loads(header_raw)
            if header.get("version") != _CACHE_VERSION:
                raise RuntimeError(
                    f"IRP cache version mismatch at {path}: "
                    f"got {header.get('version')}, expected {_CACHE_VERSION}")
            K = int(header["count"])
            scale = float(header["scale"])
            coeff_scale = float(header.get("coeff_scale", scale))
            is_int16 = bool(header.get("is_int16", True))
            is_int32 = bool(header.get("is_int32", False))
            N = int(header["N"])
            scp_bytes = N * (2 if is_int16 else 4 if is_int32 else 8)
            data_offset = 12 + header_len
            plaintexts = []
            for i in range(K):
                start = data_offset + i * scp_bytes
                chunk = bytes(mm[start:start + scp_bytes])
                plaintexts.append(
                    phantom.scp_from_bytes(chunk, scale, N, coeff_scale))
            return plaintexts
        finally:
            mm.close()


def _has_blob(path: str) -> bool:
    """True if the blob file exists and has the correct magic."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            return f.read(8) == _MAGIC
    except OSError:
        return False


# ============================ Blob key builders ===========================
# Single source of truth for the on-disk key schema. Both the *_plaintexts_cached
# wrappers and prefetch_layer build keys here so they can never drift apart.
def _wq_key(layer_idx, P_local, d, baby_steps, scale):
    return (f"wq_L{int(layer_idx)}_q{int(P_local)}_d{int(d)}_fold"
            f"_b{int(baby_steps)}_s{_scale_tag(scale)}")


def _wq_unfold_key(layer_idx, P_local, d, baby_steps, scale):
    return (f"wq_L{int(layer_idx)}_q{int(P_local)}_d{int(d)}_unfold"
            f"_b{int(baby_steps)}_s{_scale_tag(scale)}")


def _wo_key(layer_idx, d, baby_steps, scale):
    return (f"wo_L{int(layer_idx)}_d{int(d)}_fold"
            f"_b{int(baby_steps)}_s{_scale_tag(scale)}")


def _wo_unfold_key(layer_idx, d, baby_steps, scale):
    return (f"wo_L{int(layer_idx)}_d{int(d)}_unfold"
            f"_b{int(baby_steps)}_s{_scale_tag(scale)}")


def _down_unfold_key(layer_idx, d_in, d_out, baby_steps, scale):
    return (f"down_L{int(layer_idx)}_di{int(d_in)}_do{int(d_out)}_unfoldperm"
            f"_b{int(baby_steps)}_s{_scale_tag(scale)}")


def _gate_key(layer_idx, d_in, d_out, baby_steps, scale):
    return (f"gate_L{int(layer_idx)}_di{int(d_in)}_do{int(d_out)}_fold"
            f"_b{int(baby_steps)}_s{_scale_tag(scale)}")


def _up_key(layer_idx, d_in, d_out, baby_steps, scale):
    return (f"up_L{int(layer_idx)}_di{int(d_in)}_do{int(d_out)}_fold"
            f"_b{int(baby_steps)}_s{_scale_tag(scale)}")


def _down_key(layer_idx, d_in, d_out, baby_steps, scale):
    return (f"down_L{int(layer_idx)}_di{int(d_in)}_do{int(d_out)}_fold"
            f"_b{int(baby_steps)}_s{_scale_tag(scale)}")


def _cached_square(key, ctx, encoder, matrix, N, d, scale, baby_steps):
    path = _blob_path(key)
    if _has_blob(path):
        return _ram_get_or_load(key, path)
    pts = _irp.encode_irp_diagonals_host(
        ctx, encoder, matrix, N=N, d=d, scale=scale, baby_steps=baby_steps)
    _save_blob(path, pts)
    return pts


def _cached_rect(key, ctx, encoder, matrix, N, d_in, d_out, scale, baby_steps):
    path = _blob_path(key)
    if _has_blob(path):
        return _ram_get_or_load(key, path)
    pts = _irp.encode_irp_diagonals_rect_host(
        ctx, encoder, _resolve_matrix(matrix), N=N, d_in=d_in, d_out=d_out,
        scale=scale, baby_steps=baby_steps)
    _save_blob(path, pts)
    return pts


def _resolve_matrix(matrix):
    """Resolve `matrix` to a concrete array. Accepts EITHER a ready array OR a
    0-arg callable (lazy loader). Only ever called on a COLD MISS, so on warm
    runs a callable's expensive disk-load + zero-pad never fires."""
    return matrix() if callable(matrix) else matrix


def _cached_rect_folded(key, ctx, encoder, matrix, N, d_in, d_out, scale,
                         baby_steps):
    """Cache wrapper for the complex output-folded rect encoding (K/2 SCPs).

    `matrix` may be a ready array OR a 0-arg loader callable; the loader fires
    ONLY on a cold disk miss (a hit returns the cached SCPs untouched)."""
    path = _blob_path(key)
    if _has_blob(path):
        return _ram_get_or_load(key, path)
    pts = _irp.encode_irp_diagonals_rect_folded_host(
        ctx, encoder, _resolve_matrix(matrix), N=N, d_in=d_in, d_out=d_out,
        scale=scale, baby_steps=baby_steps)
    _save_blob(path, pts)
    return pts


def _cached_square_folded(key, ctx, encoder, matrix, N, d, scale, baby_steps):
    """Cache wrapper for the complex output-folded SQUARE encoding (K/2 SCPs).

    Folds the output columns of a d×d weight into the imaginary part
    (d×d → d×(d/2) tall rect), halving the SCP count. Consume with
    irp_matvec_folded_host + extract_real_imag_pair.

    `matrix` may be a ready array OR a 0-arg loader callable; the loader fires
    ONLY on a cold disk miss (a hit returns the cached SCPs untouched).
    """
    path = _blob_path(key)
    if _has_blob(path):
        return _ram_get_or_load(key, path)
    pts = _irp.encode_irp_diagonals_folded_host(
        ctx, encoder, _resolve_matrix(matrix), N=N, d=d, scale=scale,
        baby_steps=baby_steps)
    _save_blob(path, pts)
    return pts


def wq_plaintexts_cached(ctx, encoder, Wq_baked_T, N, d, scale, baby_steps,
                          layer_idx, P_local):
    """Wq_baked.T (R_P-baked, transposed) -> complex output-FOLDED IRP SCPs
    (K/2). Key includes P_local. `_fold` tag keeps the halved blobs distinct
    from stale unfolded ones. Consume with irp_matvec_folded_host +
    extract_real_imag_pair.

    `Wq_baked_T` may be a ready array OR a 0-arg loader callable (lazy)."""
    key = _wq_key(layer_idx, P_local, d, baby_steps, scale)
    return _cached_square_folded(key, ctx, encoder, Wq_baked_T, N, d, scale,
                                 baby_steps)


def wo_plaintexts_cached(ctx, encoder, Wo_T, N, d, scale, baby_steps,
                          layer_idx):
    """Wo.T -> complex output-FOLDED IRP SCPs (K/2). R_P-independent,
    layer-keyed only. `_fold` tag keeps the halved blobs distinct from stale
    unfolded ones. Consume with irp_matvec_folded_host + extract_real_imag_pair.

    `Wo_T` may be a ready array OR a 0-arg loader callable (lazy)."""
    key = _wo_key(layer_idx, d, baby_steps, scale)
    return _cached_square_folded(key, ctx, encoder, Wo_T, N, d, scale,
                                 baby_steps)


def wq_unfolded_plaintexts_cached(ctx, encoder, Wq_baked_T, N, d, scale,
                                  baby_steps, layer_idx, P_local):
    """Wq_baked.T -> UNFOLDED square IRP SCPs (K) for the bridgeless Wq path.
    Consume with irp_matvec_host → q in IRP stride-(N/d) layout fed directly
    to compute_qkt_irp (no conj-split, no SK bridge). Key includes P_local."""
    key = _wq_unfold_key(layer_idx, P_local, d, baby_steps, scale)
    return _cached_square(key, ctx, encoder, _resolve_matrix(Wq_baked_T),
                          N, d, scale, baby_steps)


def wo_unfolded_plaintexts_cached(ctx, encoder, Wo_T, N, d, scale, baby_steps,
                                  layer_idx):
    """Wo.T -> UNFOLDED square IRP SCPs (K) for the bridgeless Wo path.
    Consume with irp_matvec_host → o in IRP stride-(N/d)=stride-T_MODEL layout
    directly fed to residual1 (no conj-split, no SK bridge)."""
    key = _wo_unfold_key(layer_idx, d, baby_steps, scale)
    return _cached_square(key, ctx, encoder, _resolve_matrix(Wo_T),
                          N, d, scale, baby_steps)


def down_unfolded_plaintexts_cached(ctx, encoder, Wdown_padded, N, d_in, d_out,
                                    scale, baby_steps, layer_idx,
                                    gate_up_d_in, gate_up_d_out):
    """Wdown (padded tall) -> UNFOLDED IRP-rect SCPs (K=2048), rows ROW-PERMUTED
    to absorb the gate/up interleave layout — for the bridgeless Wdown path.
    Row permute computed against the FULL d_out (not d_out//2), giving natural
    stride-(N/d_out) output consumable by residual2 with no SK bridge."""
    key = _down_unfold_key(layer_idx, d_in, d_out, baby_steps, scale)

    def _load_perm():
        order = _irp.interleave_output_order(
            N, gate_up_d_in, gate_up_d_out,
            down_d_in=d_in, down_d_out=d_out)
        return _resolve_matrix(Wdown_padded)[order, :]

    return _cached_rect(key, ctx, encoder, _load_perm, N, d_in, d_out,
                        scale, baby_steps)


def gate_plaintexts_cached(ctx, encoder, Wgate_padded, N, d_in, d_out, scale,
                            baby_steps, layer_idx):
    """Wgate (padded wide) -> complex output-FOLDED IRP-rect SCPs (K/2).

    `_fold` key tag keeps the halved blobs distinct from stale unfolded ones.
    Consume with irp_matvec_rect_folded_host + extract_real_imag_pair +
    interleave_recombine.

    `Wgate_padded` may be a ready array OR a 0-arg loader callable (lazy):
    on a warm hit the loader never fires, so the ~1.4 GB MLP weight load +
    zero-pad is skipped entirely.
    """
    key = _gate_key(layer_idx, d_in, d_out, baby_steps, scale)
    return _cached_rect_folded(key, ctx, encoder, Wgate_padded, N, d_in, d_out,
                               scale, baby_steps)


def up_plaintexts_cached(ctx, encoder, Wup_padded, N, d_in, d_out, scale,
                          baby_steps, layer_idx):
    """Wup (padded wide) -> complex output-FOLDED IRP-rect SCPs (K/2).

    `_fold` key tag keeps the halved blobs distinct from stale unfolded ones.

    `Wup_padded` may be a ready array OR a 0-arg loader callable (lazy).
    """
    key = _up_key(layer_idx, d_in, d_out, baby_steps, scale)
    return _cached_rect_folded(key, ctx, encoder, Wup_padded, N, d_in, d_out,
                               scale, baby_steps)


def down_plaintexts_cached(ctx, encoder, Wdown_padded, N, d_in, d_out, scale,
                            baby_steps, layer_idx, gate_up_d_in, gate_up_d_out):
    """Wdown (padded tall) -> complex output-FOLDED IRP-rect SCPs (K/2), rows
    ROW-PERMUTED to absorb the gate/up interleave layout.

    Two orthogonal transforms compose on Wdown:
      ROWS (d_in=16384): interleave_output_order absorbs the gate/up interleave
        layout — the folded TALL matvec consumes the interleaved h and emits
        the gate/up-natural output order. Applied first.
      COLUMNS (d_out=4096): encode_irp_diagonals_rect_folded_host folds the
        output columns into the imaginary part (K=2048→1024 SCPs, the biggest
        remaining fold). The matvec emits a complex ct split downstream by
        extract_real_imag_pair; the output SK bridge recombines to natural
        order in numpy and re-encrypts for residual2.

    Rows and columns are independent, so row-permute THEN column-fold gives
    h_il @ fold(Wdown_perm) == h @ Wdown (validated max|err| 1.7e-13). `_fold`
    key tag keeps the halved blobs distinct from the stale unfolded `_perm` blob.

    NOTE: the FOLDED matvec runs the rect machinery at d_out_fold = d_out/2,
    so it consumes its input in the TALL layout for (d_in, d_out/2) — a FINER
    permutation than the unfolded (d_in, d_out) layout. The row permutation
    that absorbs the interleave must therefore be computed against the FOLDED
    output dim d_out//2, not d_out (the unfolded d_out gives a layout mismatch
    that blows up the recombine).

    `Wdown_padded` may be a ready array OR a 0-arg loader callable (lazy). The
    row-permute (`[order, :]`) is wrapped into the lazy path so that on a warm
    hit neither the loader nor the permute touches the matrix.
    """
    key = _down_key(layer_idx, d_in, d_out, baby_steps, scale)

    def _load_perm():
        # COLD-MISS ONLY: resolve (possibly lazy-loaded) Wdown_padded, then
        # apply the interleave row-permute. Never fires on a warm hit.
        order = _irp.interleave_output_order(
            N, gate_up_d_in, gate_up_d_out,
            down_d_in=d_in, down_d_out=d_out // 2)
        return _resolve_matrix(Wdown_padded)[order, :]

    return _cached_rect_folded(key, ctx, encoder, _load_perm, N, d_in, d_out,
                               scale, baby_steps)
