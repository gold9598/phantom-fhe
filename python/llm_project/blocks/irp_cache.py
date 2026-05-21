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

On-disk format (version 2, one blob file per weight):
  8-byte magic  : b'IRPCV2\n\x00'
  4-byte uint32 : header_len (little-endian)
  header_len bytes : UTF-8 JSON {"version":2,"count":K,"scale":f,"N":n}
  K * (N * 8) bytes : raw int64 SCP coefficients back-to-back

Cold run: encodes K SCPs, writes ONE blob file via tmp+rename.
Warm run: opens ONE file, mmap's it, slices K contiguous N*8-byte
chunks, calls phantom.scp_from_bytes per chunk — typically 200-300 MB/s
off NVMe with zero per-SCP open overhead.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import tempfile

from blocks import irp as _irp

import pyPhantom as phantom


_MAGIC = b'IRPCV2\n\x00'
_CACHE_VERSION = 2


def _scale_tag(scale: float) -> str:
    return struct.pack("<d", float(scale)).hex()


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
    N = int(plaintexts[0].N)
    scp_bytes = N * 8  # each SCP: N int64 values

    header_bytes = json.dumps(
        {"version": _CACHE_VERSION, "count": K, "scale": scale, "N": N}
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
            N = int(header["N"])
            scp_bytes = N * 8
            data_offset = 12 + header_len
            plaintexts = []
            for i in range(K):
                start = data_offset + i * scp_bytes
                chunk = bytes(mm[start:start + scp_bytes])
                plaintexts.append(phantom.scp_from_bytes(chunk, scale, N))
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


def _cached_square(key, ctx, encoder, matrix, N, d, scale, baby_steps):
    path = _blob_path(key)
    if _has_blob(path):
        return _load_blob(path)
    pts = _irp.encode_irp_diagonals_host(
        ctx, encoder, matrix, N=N, d=d, scale=scale, baby_steps=baby_steps)
    _save_blob(path, pts)
    return pts


def _cached_rect(key, ctx, encoder, matrix, N, d_in, d_out, scale, baby_steps):
    path = _blob_path(key)
    if _has_blob(path):
        return _load_blob(path)
    pts = _irp.encode_irp_diagonals_rect_host(
        ctx, encoder, matrix, N=N, d_in=d_in, d_out=d_out,
        scale=scale, baby_steps=baby_steps)
    _save_blob(path, pts)
    return pts


def _cached_rect_folded(key, ctx, encoder, matrix, N, d_in, d_out, scale,
                         baby_steps):
    """Cache wrapper for the complex output-folded rect encoding (K/2 SCPs)."""
    path = _blob_path(key)
    if _has_blob(path):
        return _load_blob(path)
    pts = _irp.encode_irp_diagonals_rect_folded_host(
        ctx, encoder, matrix, N=N, d_in=d_in, d_out=d_out,
        scale=scale, baby_steps=baby_steps)
    _save_blob(path, pts)
    return pts


def wq_plaintexts_cached(ctx, encoder, Wq_baked_T, N, d, scale, baby_steps,
                          layer_idx, P_local):
    """Wq_baked.T (R_P-baked, transposed) -> IRP SCPs. Key includes P_local."""
    key = (f"wq_L{int(layer_idx)}_q{int(P_local)}_d{int(d)}"
           f"_b{int(baby_steps)}_s{_scale_tag(scale)}")
    return _cached_square(key, ctx, encoder, Wq_baked_T, N, d, scale, baby_steps)


def wo_plaintexts_cached(ctx, encoder, Wo_T, N, d, scale, baby_steps,
                          layer_idx):
    """Wo.T -> IRP SCPs. R_P-independent, layer-keyed only."""
    key = (f"wo_L{int(layer_idx)}_d{int(d)}"
           f"_b{int(baby_steps)}_s{_scale_tag(scale)}")
    return _cached_square(key, ctx, encoder, Wo_T, N, d, scale, baby_steps)


def gate_plaintexts_cached(ctx, encoder, Wgate_padded, N, d_in, d_out, scale,
                            baby_steps, layer_idx):
    """Wgate (padded wide) -> complex output-FOLDED IRP-rect SCPs (K/2).

    `_fold` key tag keeps the halved blobs distinct from stale unfolded ones.
    Consume with irp_matvec_rect_folded_host + extract_real_imag_pair +
    interleave_recombine.
    """
    key = (f"gate_L{int(layer_idx)}_di{int(d_in)}_do{int(d_out)}_fold"
           f"_b{int(baby_steps)}_s{_scale_tag(scale)}")
    return _cached_rect_folded(key, ctx, encoder, Wgate_padded, N, d_in, d_out,
                               scale, baby_steps)


def up_plaintexts_cached(ctx, encoder, Wup_padded, N, d_in, d_out, scale,
                          baby_steps, layer_idx):
    """Wup (padded wide) -> complex output-FOLDED IRP-rect SCPs (K/2).

    `_fold` key tag keeps the halved blobs distinct from stale unfolded ones.
    """
    key = (f"up_L{int(layer_idx)}_di{int(d_in)}_do{int(d_out)}_fold"
           f"_b{int(baby_steps)}_s{_scale_tag(scale)}")
    return _cached_rect_folded(key, ctx, encoder, Wup_padded, N, d_in, d_out,
                               scale, baby_steps)


def down_plaintexts_cached(ctx, encoder, Wdown_padded, N, d_in, d_out, scale,
                            baby_steps, layer_idx, gate_up_d_in, gate_up_d_out):
    """Wdown (padded tall) -> UNFOLDED IRP-rect SCPs, rows ROW-PERMUTED to
    absorb the gate/up interleave layout.

    The MLP fold path feeds Wdown an interleaved-layout input produced by
    interleave_recombine on the gate/up folded matvecs of dims
    (gate_up_d_in, gate_up_d_out). interleave_output_order returns the row
    permutation that makes this tall matvec consume the interleaved layout and
    emit NATURAL order (no un-permute). `_perm` key tag keeps it distinct from
    the old naturally-ordered down blob.
    """
    order = _irp.interleave_output_order(
        N, gate_up_d_in, gate_up_d_out, down_d_in=d_in, down_d_out=d_out)
    Wdown_perm = Wdown_padded[order, :]
    key = (f"down_L{int(layer_idx)}_di{int(d_in)}_do{int(d_out)}_perm"
           f"_b{int(baby_steps)}_s{_scale_tag(scale)}")
    return _cached_rect(key, ctx, encoder, Wdown_perm, N, d_in, d_out,
                        scale, baby_steps)
