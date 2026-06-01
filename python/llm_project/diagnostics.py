"""Diagnostic / instrumentation helpers split out of llama3_mrpc.py.

design: doc/design/diagnostics.md#probe-dump-layer-list-identity
"""
import ctypes
import os

import numpy as np


# design: doc/design/diagnostics.md#malloc-trim-rationale
try:
    _LIBC = ctypes.CDLL("libc.so.6")
    _LIBC.malloc_trim.argtypes = [ctypes.c_size_t]
    _LIBC.malloc_trim.restype = ctypes.c_int
except Exception:
    _LIBC = None


def _malloc_trim():
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


_PROBE_DECRYPT_STAGES = os.environ.get("PROBE_DECRYPT_STAGES") == "1"
_PROBE_DUMP_DIR = os.environ.get("PROBE_DUMP_DIR", "/tmp/probe_stage_dump")
_PROBE_DUMP_LAYER = [None]  # set per-layer by run_classifier_fhe when verbose


def _probe(tag, ctx, encoder, sk, ct):
    v = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)),
                 dtype=np.float64)
    print(f"    [probe] {tag:30s} chain={ct.chain_index():2d} "
          f"max|.|={np.abs(v).max():.4e} mean|.|={np.abs(v).mean():.4e}")
    # design: doc/design/diagnostics.md#probe-decrypt-stages-dump
    if _PROBE_DECRYPT_STAGES and _PROBE_DUMP_LAYER[0] is not None:
        os.makedirs(_PROBE_DUMP_DIR, exist_ok=True)
        safe = tag.replace("/", "_").replace(" ", "_").replace("[", "").replace("]", "")
        np.save(f"{_PROBE_DUMP_DIR}/L{_PROBE_DUMP_LAYER[0]}__{safe}.npy", v)
