"""Build the rp_indep IRP disk cache layer-by-layer.

Each layer's SCPs are encoded → saved to disk → freed before the next
layer is loaded. Peak in-RAM footprint is bounded to ~4-5 GB (one
layer's pinned host SCPs + transient encoder scratch), so this fits
the 62 GB 5090 dev box without OOMing.

Output layout (different from the legacy single-dict
save_scp_dict_to_disk format used by mrpc_sweep_parallel):

  <out>/MANIFEST.json                       # {"layout":"per-layer","num_layers":32, ...}
  <out>/layer_00/index.json                 # one save_scp_dict_to_disk call
  <out>/layer_00/*.bin
  ...
  <out>/layer_31/...

The companion loader `load_rp_indep_per_layer(root)` walks the
per-layer subdirs and merges the SCPs back into a single dict that
matches what `mrpc_sweep_parallel._build_shared_rp_indep_cache` expects.

Usage:
  python build_disk_cache.py              # writes <repo>/cache/rp_indep
  python build_disk_cache.py --out /path  # custom destination
"""
import argparse
import ctypes
import gc
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MALLOC_ARENA_MAX", "2")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, os.path.join(_REPO, "build", "lib"))
sys.path.insert(0, _THIS_DIR)

import pyPhantom as phantom  # noqa: E402

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


def _layer_dir(out_root, layer_idx):
    return os.path.join(out_root, f"layer_{layer_idx:02d}")


def has_per_layer_cache(out_root, num_layers):
    """Return True if every per-layer subdir exists and looks valid."""
    if not os.path.isdir(out_root):
        return False
    manifest_path = os.path.join(out_root, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        return False
    try:
        from blocks.scp_disk_cache import has_cache  # noqa: WPS433
    except Exception:
        return False
    for L in range(num_layers):
        if not has_cache(_layer_dir(out_root, L)):
            return False
    return True


def load_rp_indep_per_layer(out_root):
    """Inverse of the per-layer save: iterate <out>/layer_NN/ and merge
    into a single {layer_idx -> tuple} dict matching what
    mrpc_sweep_parallel._build_shared_rp_indep_cache returns."""
    from blocks.scp_disk_cache import load_scp_dict_from_disk
    manifest_path = os.path.join(out_root, "MANIFEST.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    num_layers = int(manifest["num_layers"])
    cache = {}
    for L in range(num_layers):
        sub = load_scp_dict_from_disk(_layer_dir(out_root, L))
        # `sub` is a dict {L: result_tuple}; merge in.
        cache.update(sub)
    if len(cache) != num_layers:
        raise RuntimeError(
            f"load_rp_indep_per_layer: expected {num_layers} entries, "
            f"got {len(cache)} keys: {sorted(cache.keys())[:5]}..")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.path.join(_REPO, "cache", "rp_indep"),
        help="Output directory. Default: <repo>/cache/rp_indep")
    ap.add_argument(
        "--no-pack-gate-up", action="store_true",
        help="Encode gate + up as separate IRPs (legacy). Default: packed.")
    ap.add_argument(
        "--num-decoders", type=int, default=32)
    ap.add_argument(
        "--gpu", type=int, default=0)
    ap.add_argument(
        "--start", type=int, default=0,
        help="Layer index to start at. Skips layers already saved on disk.")
    args = ap.parse_args()

    from helpers.llama3 import load_layer_weights, encode_layer_rp_indep_irps
    from fhe.llama3_mrpc import build_user_steps_mrpc, setup_engine
    from blocks.scp_disk_cache import save_scp_dict_to_disk, has_cache

    os.makedirs(args.out, exist_ok=True)
    pack_gate_up = not args.no_pack_gate_up

    print(f"[build_disk_cache] target: {args.out}", flush=True)
    print(f"[build_disk_cache] pack_gate_up: {pack_gate_up}", flush=True)

    # Build CKKS engine once.
    phantom.set_cuda_device(args.gpu)
    user_steps, step_categories = build_user_steps_mrpc()
    print(f"[build_disk_cache] building engine on GPU {args.gpu}...",
          flush=True)
    t_eng0 = time.perf_counter()
    engine = setup_engine(user_steps, step_categories=step_categories)
    ctx = engine.context()
    encoder = engine.encoder()
    print(f"[build_disk_cache] engine built in "
          f"{time.perf_counter() - t_eng0:.1f}s", flush=True)

    # Encode + save each layer independently. Peak in-RAM cost stays at
    # ~one layer (~1.15 GB SCPs + transient ~2 GB encoder scratch).
    t_build0 = time.perf_counter()
    skipped = 0
    encoded = 0
    for L in range(args.num_decoders):
        if L < args.start:
            continue
        layer_path = _layer_dir(args.out, L)
        if has_cache(layer_path):
            print(f"[build_disk_cache] layer {L:02d}/{args.num_decoders}  "
                  f"SKIP (already on disk)", flush=True)
            skipped += 1
            continue

        t0 = time.perf_counter()
        w = load_layer_weights(L)
        result = encode_layer_rp_indep_irps(
            ctx, encoder, w, pack_gate_up=pack_gate_up)
        t_enc = time.perf_counter() - t0

        # Save this layer's SCPs to its own subdir, then drop refs.
        t_save0 = time.perf_counter()
        save_scp_dict_to_disk({L: result}, layer_path)
        t_save = time.perf_counter() - t_save0

        elapsed = time.perf_counter() - t0
        print(f"[build_disk_cache] layer {L:02d}/{args.num_decoders}  "
              f"encode={t_enc:.1f}s save={t_save:.1f}s total={elapsed:.1f}s",
              flush=True)
        encoded += 1

        del w, result
        gc.collect()
        _malloc_trim()

    # Write a manifest so the loader can detect the per-layer layout.
    manifest = {
        "layout": "per-layer",
        "num_layers": args.num_decoders,
        "pack_gate_up": pack_gate_up,
    }
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    t_total = time.perf_counter() - t_build0
    print(f"[build_disk_cache] done: encoded={encoded} skipped={skipped} "
          f"total wall={t_total:.1f}s. cache at {args.out}", flush=True)


if __name__ == "__main__":
    main()
