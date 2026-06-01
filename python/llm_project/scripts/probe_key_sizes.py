"""Differential GPU memory measurement for engine, relin, bootstrap, and
user-rotation Galois keys.

Strategy: construct three engine variants with bootstrap and user rotations
toggled. The deltas isolate each key category. Run as:

    python3 python/llm_project/probe_key_sizes.py

Caveats:
- BootstrapTo17Levels skips bootstrap key construction but uses a DIFFERENT
  chain layout, so "engine + relin (no boot)" measured here is comparable
  to the standard engine's relin only up to the per-prime size difference.
  The bootstrap delta is still a faithful upper-bound on bootstrap-key cost.
- One run per config; allocator pool reuse across configs in a single
  process can confound deltas, so each variant runs as its own subprocess.
"""

import os
import subprocess
import sys
import time

import numpy as np

LOG_N = 16
NUM_SLOTS = (1 << LOG_N) // 2
SCALE = 2.0 ** 40
SPARSE_HW = 128
NUM_SCALE_LEVELS = 14
NUM_SPECIAL_PRIMES = 6


def gpu_used_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().strip().splitlines()[0])


def construct(*, with_bootstrap: bool, with_rotations: bool):
    """Construct a CKKS engine with the requested key categories. Returns
    GPU MiB after engine construction (so caller subtracts the pre-import
    baseline).

    `with_bootstrap=False` selects the BootstrapTo17Levels chain layout
    (NSL=18, NSP=8) which skips bootstrap-key creation. This is a
    DIFFERENT chain than the standard (NSL=14, NSP=6) so the engine and
    relin key are sized differently — interpret the bootstrap-key delta
    as approximate."""
    sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
    import pyPhantom as phantom

    pre_engine = gpu_used_mib()
    print(f"  pre-engine: {pre_engine} MiB", flush=True)

    cfg = phantom.ckks_engine_config()
    cfg.log_n = LOG_N
    cfg.user_scale = SCALE
    cfg.sparse_hw = SPARSE_HW
    cfg.include_user_rotations = False
    cfg.use_bootstrap_to_17_levels = not with_bootstrap

    if with_bootstrap:
        cfg.num_scale_levels = NUM_SCALE_LEVELS  # 14
        cfg.num_special_primes = NUM_SPECIAL_PRIMES  # 6
    else:
        cfg.num_scale_levels = 18
        cfg.num_special_primes = 8

    # Mirror llama3.py's per-chain step distribution (12@16 + 4@17 + 9@21 +
    # 2@23 + 42@26 = 69 total).  Most keys sit at chain 26 where partitions=1
    # so the bundle is small (~38 MiB each); a uniform distribution across
    # chains 16–29 OOMs at 32 GiB.
    if with_rotations:
        steps, chains = [], []
        # Distinct integer steps spread across the per-chain buckets.
        cursor = 1
        for n_at_chain, target in [(12, 16), (4, 17), (9, 21), (2, 23), (42, 26)]:
            for _ in range(n_at_chain):
                steps.append(cursor)
                chains.append(target)
                cursor += 1
        cfg.user_rotation_steps = steps
        cfg.user_rotation_target_chain_indices = chains
    else:
        cfg.user_rotation_steps = []
        cfg.user_rotation_target_chain_indices = []

    t0 = time.perf_counter()
    engine = phantom.ckks_engine(cfg)
    elapsed = time.perf_counter() - t0
    post_engine = gpu_used_mib()
    print(f"  engine ctor: {elapsed:.2f}s -> {post_engine} MiB "
          f"(delta over pre-engine: +{post_engine - pre_engine} MiB)",
          flush=True)

    # Keep engine alive briefly so the measurement is post-allocation.
    _ = engine.context()
    return post_engine, pre_engine


def main():
    if len(sys.argv) > 1:
        # Subprocess mode: run a single config.
        with_boot = sys.argv[1] == "1"
        with_rot = sys.argv[2] == "1"
        print(f"== with_bootstrap={with_boot}  with_rotations={with_rot} ==",
              flush=True)
        post, pre = construct(with_bootstrap=with_boot, with_rotations=with_rot)
        print(f"RESULT pre={pre} post={post} delta={post - pre}",
              flush=True)
        return

    # Driver: run three subprocess variants and report deltas.
    print("=" * 70)
    print(f"GPU baseline before any phantom import: {gpu_used_mib()} MiB")
    print("=" * 70)

    variants = [
        ("Engine + relin (no bootstrap, no user rotations)", "0", "0"),
        ("+ Bootstrap key", "1", "0"),
        ("+ Bootstrap + 69 user-rotation Galois keys", "1", "1"),
    ]
    deltas = {}
    for label, b, r in variants:
        print(f"\n--- {label} ---")
        env = os.environ.copy()
        out = subprocess.run(
            [sys.executable, __file__, b, r],
            env=env, capture_output=True, text=True, check=True,
        )
        print(out.stdout)
        for line in out.stdout.splitlines():
            if line.startswith("RESULT"):
                kv = dict(p.split("=") for p in line[7:].split())
                deltas[label] = (int(kv["pre"]), int(kv["post"]),
                                  int(kv["delta"]))
                break

    print("\n" + "=" * 70)
    print("SUMMARY (MiB)")
    print("=" * 70)
    base = deltas[variants[0][0]][1]
    boot = deltas[variants[1][0]][1] - base
    rot  = deltas[variants[2][0]][1] - deltas[variants[1][0]][1]
    print(f"  CUDA + libcuPhantom (pre-engine):   ~{deltas[variants[0][0]][0]:>6} MiB")
    print(f"  Engine + relin + workspace:         {base - deltas[variants[0][0]][0]:>+7} MiB")
    print(f"  Bootstrap key:                      {boot:>+7} MiB")
    print(f"  User-rotation Galois keys (69x):    {rot:>+7} MiB")
    print(f"  ---------------------------------- ")
    print(f"  Steady state (eng + boot + rot):    {deltas[variants[2][0]][1]:>7} MiB")


if __name__ == "__main__":
    main()
