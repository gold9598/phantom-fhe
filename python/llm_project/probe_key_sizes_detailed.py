"""Element-by-element GPU memory breakdown for the engine + bootstrap key
+ user-rotation key bundle. Each variant runs in its own subprocess so
allocator state doesn't carry over.

Variants:
  A. import phantom (no engine)               -> CUDA + libcuPhantom
  B. + engine ctor, BootstrapTo17, no users   -> + engine + relin (no bootstrap key)
  C. + engine ctor, standard, no users        -> + bootstrap key
  D. + engine, standard, 1 user step @ ch 17  -> + 1 KSK at chain 17
  E. + engine, standard, 1 user step @ ch 23  -> + 1 KSK at chain 23
  F. + engine, standard, 1 user step @ ch 26  -> + 1 KSK at chain 26
  G. + engine, standard, production-dist      -> + full owned-key bundle

Deltas isolate each component. The single-key variants (D/E/F) give the
per-chain KSK cost directly.
"""

import os
import subprocess
import sys
import time

LOG_N = 16
NUM_SLOTS = (1 << LOG_N) // 2
SCALE = 2.0 ** 40
SPARSE_HW = 128


def gpu_used_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().strip().splitlines()[0])


def construct(variant: str):
    sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
    pre_import = gpu_used_mib()
    import pyPhantom as phantom
    pre_engine = gpu_used_mib()
    print(f"  pre_import={pre_import} MiB, post_import={pre_engine} MiB", flush=True)

    if variant == "A":
        return pre_engine

    cfg = phantom.ckks_engine_config()
    cfg.log_n = LOG_N
    cfg.user_scale = SCALE
    cfg.sparse_hw = SPARSE_HW
    cfg.include_user_rotations = False

    if variant == "B":
        cfg.use_bootstrap_to_17_levels = True
        cfg.num_scale_levels = 18
        cfg.num_special_primes = 8
        cfg.user_rotation_steps = []
        cfg.user_rotation_target_chain_indices = []
    elif variant == "C":
        cfg.use_bootstrap_to_17_levels = False
        cfg.num_scale_levels = 14
        cfg.num_special_primes = 6
        cfg.user_rotation_steps = []
        cfg.user_rotation_target_chain_indices = []
    elif variant in ("D", "E", "F"):
        cfg.use_bootstrap_to_17_levels = False
        cfg.num_scale_levels = 14
        cfg.num_special_primes = 6
        chain = {"D": 17, "E": 23, "F": 26}[variant]
        cfg.user_rotation_steps = [1]
        cfg.user_rotation_target_chain_indices = [chain]
    elif variant == "G":
        cfg.use_bootstrap_to_17_levels = False
        cfg.num_scale_levels = 14
        cfg.num_special_primes = 6
        # Production-like: 19@chain17 + 7@chain23 + 43@chain26 = 69 distinct steps.
        steps, chains = [], []
        cursor = 1
        for n_at, target in [(19, 17), (7, 23), (43, 26)]:
            for _ in range(n_at):
                steps.append(cursor)
                chains.append(target)
                cursor += 1
        cfg.user_rotation_steps = steps
        cfg.user_rotation_target_chain_indices = chains

    t0 = time.perf_counter()
    engine = phantom.ckks_engine(cfg)
    elapsed = time.perf_counter() - t0
    post = gpu_used_mib()
    print(f"  engine ctor: {elapsed:.2f}s -> post={post} MiB "
          f"(delta over post_import: +{post - pre_engine} MiB)",
          flush=True)
    _ = engine.context()
    return post


def main():
    if len(sys.argv) > 1:
        variant = sys.argv[1]
        print(f"== variant {variant} ==", flush=True)
        post = construct(variant)
        print(f"RESULT variant={variant} post={post}", flush=True)
        return

    print("=" * 70)
    print("Element-by-element GPU memory breakdown")
    print("=" * 70)
    variants = [
        ("A", "import phantom only"),
        ("B", "+ engine ctor (BootstrapTo17, no user keys, no bootstrap key)"),
        ("C", "+ standard engine + bootstrap key, no user keys"),
        ("D", "+ 1 user-rotation KSK at chain 17"),
        ("E", "+ 1 user-rotation KSK at chain 23"),
        ("F", "+ 1 user-rotation KSK at chain 26"),
        ("G", "+ production user-rotation bundle (19@17 + 7@23 + 43@26)"),
    ]
    results = {}
    for v, label in variants:
        print(f"\n--- {v}: {label} ---")
        out = subprocess.run(
            [sys.executable, __file__, v],
            capture_output=True, text=True, check=False,
        )
        print(out.stdout)
        if out.returncode != 0:
            print(f"  stderr: {out.stderr}")
            continue
        for line in out.stdout.splitlines():
            if line.startswith("RESULT"):
                kv = dict(p.split("=") for p in line[7:].split())
                results[v] = int(kv["post"])

    print("\n" + "=" * 70)
    print("BREAKDOWN")
    print("=" * 70)
    print(f"  CUDA + libcuPhantom (variant A):                 {results.get('A', 0):>7} MiB")
    delta_B_A = results.get('B', 0) - results.get('A', 0)
    print(f"  Engine workspace + relin (B - A, BootstrapTo17): {delta_B_A:>+7} MiB")
    delta_C_B = results.get('C', 0) - results.get('B', 0)
    print(f"  Bootstrap key            (C - B, chain caveat):  {delta_C_B:>+7} MiB")
    delta_D_C = results.get('D', 0) - results.get('C', 0)
    delta_E_C = results.get('E', 0) - results.get('C', 0)
    delta_F_C = results.get('F', 0) - results.get('C', 0)
    print(f"  1 user KSK at chain 17:                          {delta_D_C:>+7} MiB")
    print(f"  1 user KSK at chain 23:                          {delta_E_C:>+7} MiB")
    print(f"  1 user KSK at chain 26:                          {delta_F_C:>+7} MiB")
    delta_G_C = results.get('G', 0) - results.get('C', 0)
    print(f"  Full production user bundle (G - C):             {delta_G_C:>+7} MiB")
    print(f"    (= 19 × {delta_D_C} + 7 × {delta_E_C} + 43 × {delta_F_C} ?")
    print(f"     = {19*delta_D_C + 7*delta_E_C + 43*delta_F_C} MiB if no dedup)")
    print(f"  Steady state (variant G):                        {results.get('G', 0):>7} MiB")


if __name__ == "__main__":
    main()
