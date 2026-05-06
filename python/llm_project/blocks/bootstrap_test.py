"""Bootstrap correctness test for `boot_centered`.

Exercises the cases that previously broke the LLM pipeline:

  * `[-0.4, 0.4]` zero-mean   — bootstrap's native domain (no scaling)
  * `[0.6, 1.4]`  mean = 1    — non-zero mean, small radius (centering only)
  * `[9.6, 10.4]` mean = 10   — large non-zero mean (centering only)
  * `[99.6, 100.4]` mean=100  — even larger mean (centering only)
  * `[-256, 256]` zero-mean   — large per-slot magnitude (scaling required)
  * `[-156, 356]` mean = 100  — large mean *and* large radius (centering + scaling)
  * `[-100, 100]` zero-mean   — moderate magnitude (scaling required)

Each case asserts that:
  - the slot-wise reconstruction error stays under a per-case tolerance,
  - the recovered slot mean matches the input mean to bootstrap precision.

Run with `python3 python/llm_project/blocks/bootstrap_test.py`.
"""

import math
import random
import sys
import time

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
import pyPhantom as phantom

sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
from blocks.bootstrap import boot_centered, TARGET_MAG


LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2
USER_SCALE = 2.0 ** 40
HW = 128
NUM_SCALE_LEVELS = 14
NUM_SPECIAL_PRIMES = 6


def _setup_engine():
    cfg = phantom.ckks_engine_config()
    cfg.log_n = LOG_N
    cfg.user_scale = USER_SCALE
    cfg.num_scale_levels = NUM_SCALE_LEVELS
    cfg.sparse_hw = HW
    cfg.num_special_primes = NUM_SPECIAL_PRIMES
    cfg.include_user_rotations = False
    cfg.user_rotation_steps = []
    e = phantom.ckks_engine(cfg)
    return e, e.context(), e.encoder(), e.secret_key()


def _encrypt_at_fresh(engine, ctx, enc, sk, msg):
    fresh_ci = engine.user_level_chain_index(0)
    return sk.encrypt_symmetric(
        ctx, enc.encode_double_vector(ctx, msg, USER_SCALE, fresh_ci))


def _summarize(label, msg, dec, dt_ms):
    n = NUM_SLOTS
    in_mean = sum(msg) / n
    out_mean = sum(dec[:n]) / n
    errs = [abs(dec[i] - msg[i]) for i in range(n)]
    max_err = max(errs)
    avg_err = sum(errs) / n
    radius = max(abs(v) for v in msg) or 1.0
    print(
        f"  {label:30s} in_mean={in_mean:+10.4f}  out_mean={out_mean:+10.4f}  "
        f"max|err|={max_err:.3e}  avg|err|={avg_err:.3e}  "
        f"max_rel={max_err/radius:.3e}  ({dt_ms:.1f} ms)"
    )
    return max_err, avg_err, in_mean, out_mean


def _check(label, max_err, in_mean, out_mean, *, abs_tol, mean_tol):
    fail = False
    if max_err > abs_tol:
        print(f"    FAIL [{label}]: max|err|={max_err:.3e} > tol={abs_tol:.3e}")
        fail = True
    if abs(out_mean - in_mean) > mean_tol:
        print(
            f"    FAIL [{label}]: |out_mean - in_mean|={abs(out_mean - in_mean):.3e} "
            f"> tol={mean_tol:.3e}"
        )
        fail = True
    return fail


def _run(label, msg, engine, ctx, enc, sk, *, abs_tol, mean_tol):
    ct = _encrypt_at_fresh(engine, ctx, enc, sk, msg)
    t0 = time.perf_counter()
    out = boot_centered(engine, ctx, enc, sk, ct)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    dec = enc.decode_double_vector(ctx, sk.decrypt(ctx, out))
    max_err, _, in_mean, out_mean = _summarize(label, msg, dec, dt_ms)
    return _check(label, max_err, in_mean, out_mean,
                  abs_tol=abs_tol, mean_tol=mean_tol)


def main():
    engine, ctx, enc, sk = _setup_engine()
    random.seed(0xB007B007)
    print(
        f"boot_centered test: logN={LOG_N} num_slots={NUM_SLOTS} "
        f"user_scale=2^{int(math.log2(USER_SCALE))} TARGET_MAG={TARGET_MAG}"
    )
    print(f"  max_user_level={engine.max_user_level()}")

    failures = 0

    # ---- Bootstrap's native domain. ----
    failures += _run(
        "[-0.4, 0.4] zero-mean",
        [random.uniform(-0.4, 0.4) for _ in range(NUM_SLOTS)],
        engine, ctx, enc, sk,
        abs_tol=1e-3, mean_tol=1e-3,
    )

    # ---- Non-zero mean, small radius (mean centering only). ----
    failures += _run(
        "mean=1.0  ± 0.4",
        [1.0 + random.uniform(-0.4, 0.4) for _ in range(NUM_SLOTS)],
        engine, ctx, enc, sk,
        abs_tol=1e-3, mean_tol=1e-3,
    )
    failures += _run(
        "mean=10.0 ± 0.4",
        [10.0 + random.uniform(-0.4, 0.4) for _ in range(NUM_SLOTS)],
        engine, ctx, enc, sk,
        abs_tol=1e-3, mean_tol=1e-3,
    )
    failures += _run(
        "mean=100  ± 0.4",
        [100.0 + random.uniform(-0.4, 0.4) for _ in range(NUM_SLOTS)],
        engine, ctx, enc, sk,
        abs_tol=1e-3, mean_tol=1e-3,
    )

    # ---- Large per-slot magnitude (scaling kicks in). ----
    # Bootstrap absolute noise on its native [-TARGET_MAG, TARGET_MAG] domain
    # is ~8.5e-5 (the K=28 R=3 polynomial floor — measured flat across input
    # magnitudes from 1e-5 to TARGET_MAG; reducing HW from 192 to 64 doesn't
    # move it). The post-bootstrap scale-up by `max_centered / TARGET_MAG`
    # multiplies that floor linearly with the signal magnitude, so the best
    # achievable absolute error is (8.5e-5 / TARGET_MAG) * max_centered ≈
    # 1.74e-4 * max_centered. Tolerance is set to 4e-4 * max_centered (~2.3x
    # the expected floor) so per-slot noise variance doesn't flake the test.
    failures += _run(
        "[-100, 100] zero-mean",
        [random.uniform(-100, 100) for _ in range(NUM_SLOTS)],
        engine, ctx, enc, sk,
        abs_tol=100 * 4e-4, mean_tol=1.0,
    )
    failures += _run(
        "[-256, 256] zero-mean",
        [random.uniform(-256, 256) for _ in range(NUM_SLOTS)],
        engine, ctx, enc, sk,
        abs_tol=256 * 4e-4, mean_tol=1.0,
    )

    # ---- Large mean AND large radius (both centering + scaling). ----
    failures += _run(
        "mean=100 ± 256",
        [100.0 + random.uniform(-256, 256) for _ in range(NUM_SLOTS)],
        engine, ctx, enc, sk,
        abs_tol=356 * 4e-4, mean_tol=1.0,
    )

    # ---- Edge: scaling at max_user_level should raise, not silently break. ----
    edge_msg = [random.uniform(-256, 256) for _ in range(NUM_SLOTS)]
    deep_ci = engine.user_level_chain_index(engine.max_user_level())
    deep_pt = enc.encode_double_vector(ctx, edge_msg, USER_SCALE, deep_ci)
    deep_ct = sk.encrypt_symmetric(ctx, deep_pt)
    raised = False
    try:
        boot_centered(engine, ctx, enc, sk, deep_ct)
    except ValueError as exc:
        raised = True
        print(f"  edge: scaling at max_user_level correctly raised ValueError")
    if not raised:
        print("    FAIL [edge]: expected ValueError at max_user_level + scaling")
        failures += 1

    if failures:
        raise SystemExit(f"\nbootstrap_test: {failures} case(s) FAILED")
    print("\nbootstrap_test: PASS")


if __name__ == "__main__":
    main()
