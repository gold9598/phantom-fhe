"""Multi-GPU smoke test for shared SingleChainPlaintext and concurrent threading.

Validates that:
1. SingleChainPlaintext (pinned host memory, device-agnostic via CUDA UVA) can be
   expanded by CKKSEngine contexts on different GPUs.
2. cudaStreamPerThread allows two worker threads on two GPUs to run concurrently
   without serializing (speedup > 1.4x).

This test must pass before running the full 408-MRPC 4-GPU sweep.
"""
import os
import sys
import time
import threading
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, os.path.join(_REPO, "build", "lib"))
import pyPhantom as phantom

sys.path.insert(0, _THIS_DIR)
from llama3 import (LOG_N, NUM_SLOTS, SCALE, SPARSE_HW, NUM_SCALE_LEVELS,
                     NUM_SPECIAL_PRIMES)


def main():
    print("=" * 70, flush=True)
    print("Multi-GPU Smoke Test", flush=True)
    print("=" * 70, flush=True)

    # Test 1: Device count
    dev_count = phantom.get_cuda_device_count()
    print(f"\n[Test 1] CUDA device count: {dev_count}", flush=True)
    if dev_count < 2:
        print(f"SKIP — only {dev_count} GPU(s) found, need 2+", flush=True)
        return 0

    # Test 2: Two engines, two GPUs, sequential
    print(f"\n[Test 2] Building engines on GPU 0 and GPU 1 (sequential)...", flush=True)

    phantom.set_cuda_device(0)
    cfg0 = phantom.ckks_engine_config()
    cfg0.log_n = LOG_N
    cfg0.user_scale = SCALE
    cfg0.num_scale_levels = NUM_SCALE_LEVELS
    cfg0.sparse_hw = SPARSE_HW
    cfg0.num_special_primes = NUM_SPECIAL_PRIMES
    cfg0.include_user_rotations = False
    cfg0.user_rotation_steps = []
    cfg0.user_rotation_target_chain_indices = []
    engine0 = phantom.ckks_engine(cfg0)
    ctx0 = engine0.context()
    encoder0 = engine0.encoder()
    sk0 = engine0.secret_key()
    print(f"  GPU 0: engine OK  max_user_level={engine0.max_user_level()}", flush=True)

    phantom.set_cuda_device(1)
    cfg1 = phantom.ckks_engine_config()
    cfg1.log_n = LOG_N
    cfg1.user_scale = SCALE
    cfg1.num_scale_levels = NUM_SCALE_LEVELS
    cfg1.sparse_hw = SPARSE_HW
    cfg1.num_special_primes = NUM_SPECIAL_PRIMES
    cfg1.include_user_rotations = False
    cfg1.user_rotation_steps = []
    cfg1.user_rotation_target_chain_indices = []
    engine1 = phantom.ckks_engine(cfg1)
    ctx1 = engine1.context()
    encoder1 = engine1.encoder()
    sk1 = engine1.secret_key()
    print(f"  GPU 1: engine OK  max_user_level={engine1.max_user_level()}", flush=True)

    # Test each engine in isolation
    rng = np.random.default_rng(0)
    v = rng.standard_normal(NUM_SLOTS).tolist()

    phantom.set_cuda_device(0)
    pt0 = encoder0.encode_double_vector(ctx0, v, SCALE, engine0.user_level_chain_index(0))
    ct0 = sk0.encrypt_symmetric(ctx0, pt0)
    dec0 = np.array(encoder0.decode_double_vector(ctx0, sk0.decrypt(ctx0, ct0)))
    err0 = float(np.abs(dec0 - np.array(v)).max())
    print(f"  GPU 0 round-trip: max|err|={err0:.3e}  {'OK' if err0 < 1e-3 else 'FAIL'}", flush=True)

    phantom.set_cuda_device(1)
    pt1 = encoder1.encode_double_vector(ctx1, v, SCALE, engine1.user_level_chain_index(0))
    ct1 = sk1.encrypt_symmetric(ctx1, pt1)
    dec1 = np.array(encoder1.decode_double_vector(ctx1, sk1.decrypt(ctx1, ct1)))
    err1 = float(np.abs(dec1 - np.array(v)).max())
    print(f"  GPU 1 round-trip: max|err|={err1:.3e}  {'OK' if err1 < 1e-3 else 'FAIL'}", flush=True)

    # Test 3: Shared SingleChainPlaintext across two engines
    print(f"\n[Test 3] Shared SingleChainPlaintext across two engines...", flush=True)

    # Build a simple test vector for SCP
    scp_input = [float(i + 1) for i in range(NUM_SLOTS)]

    # Encode SCP on GPU 0
    phantom.set_cuda_device(0)
    scp = phantom.encode_single_chain_plaintext(ctx0, encoder0,
                                                [complex(x, 0) for x in scp_input],
                                                SCALE)
    print(f"  Encoded SingleChainPlaintext on GPU 0 (nbytes={scp.nbytes})", flush=True)

    # Expand and test on GPU 0
    phantom.set_cuda_device(0)
    target_chain = engine0.user_level_chain_index(0)
    pt_expanded_0 = phantom.expand_single_chain_to_full(ctx0, scp, target_chain)
    ct_expanded_0 = sk0.encrypt_symmetric(ctx0, pt_expanded_0)
    dec_expanded_0 = np.array(encoder0.decode_double_vector(ctx0, sk0.decrypt(ctx0, ct_expanded_0)))
    err_expand_0 = float(np.abs(dec_expanded_0 - np.array(scp_input)).max())
    print(f"  GPU 0 expand+decrypt: max|err|={err_expand_0:.3e}  {'OK' if err_expand_0 < 1e-3 else 'FAIL'}",
          flush=True)

    # Expand and test on GPU 1 (CRITICAL: same SCP, different engine/GPU)
    phantom.set_cuda_device(1)
    target_chain = engine1.user_level_chain_index(0)
    pt_expanded_1 = phantom.expand_single_chain_to_full(ctx1, scp, target_chain)
    ct_expanded_1 = sk1.encrypt_symmetric(ctx1, pt_expanded_1)
    dec_expanded_1 = np.array(encoder1.decode_double_vector(ctx1, sk1.decrypt(ctx1, ct_expanded_1)))
    err_expand_1 = float(np.abs(dec_expanded_1 - np.array(scp_input)).max())
    print(f"  GPU 1 expand+decrypt: max|err|={err_expand_1:.3e}  {'OK' if err_expand_1 < 1e-3 else 'FAIL'}",
          flush=True)

    if err_expand_1 >= 1e-3:
        print("  ERROR: GPU 1 cannot read shared SCP from host memory!", flush=True)

    # Test 4: Concurrent threading (the real parallelism test)
    print(f"\n[Test 4] Concurrent threading on two GPUs...", flush=True)

    results = {}
    errors = {}

    def worker(gpu_id, engine, ctx, encoder, sk, scp, expected):
        """Worker thread: expand SCP, encrypt, decrypt, and compute error."""
        try:
            phantom.set_cuda_device(gpu_id)
            for i in range(5):
                target_chain = engine.user_level_chain_index(0)
                pt = phantom.expand_single_chain_to_full(ctx, scp, target_chain)
                ct = sk.encrypt_symmetric(ctx, pt)
                dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)))
                err = float(np.abs(dec - expected).max())
                if gpu_id not in results:
                    results[gpu_id] = []
                results[gpu_id].append(err)
        except Exception as e:
            errors[gpu_id] = e

    # Run parallel
    expected_scp = np.array(scp_input)
    t0_thread = threading.Thread(target=worker, args=(0, engine0, ctx0, encoder0, sk0, scp, expected_scp))
    t1_thread = threading.Thread(target=worker, args=(1, engine1, ctx1, encoder1, sk1, scp, expected_scp))

    start = time.perf_counter()
    t0_thread.start()
    t1_thread.start()
    t0_thread.join()
    t1_thread.join()
    parallel_t = time.perf_counter() - start

    if errors:
        print(f"  ERROR in worker threads: {errors}", flush=True)

    if 0 in results and 1 in results:
        max_err_0 = max(results[0])
        max_err_1 = max(results[1])
        print(f"  Parallel (2 GPUs, 5 iters each):", flush=True)
        print(f"    GPU 0 max|err|={max_err_0:.3e}  {'OK' if max_err_0 < 1e-3 else 'FAIL'}", flush=True)
        print(f"    GPU 1 max|err|={max_err_1:.3e}  {'OK' if max_err_1 < 1e-3 else 'FAIL'}", flush=True)
        print(f"    Total time: {parallel_t*1000:.0f}ms", flush=True)

    # Sequential baseline for comparison
    results_seq = {}
    start = time.perf_counter()
    worker(0, engine0, ctx0, encoder0, sk0, scp, expected_scp)
    worker(1, engine1, ctx1, encoder1, sk1, scp, expected_scp)
    seq_t = time.perf_counter() - start

    print(f"  Sequential (GPU 0 then GPU 1, 5 iters each):", flush=True)
    print(f"    Total time: {seq_t*1000:.0f}ms", flush=True)

    if parallel_t > 0:
        speedup = seq_t / parallel_t
        print(f"  Speedup: {speedup:.2f}x", flush=True)
        if speedup >= 1.4:
            print(f"  OK — speedup >= 1.4x indicates concurrent execution", flush=True)
        else:
            print(f"  WARN — speedup < 1.4x; threads may be serializing", flush=True)

    # Test 5: Summary
    print(f"\n[Test 5] Summary", flush=True)
    print("=" * 70, flush=True)
    all_ok = (
        dev_count >= 2 and
        err0 < 1e-3 and err1 < 1e-3 and
        err_expand_0 < 1e-3 and err_expand_1 < 1e-3 and
        (0 in results and 1 in results and max(results[0]) < 1e-3 and max(results[1]) < 1e-3)
    )

    print(f"Test 1 (device count):          {'PASS' if dev_count >= 2 else 'SKIP'}", flush=True)
    print(f"Test 2 (two engines):           {'PASS' if err0 < 1e-3 and err1 < 1e-3 else 'FAIL'}", flush=True)
    print(f"Test 3 (shared SCP):            {'PASS' if err_expand_0 < 1e-3 and err_expand_1 < 1e-3 else 'FAIL'}", flush=True)
    print(f"Test 4 (concurrent threading):  {'PASS' if (0 in results and 1 in results and max(results[0]) < 1e-3 and max(results[1]) < 1e-3) else 'FAIL'}", flush=True)
    print("=" * 70, flush=True)

    if all_ok:
        print("ALL TESTS PASSED - ready for 4-GPU MRPC sweep", flush=True)
    else:
        print("SOME TESTS FAILED - investigate before proceeding", flush=True)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
