"""Minimal Phantom sanity test. Runs basic CKKS ops and reports if any
returns silently-zeroed output. If this fails on A6000, the Phantom
build (or arch) is the root cause — not our pipeline code."""
import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, os.path.join(_REPO, "build", "lib"))
import pyPhantom as phantom

sys.path.insert(0, _THIS_DIR)
from llama3 import (LOG_N, NUM_SLOTS, SCALE, SPARSE_HW, NUM_SCALE_LEVELS,
                     NUM_SPECIAL_PRIMES)


def main():
    cfg = phantom.ckks_engine_config()
    cfg.log_n = LOG_N; cfg.user_scale = SCALE
    cfg.num_scale_levels = NUM_SCALE_LEVELS
    cfg.sparse_hw = SPARSE_HW; cfg.num_special_primes = NUM_SPECIAL_PRIMES
    cfg.include_user_rotations = False
    cfg.user_rotation_steps = []; cfg.user_rotation_target_chain_indices = []
    eng = phantom.ckks_engine(cfg)
    ctx = eng.context(); encoder = eng.encoder(); sk = eng.secret_key()
    gk = eng.galois_key()
    fresh_ci = eng.user_level_chain_index(0)
    print(f"engine OK  max_user_level={eng.max_user_level()}", flush=True)

    rng = np.random.default_rng(0)
    v = rng.standard_normal(NUM_SLOTS).tolist()

    # 1a) Encode + decode (no encryption): tests encoder NTT roundtrip alone.
    pt = encoder.encode_double_vector(ctx, v, SCALE, fresh_ci)
    dec_pt = np.array(encoder.decode_double_vector(ctx, pt))
    err = float(np.abs(dec_pt - np.array(v)).max())
    print(f"[1a] encode/decode (no encrypt)  max|err|={err:.3e}  norm={np.linalg.norm(dec_pt):.4f}  "
          f"{'OK' if err < 1e-3 else 'FAIL'}", flush=True)

    # 1b) Encrypt + decrypt (no compute): tests sk roundtrip alone.
    ct = sk.encrypt_symmetric(ctx, pt)
    dec_ct = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct)))
    err = float(np.abs(dec_ct - np.array(v)).max())
    print(f"[1b] encrypt/decrypt + decode    max|err|={err:.3e}  norm={np.linalg.norm(dec_ct):.4f}  "
          f"{'OK' if err < 1e-3 else 'FAIL'}", flush=True)

    # 2) Multiply by plaintext constant 2
    pt2 = encoder.encode_double_vector(ctx, [2.0]*NUM_SLOTS, SCALE, ct.chain_index())
    ct2 = phantom.multiply_plain(ctx, ct, pt2)
    ct2 = phantom.rescale_to_next(ctx, ct2)
    ct2.set_scale(SCALE)
    dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct2)))
    err = float(np.abs(dec - 2.0 * np.array(v)).max())
    print(f"[2] multiply_plain by 2        max|err|={err:.3e}  {'OK' if err < 1e-3 else 'FAIL'}",
          flush=True)

    # 3) Conjugation (rotate step=0)
    ct_conj = phantom.rotate(ctx, ct, 0, gk)
    dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct_conj)))
    # For real-only ct, conjugation should preserve values (imag was 0).
    err = float(np.abs(dec - np.array(v)).max())
    print(f"[3] conjugation (rotate 0)     max|err|={err:.3e}  {'OK' if err < 1e-3 else 'FAIL'}",
          flush=True)

    # 4) Bootstrap on a deep ct (simulate post-compute chain)
    ct_deep = phantom.mod_switch_to(ctx, ct, eng.user_level_chain_index(10))
    ct_deep.set_scale(SCALE)
    from blocks.bootstrap import bootstrap_safe
    ct_boot = bootstrap_safe(eng, ctx, encoder, ct_deep,
                              max_abs=4.0, slot_count=NUM_SLOTS)
    dec = np.array(encoder.decode_double_vector(ctx, sk.decrypt(ctx, ct_boot)))
    err = float(np.abs(dec - np.array(v)).max())
    print(f"[4] bootstrap_safe (max_abs=4) max|err|={err:.3e}  {'OK' if err < 1e-2 else 'FAIL'}",
          flush=True)

    # 5) Output norm — sanity that ct values are nonzero
    norm = float(np.linalg.norm(dec))
    print(f"[5] post-bootstrap output norm = {norm:.4f}  "
          f"(should be ~{np.linalg.norm(v):.4f})  "
          f"{'OK' if norm > 100 else 'FAIL — ZERO OUTPUT'}", flush=True)


if __name__ == "__main__":
    main()
