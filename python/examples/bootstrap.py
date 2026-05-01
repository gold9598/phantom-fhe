"""
CKKS bootstrap smoke test (Phase 6 Python binding).

Mirrors the C++ `run_bootstrap_round_trip` test from test/bootstrap_test.cu:
build the lapis 4-section heterogeneous chain, generate a sparse-secret bootstrap
key, encrypt a random real message at user_scale=2^40, deplete the ciphertext
to one level above the bottom prime, run `bootstrap`, and verify the decoded
output matches the input within 1e-3.
"""

import math
import random

import pyPhantom as phantom

LOG_N = 16
N = 1 << LOG_N
NUM_SLOTS = N // 2
USER_SCALE = 2.0 ** 40
HW = 128
EVAL_MOD_LEVELS = 9
TOL = 1e-3

# Lapis 4-section heterogeneous chain (bottom -> top, special last):
#   bits = [msg(58) | scale(40)x4 | S2C(58)x3 | ER(58)x9 | C2S(29)x3 | special(58)x4]
bits = [58]                # msg q_0 (bottom)
bits += [40] * 4           # scale segment
bits += [58] * 3           # S2C
bits += [58] * 9           # EvalMod / EvalRound
bits += [29] * 3           # C2S
bits += [58] * 4           # special

params = phantom.params(phantom.scheme_type.ckks)
params.set_poly_modulus_degree(N)
params.set_special_modulus_size(4)
params.set_coeff_modulus(phantom.create_coeff_modulus(N, bits))

# Compute the exact union of Galois elements needed by C2S + S2C (stages={5,5,5})
# plus the conjugation element. This matches the C++ bootstrap_test setup.
galois_elts = phantom.bootstrap_required_galois_elts(LOG_N, [5, 5, 5])
params.set_galois_elts(galois_elts)

context = phantom.context(params)

sk = phantom.secret_key()
sk.generate_sparse(context, HW)

encoder = phantom.ckks_encoder(context)
slot_count = encoder.slot_count()
assert slot_count == NUM_SLOTS

bk = phantom.create_bootstrap_key(
    context, encoder, sk,
    sparse_hamming_weight=HW,
    eval_mod_levels=EVAL_MOD_LEVELS,
    user_scale=USER_SCALE,
)

# Encode a random real message in [-0.4, 0.4] at user_scale.
random.seed(0xB007B007)
msg = [random.uniform(-0.4, 0.4) for _ in range(slot_count)]

pt = encoder.encode_double_vector(context, msg, USER_SCALE, chain_index=1)
ct = sk.encrypt_symmetric(context, pt)

# Deplete to one level above the bottom (so scale_up_for_bootstrap can do
# multiply + rescale). bottom = total_parm_size() - 1.
bottom_index = context.total_parm_size() - 1
pre_boot_index = bottom_index - 1
phantom.mod_switch_to_inplace(context, ct, pre_boot_index)

print(f"bootstrap (python): logN={LOG_N}, user_scale=2^40, hw={HW}")
print(f"  ct before bootstrap: chain_index={ct.chain_index()}, "
      f"scale=2^{math.log2(ct.scale()):.1f}")

out = phantom.bootstrap(context, encoder, ct, bk, USER_SCALE)

print(f"  ct after  bootstrap: chain_index={out.chain_index()}, "
      f"scale=2^{math.log2(out.scale()):.1f}")

dec_pt = sk.decrypt(context, out)
decoded = encoder.decode_double_vector(context, dec_pt)

errors = [abs(decoded[i] - msg[i]) for i in range(slot_count)]
max_err = max(errors)
avg_err = sum(errors) / slot_count

for i in range(4):
    print(f"  slot[{i}] dec={decoded[i]:.6e}  in={msg[i]:.6e}  err={errors[i]:.3e}")
print(f"  avg |err| = {avg_err:.3e}")
print(f"  max |err| = {max_err:.3e}")

if max_err > TOL:
    raise SystemExit(f"FAIL: bootstrap max abs error {max_err:.3e} > {TOL:.3e}")

print("PASS")
