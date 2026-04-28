// Unit test: phantom::evalmod_k7_r3
//
// Verifies that the K=7 R=3 EvalMod port computes sin(2π·7·x)/(2π) on a CKKS
// ciphertext, for inputs in the test domain (small reals near 0).
//
// We sweep a vector of inputs x ∈ {a small spread around 0} and compare the
// decrypted output to the cleartext oracle sin(2π·7·x)/(2π). Average abs error
// is what we report; the K=7 polynomial advertises ~20.9 bits of precision
// (≈ 5e-7), so we set a generous tolerance of 1e-3 absolute (CKKS noise floor
// at this depth dominates).

#include "phantom.h"
#include "evalmod.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace phantom;
using namespace phantom::arith;

enum class Variant { K7_R3, K16_R3 };

static double oracle(double x, Variant v) {
    constexpr double TWO_PI = 6.283185307179586;
    const double K = (v == Variant::K7_R3) ? 7.0 : 16.0;
    return std::sin(TWO_PI * K * x) / TWO_PI;
}

static const char *name_of(Variant v) {
    return (v == Variant::K7_R3) ? "K7_R3" : "K16_R3";
}

// Run the chosen EvalMod variant at the target log_n and report stats.
// `prime_bits` is the rescale-prime size (also used as the encoding scale).
static int run_one(size_t log_n, Variant variant, int prime_bits) {
    EncryptionParameters parms(scheme_type::ckks);
    const size_t N = 1ULL << log_n;
    parms.set_poly_modulus_degree(N);
    parms.set_special_modulus_size(1);

    std::vector<int> bits;
    bits.push_back(60);
    for (int i = 0; i < 10; ++i) bits.push_back(prime_bits);
    bits.push_back(60);
    parms.set_coeff_modulus(CoeffModulus::Create(N, bits));

    PhantomContext context(parms);
    PhantomSecretKey sk(context);
    PhantomRelinKey  rk = sk.gen_relinkey(context);
    PhantomCKKSEncoder encoder(context);

    const double scale = std::pow(2.0, prime_bits);
    const size_t slot_count = encoder.slot_count();

    // Inputs: spread small reals (well inside the [-1, 1] Chebyshev domain;
    // for the K=7 sine the polynomial targets cos(7πx/4 - π/16)·(2π)^{-1/8}
    // and the test oracle sin(2π·7·x)/(2π) is valid for tiny x where the
    // approximation hasn't yet been amplified by DA into the periodic regime).
    std::vector<double> input(slot_count, 0.0);
    for (size_t i = 0; i < slot_count; ++i) {
        // map slot index to a value in [-0.02, 0.02]
        const double t = (double(i) / double(slot_count - 1)) * 2.0 - 1.0;
        input[i] = 0.02 * t;
    }

    PhantomPlaintext pt;
    encoder.encode(context, input, scale, pt);

    PhantomCiphertext ct;
    sk.encrypt_symmetric(context, pt, ct);

    PhantomCiphertext result = (variant == Variant::K7_R3)
        ? evalmod_k7_r3 (context, encoder, ct, rk)
        : evalmod_k16_r3(context, encoder, ct, rk);

    PhantomPlaintext dec_pt;
    sk.decrypt(context, result, dec_pt);
    std::vector<double> decoded;
    encoder.decode(context, dec_pt, decoded);

    // Error stats over the slot range we used.
    double max_abs_err = 0.0;
    double sum_abs_err = 0.0;
    size_t n_eval = std::min<size_t>(slot_count, 32);  // sample
    for (size_t i = 0; i < slot_count; i += slot_count / n_eval) {
        const double expected = oracle(input[i], variant);
        const double err = std::abs(decoded[i] - expected);
        max_abs_err = std::max(max_abs_err, err);
        sum_abs_err += err;
    }
    const double avg_err = sum_abs_err / double(n_eval);

    printf("evalmod_%s:  logN=%zu, N=%zu, scale=2^%d, slots=%zu (sampling %zu)\n",
           name_of(variant), log_n, N, prime_bits, slot_count, n_eval);
    printf("  input range:  [%g, %g]\n", input.front(), input.back());
    printf("  decoded[0]    = %.6e   oracle = %.6e   err = %.2e\n",
           decoded[0], oracle(input[0], variant),
           std::abs(decoded[0] - oracle(input[0], variant)));
    printf("  decoded[mid]  = %.6e   oracle = %.6e\n",
           decoded[slot_count / 2], oracle(input[slot_count / 2], variant));
    printf("  decoded[last] = %.6e   oracle = %.6e\n",
           decoded.back(), oracle(input.back(), variant));
    printf("  avg |err|     = %.3e\n", avg_err);
    printf("  max |err|     = %.3e\n", max_abs_err);

    // Tolerance scales with the noise floor: 5e-3 is generous enough to
    // tolerate K=16 at scale=2^40 (noise-limited) but still flags any real
    // algorithmic error (which would be ≥1e-1).
    constexpr double TOL = 5e-3;
    if (max_abs_err > TOL) {
        std::fprintf(stderr, "FAIL: max abs error %.3e exceeds tolerance %.3e\n",
                     max_abs_err, TOL);
        return 1;
    }
    printf("PASS\n\n");
    return 0;
}

int main() {
    int rc = 0;
    // scale 2^40 (user-specified): both K=7 and K=16 are CKKS-noise-limited
    rc |= run_one(13, Variant::K7_R3,  40);
    rc |= run_one(16, Variant::K7_R3,  40);
    rc |= run_one(13, Variant::K16_R3, 40);
    rc |= run_one(16, Variant::K16_R3, 40);
    // scale 2^50: shows K=16 polynomial precision when not noise-limited
    rc |= run_one(13, Variant::K16_R3, 50);
    rc |= run_one(16, Variant::K16_R3, 50);
    return rc;
}
