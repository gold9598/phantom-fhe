// phantom::CKKSEngine integration tests.
//
// Three cases:
//   1. engine_round_trip:           encode -> encrypt -> decrypt round-trip.
//   2. engine_mul_and_rescale:      ciphertext-ciphertext multiply + rescale.
//   3. engine_deplete_then_bootstrap:
//        deplete to max_user_level via plaintext multiplies + rescales, then
//        bootstrap, verify user_level returns to 0 and the message survives.
//
// NOTE: A single CKKSEngine is constructed in main() and shared across all
// three test functions. The Galois KSK bundle for logN=16 is ~20 GB; building
// three separate engines would exhaust the CUDA async memory pool even on a
// 32 GB card because cudaFreeAsync() returns memory lazily.

#include "ckks_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace phantom;
using C64 = std::complex<double>;

static int run_engine_round_trip(CKKSEngine &engine) {
    const std::size_t slot_count = engine.slot_count();
    std::mt19937_64 rng(0xCAFEUL);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    std::vector<C64> input(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i) {
        input[i] = C64(dist(rng), dist(rng));
    }

    auto pt = engine.encode(input);
    auto ct = engine.encrypt(pt);

    if (engine.user_level(ct) != 0) {
        std::fprintf(stderr, "FAIL engine_round_trip: fresh ct user_level=%d (expected 0)\n",
                     engine.user_level(ct));
        return 1;
    }

    auto decoded = engine.decrypt_decode(ct);

    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < slot_count; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs(decoded[i] - input[i]));
    }
    std::printf("engine_round_trip: slots=%zu, user_level=%d, max |err| = %.3e\n",
                slot_count, engine.user_level(ct), max_abs_err);
    if (max_abs_err > 1e-7) {
        std::fprintf(stderr, "FAIL engine_round_trip: max abs error %.3e > 1e-7\n", max_abs_err);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

static int run_engine_mul_and_rescale(CKKSEngine &engine) {
    const std::size_t slot_count = engine.slot_count();
    std::mt19937_64 rng(0xBABEUL);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    std::vector<C64> a(slot_count), b(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i) {
        a[i] = C64(dist(rng), 0.0);
        b[i] = C64(dist(rng), 0.0);
    }

    auto ct_a = engine.encrypt(engine.encode(a, 0));
    auto ct_b = engine.encrypt(engine.encode(b, 0));

    engine.mul_and_relin_inplace(ct_a, ct_b);
    engine.rescale_inplace(ct_a);

    if (engine.user_level(ct_a) != 1) {
        std::fprintf(stderr, "FAIL engine_mul_and_rescale: post mul+rescale user_level=%d (expected 1)\n",
                     engine.user_level(ct_a));
        return 1;
    }

    auto decoded = engine.decrypt_decode(ct_a);

    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < slot_count; ++i) {
        const C64 expected = a[i] * b[i];
        max_abs_err = std::max(max_abs_err, std::abs(decoded[i] - expected));
    }
    std::printf("engine_mul_and_rescale: user_level=%d, max |err| = %.3e\n",
                engine.user_level(ct_a), max_abs_err);
    if (max_abs_err > 1e-5) {
        std::fprintf(stderr, "FAIL engine_mul_and_rescale: max abs error %.3e > 1e-5\n", max_abs_err);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

static int run_engine_deplete_then_bootstrap(CKKSEngine &engine) {
    const std::size_t slot_count = engine.slot_count();
    std::mt19937_64 rng(0xB00B007UL);
    // CKKS native message bound: |m| < 0.5. Larger application ranges require
    // user-level preprocessing (divide by APP_SCALE before encrypt, multiply
    // back after decrypt) — that scaling propagates the bootstrap error too.
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    std::vector<C64> input(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i) {
        input[i] = C64(dist(rng), 0.0);
    }

    auto ct = engine.encrypt(engine.encode(input, 0));

    // Deplete to max_user_level via repeated rescale (mul_plain by 1 then
    // rescale). Each iteration: encode ones at current level k, mul_plain, rescale.
    std::vector<C64> ones(slot_count, C64(1.0, 0.0));
    for (int k = 0; k < engine.max_user_level(); ++k) {
        auto pt_one = engine.encode(ones, k);
        engine.mul_plain_inplace(ct, pt_one);
        engine.rescale_inplace(ct);
    }

    const int depleted_level = engine.user_level(ct);
    if (depleted_level != engine.max_user_level()) {
        std::fprintf(stderr, "FAIL engine_deplete_then_bootstrap: post-deplete user_level=%d (expected %d)\n",
                     depleted_level, engine.max_user_level());
        return 1;
    }
    std::printf("engine_deplete_then_bootstrap: depleted to user_level=%d, calling bootstrap...\n",
                depleted_level);
    std::fflush(stdout);

    // Bench: run bootstrap_inplace 5 times (re-deplete each iter) and time.
    std::vector<double> bs_times_ms;
    for (int run = 0; run < 5; ++run) {
        auto pt2 = engine.encode(input, 0);
        auto ct2 = engine.encrypt(pt2);
        for (int k = 0; k < engine.max_user_level(); ++k) {
            auto pt_one = engine.encode(ones, k);
            engine.mul_plain_inplace(ct2, pt_one);
            engine.rescale_inplace(ct2);
        }
        cudaDeviceSynchronize();
        auto ts = std::chrono::high_resolution_clock::now();
        engine.bootstrap_inplace(ct2);
        cudaDeviceSynchronize();
        auto te = std::chrono::high_resolution_clock::now();
        bs_times_ms.push_back(std::chrono::duration<double, std::milli>(te - ts).count());
    }
    double sum = 0, mn = 1e18, mx = 0;
    for (double t : bs_times_ms) { sum += t; mn = std::min(mn, t); mx = std::max(mx, t); }
    std::printf("[BENCH] bootstrap latency (5 runs): min=%.1f ms, max=%.1f ms, avg=%.1f ms (per-slot %.2f us)\n",
                mn, mx, sum / 5.0, (sum / 5.0) * 1000.0 / engine.slot_count());

    engine.bootstrap_inplace(ct);

    if (engine.user_level(ct) != 0) {
        std::fprintf(stderr, "FAIL engine_deplete_then_bootstrap: post-bootstrap user_level=%d (expected 0)\n",
                     engine.user_level(ct));
        return 1;
    }

    auto decoded = engine.decrypt_decode(ct);

    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < slot_count; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs(decoded[i].real() - input[i].real()));
    }
    for (std::size_t i = 0; i < 4; ++i) {
        std::printf("  slot[%zu] dec=%.6e  in=%.6e  err=%.3e\n",
                    i, decoded[i].real(), input[i].real(),
                    std::abs(decoded[i].real() - input[i].real()));
    }
    std::printf("  post-bootstrap user_level=%d, max |err| = %.3e\n",
                engine.user_level(ct), max_abs_err);

    constexpr double tol = 1e-3;
    if (max_abs_err > tol) {
        std::fprintf(stderr, "FAIL engine_deplete_then_bootstrap: max abs error %.3e > %.3e\n",
                     max_abs_err, tol);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

int main() {
    std::printf("Constructing CKKSEngine (logN=16, scale=2^40, levels=4, hw=128)...\n");
    std::fflush(stdout);
    CKKSEngineConfig cfg{};
    CKKSEngine engine(cfg);
    std::printf("Engine ready. slot_count=%zu  max_user_level=%d\n\n",
                engine.slot_count(), engine.max_user_level());
    std::fflush(stdout);

    int rc = 0;
    rc |= run_engine_round_trip(engine);
    rc |= run_engine_mul_and_rescale(engine);
    rc |= run_engine_deplete_then_bootstrap(engine);
    return rc;
}
