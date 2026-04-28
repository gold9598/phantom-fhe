// Microbenchmark for homomorphic multiply (mult+relin) on BFV / CKKS / BGV.
// Times event-recorded GPU runs for a fixed parameter set.
#include "phantom.h"

#include <chrono>
#include <cstdio>
#include <vector>
#include <cuComplex.h>

using namespace phantom;
using namespace phantom::arith;

static double bench_kernel(cudaStream_t s, std::function<void()> op,
                           int warmup, int iters) {
    for (int i = 0; i < warmup; i++) op();
    cudaStreamSynchronize(s);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; i++) op();
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return (ms * 1000.0) / iters; // µs per call
}

static void bench_bfv(size_t n, const std::vector<int> &mod_bits, int alpha,
                      mul_tech_type tech, const char *tech_name) {
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(n);
    parms.set_coeff_modulus(CoeffModulus::Create(n, mod_bits));
    parms.set_mul_tech(tech);
    parms.set_special_modulus_size(alpha);
    parms.set_plain_modulus(PlainModulus::Batching(n, 20));

    PhantomContext context(parms);
    PhantomSecretKey sk(context);
    PhantomBatchEncoder enc(context);
    PhantomPlaintext pt;
    std::vector<uint64_t> data(enc.slot_count(), 7);
    enc.encode(context, data, pt);

    PhantomCiphertext a, b;
    sk.encrypt_symmetric(context, pt, a);
    sk.encrypt_symmetric(context, pt, b);

    PhantomRelinKey rk = sk.gen_relinkey(context);

    auto stream = cudaStreamPerThread;
    PhantomCiphertext scratch(a);
    auto op = [&]() {
        scratch = a;
        multiply_and_relin_inplace(context, scratch, b, rk);
    };
    const double us = bench_kernel(stream, op, /*warmup=*/10, /*iters=*/200);
    printf("BFV/%s,n=%zu,L=%zu  %10.2f us\n",
           tech_name, n, mod_bits.size(), us);
}

static void bench_ckks(size_t n, const std::vector<int> &mod_bits, int alpha) {
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(n);
    parms.set_coeff_modulus(CoeffModulus::Create(n, mod_bits));
    parms.set_special_modulus_size(alpha);

    PhantomContext context(parms);
    PhantomSecretKey sk(context);
    PhantomCKKSEncoder enc(context);
    PhantomRelinKey rk = sk.gen_relinkey(context);

    std::vector<cuDoubleComplex> data(enc.slot_count(), make_cuDoubleComplex(0.5, 0.25));
    PhantomPlaintext pt;
    enc.encode(context, data, std::pow(2.0, 40), pt);

    PhantomCiphertext a, b;
    sk.encrypt_symmetric(context, pt, a);
    sk.encrypt_symmetric(context, pt, b);

    auto stream = cudaStreamPerThread;
    PhantomCiphertext scratch(a);
    auto op = [&]() {
        scratch = a;
        multiply_and_relin_inplace(context, scratch, b, rk);
    };
    const double us = bench_kernel(stream, op, /*warmup=*/10, /*iters=*/200);
    printf("CKKS,n=%zu,L=%zu          %10.2f us\n", n, mod_bits.size(), us);
}

static void bench_bgv(size_t n, const std::vector<int> &mod_bits, int alpha) {
    EncryptionParameters parms(scheme_type::bgv);
    parms.set_poly_modulus_degree(n);
    parms.set_coeff_modulus(CoeffModulus::Create(n, mod_bits));
    parms.set_special_modulus_size(alpha);
    parms.set_plain_modulus(PlainModulus::Batching(n, 20));

    PhantomContext context(parms);
    PhantomSecretKey sk(context);
    PhantomBatchEncoder enc(context);
    PhantomRelinKey rk = sk.gen_relinkey(context);

    std::vector<uint64_t> data(enc.slot_count(), 3);
    PhantomPlaintext pt;
    enc.encode(context, data, pt);

    PhantomCiphertext a, b;
    sk.encrypt_symmetric(context, pt, a);
    sk.encrypt_symmetric(context, pt, b);

    auto stream = cudaStreamPerThread;
    PhantomCiphertext scratch(a);
    auto op = [&]() {
        scratch = a;
        multiply_and_relin_inplace(context, scratch, b, rk);
    };
    const double us = bench_kernel(stream, op, /*warmup=*/10, /*iters=*/200);
    printf("BGV,n=%zu,L=%zu           %10.2f us\n", n, mod_bits.size(), us);
}

int main() {
    printf("# scheme,params  median µs/HMult\n");

    // CKKS logN=16 (n=65536), several depths and α settings.
    // Pattern: 60-bit first + 50-bit work primes + 60-bit special prime(s).
    // L counts the work primes (modswitch budget); total primes = L + 2 (or L+1+α).
    auto chain = [](int first_bits, int work_bits, int n_work, int special_bits, int alpha) {
        std::vector<int> v;
        v.push_back(first_bits);
        for (int i = 0; i < n_work; i++) v.push_back(work_bits);
        for (int i = 0; i < alpha; i++) v.push_back(special_bits);
        return v;
    };

    // logN=16
    bench_ckks(1 << 16, chain(60, 50, 8,  60, 1), 1);   // L=8,  α=1
    bench_ckks(1 << 16, chain(60, 50, 16, 60, 1), 1);   // L=16, α=1
    bench_ckks(1 << 16, chain(60, 50, 24, 60, 1), 1);   // L=24, α=1
    bench_ckks(1 << 16, chain(60, 50, 32, 60, 1), 1);   // L=32, α=1

    // logN=15 / 14 keeps prior coverage for context.
    bench_ckks(1 << 15, std::vector<int>{60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}, 1);
    bench_ckks(1 << 14, std::vector<int>{60, 40, 40, 40, 40, 60}, 1);

    return 0;
}
