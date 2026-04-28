// Microbench: time the key-switch inner-product matmul kernel in isolation.
// Compares the legacy kernel vs the new cp.async pipelined kernel for
// realistic CKKS logN=16 parameters.
#include "phantom.h"
#include "evaluate.cuh"

#include <cstdio>
#include <vector>

using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

static double bench(cudaStream_t s, std::function<void()> launch, int iters, int warmup) {
    for (int i = 0; i < warmup; i++) launch();
    cudaStreamSynchronize(s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; i++) launch();
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (ms * 1000.0) / iters;
}

// Mirrors the call shape used by key_switch_inner_prod() in eval_key_switch.cu:
//   - c2  is layout [beta][size_QlP][n]
//   - evks[i] is layout [2][size_QP][n]   (poly0 then poly1)
//   - dst is layout [2][size_QlP][n]
static void run_one(size_t log_n, size_t size_Ql, size_t size_P, size_t beta) {
    const size_t n = 1ULL << log_n;
    const size_t size_QP = size_Ql + size_P;
    const size_t size_Q  = size_QP - size_P;
    const size_t size_QlP = size_Ql + size_P;
    const size_t size_QP_n  = size_QP  * n;
    const size_t size_QlP_n = size_QlP * n;

    cuda_stream_wrapper sw;
    auto s = sw.get_stream();

    // Build moduli (50-bit work primes + 60-bit special).
    std::vector<int> bits(size_Ql, 50);
    for (size_t i = 0; i < size_P; i++) bits.push_back(60);
    auto h_mod = CoeffModulus::Create(n, bits);
    auto modulus = make_cuda_auto_ptr<DModulus>(size_QP, s);
    for (size_t i = 0; i < size_QP; i++)
        modulus.get()[i].set(h_mod[i].value(), h_mod[i].const_ratio()[0], h_mod[i].const_ratio()[1]);

    // Allocate input/output buffers.
    auto c2  = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);
    auto dst = make_cuda_auto_ptr<uint64_t>(2    * size_QlP_n, s);
    cudaMemsetAsync(c2.get(),  0x42, beta * size_QlP_n * sizeof(uint64_t), s);
    cudaMemsetAsync(dst.get(), 0,    2    * size_QlP_n * sizeof(uint64_t), s);

    // Allocate beta evk arrays (poly0 + poly1, size_QP primes each).
    std::vector<cuda_auto_ptr<uint64_t>> evks_owned;
    std::vector<const uint64_t *> evks_host(beta);
    for (size_t i = 0; i < beta; i++) {
        evks_owned.emplace_back(make_cuda_auto_ptr<uint64_t>(2 * size_QP_n, s));
        cudaMemsetAsync(evks_owned.back().get(), 0x55, 2 * size_QP_n * sizeof(uint64_t), s);
        evks_host[i] = evks_owned.back().get();
    }
    auto evks_dev = make_cuda_auto_ptr<const uint64_t *>(beta, s);
    cudaMemcpyAsync((void *)evks_dev.get(), evks_host.data(),
                    beta * sizeof(const uint64_t *), cudaMemcpyHostToDevice, s);

    const size_t reduction_threshold =
        (1ULL << (64 - static_cast<size_t>(std::log2(h_mod.front().value())) - 1)) - 1;

    auto launch_legacy = [&]() {
        launch_ks_matmul_legacy(dst.get(), c2.get(), evks_dev.get(), modulus.get(),
                                n, size_QP, size_QP_n, size_QlP, size_QlP_n,
                                size_Q, size_Ql, beta, reduction_threshold, s);
    };
    auto launch_pipe = [&]() {
        launch_ks_matmul_pipelined(dst.get(), c2.get(), evks_dev.get(), modulus.get(),
                                   n, size_QP, size_QP_n, size_QlP, size_QlP_n,
                                   size_Q, size_Ql, beta, s);
    };
    auto launch_vec = [&]() {
        launch_ks_matmul_vec(dst.get(), c2.get(), evks_dev.get(), modulus.get(),
                             n, size_QP, size_QP_n, size_QlP, size_QlP_n,
                             size_Q, size_Ql, beta, s);
    };

    constexpr int warmup = 20;
    constexpr int iters  = 200;
    const double leg = bench(s, launch_legacy, iters, warmup);
    const double pip = bench(s, launch_pipe,   iters, warmup);
    const double vec = bench(s, launch_vec,    iters, warmup);
    const double pct_p = (pip - leg) / leg * 100.0;
    const double pct_v = (vec - leg) / leg * 100.0;
    printf("logN=%2zu L=%2zu α=%zu β=%2zu  legacy=%8.2f us  pipe=%8.2f us (%+5.1f%%)  vec=%8.2f us (%+5.1f%%)\n",
           log_n, size_Ql, size_P, beta, leg, pip, pct_p, vec, pct_v);
}

int main() {
    // CKKS logN=16 sweeps matching hmult_perf cases. β = ⌈L/α⌉.
    run_one(16,  9, 1,  9);   // L=10 chain, β=9
    run_one(16, 17, 1, 17);   // L=18 chain
    run_one(16, 25, 1, 25);   // L=26 chain
    run_one(16, 33, 1, 33);   // L=34 chain

    run_one(15, 13, 1, 13);   // logN=15 reference
    run_one(14,  5, 1,  5);   // logN=14 reference
    return 0;
}
