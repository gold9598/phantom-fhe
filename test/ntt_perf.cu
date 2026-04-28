// Microbenchmark for nwt_2d_radix8_forward_inplace / _backward_inplace.
// Times a stream of identical kernel launches with CUDA events.
#include "phantom.h"
#include <chrono>
#include <cstdio>
#include <vector>

using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

struct Result {
    size_t log_dim;
    size_t batch_size;
    double fwd_us;
    double bwd_us;
};

static Result bench_one(size_t log_dim, size_t batch_size, int iters, int warmup) {
    cuda_stream_wrapper stream_wrapper;
    const auto &s = stream_wrapper.get_stream();

    const size_t dim = 1ULL << log_dim;
    const auto h_modulus = CoeffModulus::Create(dim, std::vector<int>(batch_size, 50));
    auto modulus = make_cuda_auto_ptr<DModulus>(batch_size, s);
    for (size_t i = 0; i < batch_size; i++)
        modulus.get()[i].set(h_modulus[i].value(),
                             h_modulus[i].const_ratio()[0],
                             h_modulus[i].const_ratio()[1]);

    DNTTTable d_ntt_tables;
    d_ntt_tables.init(dim, batch_size, s);
    for (size_t i = 0; i < batch_size; i++) {
        auto h_ntt_table = NTT(log_dim, h_modulus[i]);
        d_ntt_tables.set(&modulus.get()[i],
                         h_ntt_table.get_from_root_powers().data(),
                         h_ntt_table.get_from_root_powers_shoup().data(),
                         h_ntt_table.get_from_inv_root_powers().data(),
                         h_ntt_table.get_from_inv_root_powers_shoup().data(),
                         h_ntt_table.inv_degree_modulo(),
                         h_ntt_table.inv_degree_modulo_shoup(),
                         i, s);
    }

    auto d_data = make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    cudaMemsetAsync(d_data.get(), 1, batch_size * dim * sizeof(uint64_t), s);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // ---------------- Forward ----------------
    for (int i = 0; i < warmup; i++)
        nwt_2d_radix8_forward_inplace(d_data.get(), d_ntt_tables, batch_size, 0, s);
    cudaStreamSynchronize(s);

    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; i++)
        nwt_2d_radix8_forward_inplace(d_data.get(), d_ntt_tables, batch_size, 0, s);
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float fwd_ms = 0.f;
    cudaEventElapsedTime(&fwd_ms, e0, e1);
    const double fwd_us = (fwd_ms * 1000.0) / iters;

    // ---------------- Backward ----------------
    for (int i = 0; i < warmup; i++)
        nwt_2d_radix8_backward_inplace(d_data.get(), d_ntt_tables, batch_size, 0, s);
    cudaStreamSynchronize(s);

    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; i++)
        nwt_2d_radix8_backward_inplace(d_data.get(), d_ntt_tables, batch_size, 0, s);
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float bwd_ms = 0.f;
    cudaEventElapsedTime(&bwd_ms, e0, e1);
    const double bwd_us = (bwd_ms * 1000.0) / iters;

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return {log_dim, batch_size, fwd_us, bwd_us};
}

int main() {
    constexpr int warmup = 50;
    constexpr int iters  = 500;

    const std::vector<size_t> log_dims    = {15, 16};                   // n=32K, 64K
    const std::vector<size_t> batches     = {1, 8, 10, 14, 18, 26, 34}; // matches CKKS HMult chains

    printf("# log_dim, batch, fwd_us, bwd_us\n");
    for (auto bs : batches) {
        for (auto ld : log_dims) {
            auto r = bench_one(ld, bs, iters, warmup);
            printf("%2zu, %4zu, %10.3f, %10.3f\n", r.log_dim, r.batch_size, r.fwd_us, r.bwd_us);
            fflush(stdout);
        }
    }
    return 0;
}
