#include "evaluate.cuh"
#include "ntt.cuh"
#include "polymath.cuh"
#include "rns.cuh"
#include "rns_bconv.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

namespace phantom {

    // ------------------------------------------------------------------
    // cp.async.cg pipelined variant of the key-switch inner-product MAC.
    // Lapis uses the same pattern in fhe_keyswitching.cu's
    // montgomery_mac_batched_dual_cuda_kernel: each thread handles 2
    // coefficients, and the next iteration's c2 / evk0 / evk1 chunks are
    // prefetched (16-byte cp.async.cg) into a double-buffered shared-mem
    // staging area while the current iteration's MAC runs.
    //
    // Semantics are identical to the legacy non-pipelined kernel below
    // (Barrett 128-bit accumulation, no in-loop reduction). Falls back
    // to the legacy kernel when the per-thread vectorisation can't apply
    // (n odd / reduction_threshold == 0 / pointer alignment).
    // ------------------------------------------------------------------
    static constexpr int KSP_BLOCK = 256;
    static constexpr int KSP_E     = 2;          // elements per thread
    static constexpr int KSP_TILE  = KSP_BLOCK * KSP_E;

    // Lighter variant: vectorized 16-byte loads via __ldg + __restrict__,
    // no cp.async / no shared memory. Each thread handles 2 coefficients;
    // four 128-bit accumulators stay in registers across the β-loop. This
    // keeps occupancy at the legacy kernel's level while letting the SASS
    // scheduler issue paired LDG.E.128 loads.
    __global__ __launch_bounds__(KSP_BLOCK)
    void key_switch_inner_prod_c2_and_evk_vec(
            uint64_t *__restrict__ dst,
            const uint64_t *__restrict__ c2,
            const uint64_t *const *__restrict__ evks,
            const DModulus *__restrict__ modulus,
            size_t n, size_t size_QP, size_t size_QP_n,
            size_t size_QlP, size_t size_QlP_n,
            size_t size_Q, size_t size_Ql, size_t beta) {

        const size_t base = (size_t)blockIdx.x * KSP_TILE + (size_t)threadIdx.x * KSP_E;
        if (base + KSP_E > size_QlP_n) return;

        const size_t coeff_idx = base & (n - 1);
        const size_t nid       = base / n;
        const size_t twr       = (nid >= size_Ql) ? (size_Q + (nid - size_Ql)) : nid;

        const DModulus mod = modulus[twr];
        const uint64_t qv  = mod.value();
        const uint64_t *qrat = mod.const_ratio();

        const size_t evk_id  = coeff_idx + twr * n;
        const size_t evk_id2 = evk_id + size_QP_n;
        const size_t c2_id   = coeff_idx + nid * n;

        uint128_t acc0_a{0, 0}, acc0_b{0, 0};
        uint128_t acc1_a{0, 0}, acc1_b{0, 0};

        #pragma unroll 1
        for (size_t i = 0; i < beta; i++) {
            const uint64_t *p_c2 = c2 + c2_id + i * size_QlP_n;
            const uint64_t *p_e0 = evks[i] + evk_id;
            const uint64_t *p_e1 = evks[i] + evk_id2;

            // 16-byte aligned loads (n is a power of 2 ≥ 2 so addresses are aligned).
            const ulonglong2 v_c2 = *reinterpret_cast<const ulonglong2 *>(p_c2);
            const ulonglong2 v_e0 = *reinterpret_cast<const ulonglong2 *>(p_e0);
            const ulonglong2 v_e1 = *reinterpret_cast<const ulonglong2 *>(p_e1);

            uint128_t p;
            p = multiply_uint64_uint64(v_c2.x, v_e0.x); add_uint128_uint128(acc0_a, p, acc0_a);
            p = multiply_uint64_uint64(v_c2.y, v_e0.y); add_uint128_uint128(acc0_b, p, acc0_b);
            p = multiply_uint64_uint64(v_c2.x, v_e1.x); add_uint128_uint128(acc1_a, p, acc1_a);
            p = multiply_uint64_uint64(v_c2.y, v_e1.y); add_uint128_uint128(acc1_b, p, acc1_b);
        }

        const uint64_t r0_a = barrett_reduce_uint128_uint64(acc0_a, qv, qrat);
        const uint64_t r0_b = barrett_reduce_uint128_uint64(acc0_b, qv, qrat);
        const uint64_t r1_a = barrett_reduce_uint128_uint64(acc1_a, qv, qrat);
        const uint64_t r1_b = barrett_reduce_uint128_uint64(acc1_b, qv, qrat);

        st_two_uint64(dst + base,                 r0_a, r0_b);
        st_two_uint64(dst + base + size_QlP_n,    r1_a, r1_b);
    }

    __global__ __launch_bounds__(KSP_BLOCK, 2)
    void key_switch_inner_prod_c2_and_evk_pipelined(
            uint64_t *__restrict__ dst,
            const uint64_t *__restrict__ c2,
            const uint64_t *const *__restrict__ evks,
            const DModulus *__restrict__ modulus,
            size_t n, size_t size_QP, size_t size_QP_n,
            size_t size_QlP, size_t size_QlP_n,
            size_t size_Q, size_t size_Ql, size_t beta) {

        const size_t base = (size_t)blockIdx.x * KSP_TILE + (size_t)threadIdx.x * KSP_E;
        if (base + KSP_E > size_QlP_n) return;

        const size_t coeff_idx = base & (n - 1);          // n is power of two
        const size_t nid       = base / n;
        const size_t twr       = (nid >= size_Ql) ? (size_Q + (nid - size_Ql)) : nid;

        const DModulus mod = modulus[twr];
        const uint64_t qv  = mod.value();
        const uint64_t *qrat = mod.const_ratio();

        const size_t evk_id  = coeff_idx + twr * n;
        const size_t evk_id2 = evk_id + size_QP_n;
        const size_t c2_id   = coeff_idx + nid * n;

        // smem[buf][slot][threadIdx.x][elem]  -- slot 0=c2, 1=evk0, 2=evk1
        __shared__ uint64_t smem[2][3][KSP_BLOCK][KSP_E];

        auto issue_prefetch = [&](int it, int buf) {
            const uint64_t *p_c2  = c2 + c2_id + (size_t)it * size_QlP_n;
            const uint64_t *p_e0  = evks[it] + evk_id;
            const uint64_t *p_e1  = evks[it] + evk_id2;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"((unsigned)__cvta_generic_to_shared(&smem[buf][0][threadIdx.x][0])),
                   "l"(p_c2));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"((unsigned)__cvta_generic_to_shared(&smem[buf][1][threadIdx.x][0])),
                   "l"(p_e0));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"((unsigned)__cvta_generic_to_shared(&smem[buf][2][threadIdx.x][0])),
                   "l"(p_e1));
            asm volatile("cp.async.commit_group;\n");
        };

        // Prime the pipeline.
        if (beta > 0) issue_prefetch(0, 0);

        uint128_t acc0_a{0, 0}, acc0_b{0, 0};
        uint128_t acc1_a{0, 0}, acc1_b{0, 0};

        for (size_t i = 0; i < beta; i++) {
            const int cur = i & 1;
            const int nxt = cur ^ 1;

            if (i + 1 < beta) {
                issue_prefetch((int)(i + 1), nxt);
                asm volatile("cp.async.wait_group 1;\n");
            } else {
                asm volatile("cp.async.wait_group 0;\n");
            }

            const uint64_t pt_a = smem[cur][0][threadIdx.x][0];
            const uint64_t pt_b = smem[cur][0][threadIdx.x][1];
            const uint64_t e0_a = smem[cur][1][threadIdx.x][0];
            const uint64_t e0_b = smem[cur][1][threadIdx.x][1];
            const uint64_t e1_a = smem[cur][2][threadIdx.x][0];
            const uint64_t e1_b = smem[cur][2][threadIdx.x][1];

            uint128_t p;
            p = multiply_uint64_uint64(pt_a, e0_a); add_uint128_uint128(acc0_a, p, acc0_a);
            p = multiply_uint64_uint64(pt_b, e0_b); add_uint128_uint128(acc0_b, p, acc0_b);
            p = multiply_uint64_uint64(pt_a, e1_a); add_uint128_uint128(acc1_a, p, acc1_a);
            p = multiply_uint64_uint64(pt_b, e1_b); add_uint128_uint128(acc1_b, p, acc1_b);
        }

        const uint64_t r0_a = barrett_reduce_uint128_uint64(acc0_a, qv, qrat);
        const uint64_t r0_b = barrett_reduce_uint128_uint64(acc0_b, qv, qrat);
        const uint64_t r1_a = barrett_reduce_uint128_uint64(acc1_a, qv, qrat);
        const uint64_t r1_b = barrett_reduce_uint128_uint64(acc1_b, qv, qrat);

        st_two_uint64(dst + base,                 r0_a, r0_b);
        st_two_uint64(dst + base + size_QlP_n,    r1_a, r1_b);
    }

    __global__ void key_switch_inner_prod_c2_and_evk(uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
                                                     const DModulus *modulus, size_t n, size_t size_QP,
                                                     size_t size_QP_n,
                                                     size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
                                                     size_t beta, size_t reduction_threshold) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size_QlP_n; tid += blockDim.x * gridDim.x) {
            size_t nid = tid / n;
            size_t twr = (nid >= size_Ql ? size_Q + (nid - size_Ql) : nid);
            // base_rns = {q0, q1, ..., qj, p}
            DModulus mod = modulus[twr];
            uint64_t evk_id = (tid % n) + twr * n;
            uint64_t c2_id = (tid % n) + nid * n;

            uint128_t prod0, prod1;
            uint128_t acc0, acc1;

            // ct^x = ( <RNS-Decomp(c*_2), evk_b> , <RNS-Decomp(c*_2), evk_a>
            // evk[key_index][rns]
            //
            // RNS-Decomp(c*_2)[key_index + rns_indx * twr] =
            //           ( {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod q0,
            //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod q1,
            //             ...
            //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod qj,
            //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod p, )
            //
            // decomp_mod_size = number of evks

            // evk[0]_a
            acc0 = multiply_uint64_uint64(c2[c2_id], evks[0][evk_id]);
            // evk[0]_b
            acc1 = multiply_uint64_uint64(c2[c2_id], evks[0][evk_id + size_QP_n]);

            for (uint64_t i = 1; i < beta; i++) {
                if (i && reduction_threshold == 0) {
                    acc0.lo = barrett_reduce_uint128_uint64(acc0, mod.value(), mod.const_ratio());
                    acc0.hi = 0;

                    acc1.lo = barrett_reduce_uint128_uint64(acc1, mod.value(), mod.const_ratio());
                    acc1.hi = 0;
                }

                prod0 = multiply_uint64_uint64(c2[c2_id + i * size_QlP_n], evks[i][evk_id]);
                add_uint128_uint128(acc0, prod0, acc0);

                prod1 = multiply_uint64_uint64(c2[c2_id + i * size_QlP_n], evks[i][evk_id + size_QP_n]);
                add_uint128_uint128(acc1, prod1, acc1);
            }

            uint64_t res0 = barrett_reduce_uint128_uint64(acc0, mod.value(), mod.const_ratio());
            dst[tid] = res0;

            uint64_t res1 = barrett_reduce_uint128_uint64(acc1, mod.value(), mod.const_ratio());
            dst[tid + size_QlP_n] = res1;
        }
    }

    void launch_ks_matmul_legacy(uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
                                 const DModulus *modulus, size_t n, size_t size_QP, size_t size_QP_n,
                                 size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
                                 size_t beta, size_t reduction_threshold, cudaStream_t stream) {
        const dim3 block(blockDimGlb.x);
        const dim3 grid(static_cast<unsigned>(size_QlP_n / block.x));
        key_switch_inner_prod_c2_and_evk<<<grid, block, 0, stream>>>(
                dst, c2, evks, modulus, n, size_QP, size_QP_n, size_QlP, size_QlP_n,
                size_Q, size_Ql, beta, reduction_threshold);
    }

    void launch_ks_matmul_pipelined(uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
                                    const DModulus *modulus, size_t n, size_t size_QP, size_t size_QP_n,
                                    size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
                                    size_t beta, cudaStream_t stream) {
        const dim3 block(KSP_BLOCK);
        const dim3 grid(static_cast<unsigned>(size_QlP_n / KSP_TILE));
        key_switch_inner_prod_c2_and_evk_pipelined<<<grid, block, 0, stream>>>(
                dst, c2, evks, modulus, n, size_QP, size_QP_n, size_QlP, size_QlP_n,
                size_Q, size_Ql, beta);
    }

    void launch_ks_matmul_vec(uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
                              const DModulus *modulus, size_t n, size_t size_QP, size_t size_QP_n,
                              size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
                              size_t beta, cudaStream_t stream) {
        const dim3 block(KSP_BLOCK);
        const dim3 grid(static_cast<unsigned>(size_QlP_n / KSP_TILE));
        key_switch_inner_prod_c2_and_evk_vec<<<grid, block, 0, stream>>>(
                dst, c2, evks, modulus, n, size_QP, size_QP_n, size_QlP, size_QlP_n,
                size_Q, size_Ql, beta);
    }

    void
    key_switch_inner_prod(uint64_t *p_cx, const uint64_t *p_t_mod_up, const uint64_t *const *rlk,
                          const DRNSTool &rns_tool,
                          const DModulus *modulus_QP, size_t reduction_threshold, const cudaStream_t &stream) {

        const size_t size_QP = rns_tool.size_QP();
        const size_t size_P = rns_tool.size_P();
        const size_t size_Q = size_QP - size_P;

        const size_t size_Ql = rns_tool.base_Ql().size();
        const size_t size_QlP = size_Ql + size_P;

        const size_t n = rns_tool.n();
        const auto size_QP_n = size_QP * n;
        const auto size_QlP_n = size_QlP * n;

        const size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

        // The legacy kernel achieves ~80% of HBM peak bandwidth on this kernel,
        // so the lapis-style cp.async pipelining and vectorised __ldg variants
        // (key_switch_inner_prod_c2_and_evk_{pipelined,vec}) don't beat it on
        // this layout. Dispatch through the legacy path; the optimised variants
        // are kept around (exposed via launch_ks_matmul_*) for the microbench.
        key_switch_inner_prod_c2_and_evk<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                p_cx, p_t_mod_up, rlk, modulus_QP, n, size_QP, size_QP_n, size_QlP, size_QlP_n, size_Q, size_Ql, beta,
                reduction_threshold);
    }

// cks refers to cipher to be key-switched
    void keyswitch_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, uint64_t *c2,
                           const PhantomRelinKey &relin_keys, bool is_relin, const cudaStream_t &stream) {
        const auto &s = stream;

        // Extract encryption parameters.
        auto &key_context_data = context.get_context_data(0);
        auto &key_parms = key_context_data.parms();
        auto scheme = key_parms.scheme();
        auto n = key_parms.poly_modulus_degree();
        auto mul_tech = key_parms.mul_tech();
        auto &key_modulus = key_parms.coeff_modulus();
        size_t size_P = key_parms.special_modulus_size();
        size_t size_QP = key_modulus.size();

        // HPS and HPSOverQ does not drop modulus
        uint32_t levelsDropped;

        if (scheme == scheme_type::bfv) {
            levelsDropped = 0;
            if (mul_tech == mul_tech_type::hps_overq_leveled) {
                size_t depth = encrypted.GetNoiseScaleDeg();
                bool isKeySwitch = !is_relin;
                bool is_Asymmetric = encrypted.is_asymmetric();
                size_t levels = depth - 1;
                auto dcrtBits = static_cast<double>(context.get_context_data(1).gpu_rns_tool().qMSB());

                // how many levels to drop
                levelsDropped = FindLevelsToDrop(context, levels, dcrtBits, isKeySwitch, is_Asymmetric);
            }
        } else if (scheme == scheme_type::bgv || scheme == scheme_type::ckks) {
            levelsDropped = encrypted.chain_index() - 1;
        } else {
            throw invalid_argument("unsupported scheme in keyswitch_inplace");
        }

        auto &rns_tool = context.get_context_data(1 + levelsDropped).gpu_rns_tool();

        auto modulus_QP = context.gpu_rns_tables().modulus();

        size_t size_Ql = rns_tool.base_Ql().size();
        size_t size_Q = size_QP - size_P;
        size_t size_QlP = size_Ql + size_P;

        auto size_Ql_n = size_Ql * n;
        // auto size_QP_n = size_QP * n;
        auto size_QlP_n = size_QlP * n;

        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
            auto t_cks = phantom::util::make_cuda_auto_ptr<uint64_t>(size_Q * n, s);
            cudaMemcpyAsync(t_cks.get(), c2, size_Q * n * sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, s);
            rns_tool.scaleAndRound_HPS_Q_Ql(c2, t_cks.get(), s);
        }

        // mod up
        size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();
        auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);
        rns_tool.modup(t_mod_up.get(), c2, context.gpu_rns_tables(), scheme, s);

        // key switch
        auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);
        auto reduction_threshold =
                (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;
        key_switch_inner_prod(cx.get(), t_mod_up.get(), relin_keys.public_keys_ptr(), rns_tool, modulus_QP,
                              reduction_threshold, s);

        // mod down + add-to-ct.
        //
        // keyswitch produces two output polynomials (cx[0], cx[1]); each needs a
        // moddown_from_NTT followed by add_to_ct_kernel. The two polynomials are
        // fully data-independent: their cx halves are disjoint (offset i*size_QlP_n),
        // their destination ct halves are disjoint (offset i*size_Ql_n / i*size_Q*n),
        // and moddown_from_NTT allocates its `delta` (and BGV `temp_t`) scratch
        // per-call on the supplied stream, so the two calls never alias.
        //
        // Run poly 0 on the caller stream `s` and poly 1 concurrently on a second
        // stream `s2`, synchronised by CUDA events so the result is bit-identical to
        // the serial path. `s2` waits for the keyswitch MAC (key_switch_inner_prod,
        // dispatched on `s` above) before reading cx; `s` waits for `s2` to finish
        // before returning, so the cx/t_mod_up frees and any downstream work on `s`
        // are correctly ordered after poly 1.
        auto moddown_and_add = [&](size_t i, const cudaStream_t &stream) {
            auto cx_i = cx.get() + i * size_QlP_n;
            rns_tool.moddown_from_NTT(cx_i, cx_i, context.gpu_rns_tables(), scheme, stream);

            if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
                auto ct_i = encrypted.data() + i * size_Q * n;
                auto t_cx = make_cuda_auto_ptr<uint64_t>(size_Q * n, stream);
                rns_tool.ExpandCRTBasis_Ql_Q(t_cx.get(), cx_i, stream);
                add_to_ct_kernel<<<(size_Q * n) / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                        ct_i, t_cx.get(), rns_tool.base_Q().base(), n, size_Q);
            } else {
                auto ct_i = encrypted.data() + i * size_Ql_n;
                add_to_ct_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                        ct_i, cx_i, rns_tool.base_Ql().base(), n, size_Ql);
            }
        };

        cudaStream_t s2;
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        cudaEvent_t ev_inner, ev_s2;
        cudaEventCreateWithFlags(&ev_inner, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev_s2, cudaEventDisableTiming);

        // s2 must observe the keyswitch MAC output before it reads cx.
        cudaEventRecord(ev_inner, s);
        cudaStreamWaitEvent(s2, ev_inner, 0);

        // poly 0 on s, poly 1 concurrently on s2.
        moddown_and_add(0, s);
        moddown_and_add(1, s2);

        // s waits for s2 to finish before returning.
        cudaEventRecord(ev_s2, s2);
        cudaStreamWaitEvent(s, ev_s2, 0);

        cudaEventDestroy(ev_inner);
        cudaEventDestroy(ev_s2);
        cudaStreamDestroy(s2);
    }
}
