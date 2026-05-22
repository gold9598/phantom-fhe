#include "bootstrap.h"

#include "ckks.h"
#include "evalmod.h"
#include "evaluate.cuh"
#include "ntt.cuh"
#include "uintmodmath.cuh"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

namespace phantom {

    // Modulus-extension kernel: each (tower, coefficient) pair re-encodes the
    // signed centered representative of the source coefficient (read from a
    // single source tower q_msg) into the destination tower q_j.
    //
    //  src[i]                  in [0, q_msg) for i = 0..N-1
    //  c_signed = src[i] >= q_msg/2 ? src[i] - q_msg : src[i]
    //  dst[(j*N) + i] = c_signed mod q_j  (in [0, q_j))
    //
    // Both src and dst are in coefficient (non-NTT) form; the caller is
    // responsible for INTT on the input and forward NTT on the output.
    __global__ void mod_raise_signed_tile_kernel(
            const uint64_t *__restrict__ src,
            uint64_t *__restrict__ dst,
            const DModulus *__restrict__ moduli, // length = num_towers
            uint64_t q_msg,
            size_t num_towers,
            size_t N) {
        const size_t total = num_towers * N;
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < total;
             tid += blockDim.x * gridDim.x) {
            const size_t j = tid / N;          // tower index in dst
            const size_t i = tid - j * N;      // coefficient index
            // Phantom NTT may leave values in [0, 2q) (lazy reduction);
            // bring them back to [0, q) before centering.
            uint64_t v_unsigned = src[i];
            if (v_unsigned >= q_msg) v_unsigned -= q_msg;
            const uint64_t half = q_msg >> 1;

            const DModulus mod = moduli[j];
            const uint64_t qj = mod.value();
            const uint64_t mu_hi = mod.const_ratio()[1];

            uint64_t out;
            if (v_unsigned >= half) {
                // Negative branch: value = v_unsigned - q_msg, in [-q_msg/2, 0)
                // We want this mod qj. Compute (q_msg - v_unsigned) mod qj first
                // (a positive uint), then negate mod qj.
                const uint64_t mag = q_msg - v_unsigned;          // in (0, q_msg/2]
                uint64_t mag_mod = barrett_reduce_uint64_uint64(mag, qj, mu_hi);
                out = (mag_mod == 0) ? 0ULL : (qj - mag_mod);
            } else {
                out = barrett_reduce_uint64_uint64(v_unsigned, qj, mu_hi);
            }
            dst[tid] = out;
        }
    }

    // Per-tower scalar multiply kernel: dst[i] = (dst[i] * scalar) mod q_t.
    // Used by Phase 4 scale_up_for_bootstrap and K·ct (eval_round). One launch
    // operates on a single tower's coefficients (poly_degree threads).
    __global__ void multiply_scalar_per_tower_kernel(
            uint64_t *__restrict__ dst,
            uint64_t scalar,
            const DModulus *__restrict__ tower_modulus,
            size_t poly_degree) {
        const DModulus mod = *tower_modulus;
        const uint64_t q = mod.value();
        const uint64_t mu_hi = mod.const_ratio()[1];
        const uint64_t *ratio = mod.const_ratio();
        // Pre-reduce the (constant) scalar into [0, q) once per thread group.
        uint64_t s = scalar;
        if (s >= q) s = barrett_reduce_uint64_uint64(s, q, mu_hi);
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < poly_degree;
             tid += blockDim.x * gridDim.x) {
            uint64_t v = dst[tid];
            if (v >= q) v -= q;  // NTT lazy reduction guard
            dst[tid] = multiply_and_barrett_reduce_uint64(v, s, q, ratio);
        }
    }

    // Run the modulus extension on a 2-element ciphertext currently at the
    // bottom of the chain. After this call the ciphertext occupies the top of
    // the chain (chain index = first index, all Q ordinary primes), in NTT
    // form. The caller's secret expectation does not change here — that's the
    // job of the surrounding KSKs.
    static void mod_raise_extend_modulus(const PhantomContext &ctx,
                                         PhantomCiphertext &ct,
                                         const cudaStream_t &stream) {
        if (ct.size() != 2) {
            throw std::invalid_argument("mod_raise_inplace: ciphertext size must be 2");
        }
        if (!ct.is_ntt_form()) {
            throw std::invalid_argument("mod_raise_inplace: ciphertext must be in NTT form");
        }

        const auto &bottom_data = ctx.get_context_data(ct.chain_index());
        const auto &bottom_parms = bottom_data.parms();
        const auto &bottom_modulus = bottom_parms.coeff_modulus();
        if (bottom_modulus.size() != 1) {
            throw std::invalid_argument(
                "mod_raise_inplace: caller must rescale to single-prime level first");
        }
        const uint64_t q_msg = bottom_modulus[0].value();

        const size_t top_index = ctx.get_first_index();
        const auto &top_data = ctx.get_context_data(top_index);
        const auto &top_parms = top_data.parms();
        const size_t top_modulus_size = top_parms.coeff_modulus().size();
        const size_t N = top_parms.poly_modulus_degree();

        // Step 1: pull (c0, c1) into a coefficient-domain scratch buffer at
        // the bottom level (size = 2 * 1 * N).
        auto bottom_buf = make_cuda_auto_ptr<uint64_t>(2 * N, stream);
        cudaMemcpyAsync(bottom_buf.get(), ct.data(), 2 * N * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);

        // After CKKS mod_switch_to bottom, the remaining tower is q_0 (the
        // first ordinary prime). Phantom's mod_switch_drop_to_next pops the
        // *last* prime each step, so the bottom retains the first one. The
        // global gpu_rns_tables modulus layout is [q_0, q_1, ..., q_{Q-1},
        // P_0], so start_modulus_idx = 0 for the INTT.
        const size_t bottom_start_idx = 0;
        nwt_2d_radix8_backward_inplace(bottom_buf.get(), ctx.gpu_rns_tables(),
                                       1, bottom_start_idx, stream);
        nwt_2d_radix8_backward_inplace(bottom_buf.get() + N, ctx.gpu_rns_tables(),
                                       1, bottom_start_idx, stream);

        // Step 2: allocate full-modulus output buffers and signed-tile.
        auto top_buf = make_cuda_auto_ptr<uint64_t>(2 * top_modulus_size * N, stream);
        const DModulus *moduli = ctx.gpu_rns_tables().modulus();

        // Note: gpu_rns_tables modulus layout is [q_0, q_1, ..., q_{Q-1}, P_0].
        // The first `top_modulus_size` entries are exactly the ordinary primes
        // we want to tile across.
        const size_t threads = 256;
        const size_t blocks = (top_modulus_size * N + threads - 1) / threads;
        for (size_t poly = 0; poly < 2; ++poly) {
            mod_raise_signed_tile_kernel<<<blocks, threads, 0, stream>>>(
                bottom_buf.get() + poly * N,
                top_buf.get() + poly * top_modulus_size * N,
                moduli,
                q_msg,
                top_modulus_size,
                N);
        }

        // Step 3: forward NTT each tower to bring the result back into NTT form.
        nwt_2d_radix8_forward_inplace(top_buf.get(), ctx.gpu_rns_tables(),
                                      top_modulus_size, 0, stream);
        nwt_2d_radix8_forward_inplace(top_buf.get() + top_modulus_size * N,
                                      ctx.gpu_rns_tables(), top_modulus_size, 0,
                                      stream);

        // Step 4: install the new buffer + metadata. Resize allocates fresh
        // storage at the top chain level.
        ct.resize(ctx, top_index, 2, stream);
        cudaMemcpyAsync(ct.data(), top_buf.get(),
                        2 * top_modulus_size * N * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
        ct.set_ntt_form(true);
        // chain_index is set by resize().
    }

    SmallBootstrapKey
    create_small_bootstrap_key(const PhantomContext &ctx,
                               const PhantomSecretKey &dense_sk,
                               std::size_t sparse_hamming_weight) {
        if (sparse_hamming_weight == 0) {
            throw std::invalid_argument(
                "create_small_bootstrap_key: hamming_weight must be > 0");
        }

        // Generate a temporary sparse secret. It goes out of scope at the end
        // of this function, taking its device buffer with it.
        PhantomSecretKey sparse_sk;
        sparse_sk.generate_sparse(ctx, sparse_hamming_weight);

        SmallBootstrapKey bk;
        // dense ciphertext -> sparse: KSK encapsulates dense_sk under sparse_sk.
        bk.ksk_to_sparse = sparse_sk.create_kswitch_key(ctx, dense_sk);
        // sparse ciphertext -> dense: KSK encapsulates sparse_sk under dense_sk.
        bk.ksk_to_dense = dense_sk.create_kswitch_key(ctx, sparse_sk);
        return bk;
    }

    void mod_raise_inplace(const PhantomContext &ctx,
                           PhantomCiphertext &ct,
                           const SmallBootstrapKey &bk) {
        // 1. Encapsulate: switch ct from dense to sparse secret.
        apply_kswitch_inplace(ctx, ct, bk.ksk_to_sparse);

        // 2. Modulus extension: bottom -> top of chain.
        mod_raise_extend_modulus(ctx, ct, cudaStreamPerThread);

        // 3. Decapsulate: switch ct from sparse back to dense secret.
        apply_kswitch_inplace(ctx, ct, bk.ksk_to_dense);
    }

    // ========================================================================
    // Phase 2: C2S linear transform — diagonals, pre-encoding, evaluator.
    // ========================================================================

    namespace {
        using C64 = std::complex<double>;

        // numpy-style cyclic shift: result[(i + s) mod n] = vals[i].
        std::vector<C64> roll(const std::vector<C64> &vals, long long shift) {
            const long long n = static_cast<long long>(vals.size());
            std::vector<C64> out(vals.size(), C64(0.0, 0.0));
            if (n == 0) return out;
            long long s = ((shift % n) + n) % n;
            for (long long i = 0; i < n; ++i) {
                out[(i + s) % n] = vals[i];
            }
            return out;
        }

        std::vector<C64> element_mul(const std::vector<C64> &a, const std::vector<C64> &b) {
            std::vector<C64> out(a.size());
            for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
            return out;
        }

        void element_add_assign(std::vector<C64> &dst, const std::vector<C64> &src) {
            for (size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
        }

        bool is_all_zero(const std::vector<C64> &v) {
            for (const auto &c : v) {
                if (std::abs(c) >= 1e-50) return false;
            }
            return true;
        }
    } // namespace

    LinearTransformDiagonals
    build_c2s_diagonals(int log_n, std::vector<int> stages_per_layer) {
        if (log_n < 2) {
            throw std::invalid_argument("build_c2s_diagonals: log_n must be >= 2");
        }
        const int log_num_slots = log_n - 1;
        const int num_slots = 1 << log_num_slots;
        const int num_layers = static_cast<int>(stages_per_layer.size());

        int sum_stages = 0;
        for (int s : stages_per_layer) sum_stages += s;
        if (sum_stages != log_num_slots) {
            throw std::invalid_argument(
                "build_c2s_diagonals: stages_per_layer must sum to log2(N/2)");
        }

        // --- Step 1: per-stage twiddle factors a/b/c ---
        const int m_val = 4 * num_slots;
        std::vector<C64> phi_v_inv(m_val);
        const double pi = std::acos(-1.0);
        for (int k = 0; k < m_val; ++k) {
            const double angle = -2.0 * pi * static_cast<double>(k) / static_cast<double>(m_val);
            phi_v_inv[k] = C64(std::cos(angle), std::sin(angle));
        }

        std::vector<std::vector<C64>> a_stages(log_num_slots, std::vector<C64>(num_slots, C64(0.0, 0.0)));
        std::vector<std::vector<C64>> b_stages(log_num_slots, std::vector<C64>(num_slots, C64(0.0, 0.0)));
        std::vector<std::vector<C64>> c_stages(log_num_slots, std::vector<C64>(num_slots, C64(0.0, 0.0)));

        {
            int m = num_slots;
            while (m >= 2) {
                int round = log_num_slots - static_cast<int>(std::log2(static_cast<double>(m)));
                int half = m / 2;
                int four_m = 4 * m;

                std::vector<C64> phik(half);
                {
                    long long pow5 = 1;
                    for (int j = 0; j < half; ++j) {
                        long long k_idx = pow5 * static_cast<long long>(num_slots) / m;
                        phik[j] = phi_v_inv[static_cast<size_t>(k_idx % m_val)];
                        pow5 = (pow5 * 5) % four_m;
                    }
                }

                for (int i = 0; i < num_slots; i += m) {
                    for (int j = 0; j < half; ++j) {
                        a_stages[round][i + j] = C64(1.0, 0.0);
                        a_stages[round][i + half + j] = -phik[j];
                        b_stages[round][i + j] = C64(1.0, 0.0);
                        c_stages[round][i + half + j] = phik[j];
                    }
                }

                m /= 2;
            }
        }

        // --- Step 2: collapse stages into layers ---
        LinearTransformDiagonals out;
        out.stages_per_layer = stages_per_layer;
        out.layers.resize(num_layers);
        out.n2 = 4;

        int global_stage = 0;
        for (int layer = 0; layer < num_layers; ++layer) {
            const int s = stages_per_layer[layer];
            const int n1 = 1 << (s - 1);
            int stages_after = 0;
            for (int l2 = layer + 1; l2 < num_layers; ++l2) stages_after += stages_per_layer[l2];

            out.layers[layer].n1 = n1;
            out.layers[layer].rotation_unit = 1 << stages_after;

            std::unordered_map<int, std::vector<C64>> prev;

            for (int j = 0; j < s; ++j) {
                const int r = global_stage + j;
                const int half = 1 << (s - 1 - j);
                const long long rot = 1LL << (stages_after + s - 1 - j);

                std::unordered_map<int, std::vector<C64>> curr;

                if (j == 0) {
                    curr[0] = a_stages[r];
                    curr[n1] = b_stages[r];
                    curr[-n1] = c_stages[r];
                } else {
                    for (auto &kv : prev) {
                        const int diag = kv.first;
                        const std::vector<C64> &vals = kv.second;

                        // A: same diagonal
                        auto a_prod = element_mul(a_stages[r], vals);
                        if (curr.find(diag) == curr.end()) {
                            curr[diag] = std::vector<C64>(num_slots, C64(0.0, 0.0));
                        }
                        element_add_assign(curr[diag], a_prod);

                        // B: diag + half (b multiplies x[(p+rot)%n] = LEFT rot)
                        auto rolled_neg = roll(vals, -rot);
                        auto b_prod = element_mul(b_stages[r], rolled_neg);
                        if (curr.find(diag + half) == curr.end()) {
                            curr[diag + half] = std::vector<C64>(num_slots, C64(0.0, 0.0));
                        }
                        element_add_assign(curr[diag + half], b_prod);

                        // C: diag - half (c multiplies x[(p-rot)%n] = RIGHT rot)
                        auto rolled_pos = roll(vals, rot);
                        auto c_prod = element_mul(c_stages[r], rolled_pos);
                        if (curr.find(diag - half) == curr.end()) {
                            curr[diag - half] = std::vector<C64>(num_slots, C64(0.0, 0.0));
                        }
                        element_add_assign(curr[diag - half], c_prod);
                    }
                }

                prev = std::move(curr);
            }

            // Drop near-zero diagonals.
            for (auto it = prev.begin(); it != prev.end();) {
                if (is_all_zero(it->second)) it = prev.erase(it);
                else ++it;
            }
            out.layers[layer].diagonals = std::move(prev);
            global_stage += s;
        }

        return out;
    }

    LinearTransformDiagonals
    build_s2c_diagonals(int log_n, std::vector<int> stages_per_layer) {
        // Build C2S first, then derive S2C by Hermitian-conjugating each
        // layer in reversed order (port of lapis `new_s2c`).
        LinearTransformDiagonals c2s = build_c2s_diagonals(log_n, stages_per_layer);

        const int num_layers = static_cast<int>(c2s.layers.size());
        const int num_slots = 1 << (log_n - 1);

        LinearTransformDiagonals out;
        out.n2 = c2s.n2;
        out.stages_per_layer.resize(num_layers);
        out.layers.resize(num_layers);

        for (int k = 0; k < num_layers; ++k) {
            const int rev = num_layers - 1 - k;
            const auto &src_layer = c2s.layers[rev];
            const long long R = static_cast<long long>(src_layer.rotation_unit);

            LinearTransformLayerHost dst_layer;
            dst_layer.n1 = src_layer.n1;
            dst_layer.rotation_unit = src_layer.rotation_unit;

            for (const auto &kv : src_layer.diagonals) {
                const int d = kv.first;
                const std::vector<C64> &vals = kv.second;
                const long long roll_amount = static_cast<long long>(d) * R;
                std::vector<C64> rolled = roll(vals, roll_amount);
                std::vector<C64> conj_rolled(rolled.size());
                for (size_t i = 0; i < rolled.size(); ++i) {
                    conj_rolled[i] = std::conj(rolled[i]);
                }
                dst_layer.diagonals.emplace(-d, std::move(conj_rolled));
            }

            // Drop near-zero diagonals (defensive — input shouldn't have any).
            for (auto it = dst_layer.diagonals.begin(); it != dst_layer.diagonals.end();) {
                if (is_all_zero(it->second)) it = dst_layer.diagonals.erase(it);
                else ++it;
            }

            out.layers[k] = std::move(dst_layer);
            out.stages_per_layer[k] = c2s.stages_per_layer[rev];
        }
        (void)num_slots; // unused (only retained for symmetry with build_c2s)
        return out;
    }

    // Helper: normalize a raw rotation amount into the canonical signed range
    // (-num_slots/2, num_slots/2]. Used by both the BSGS evaluator and the
    // rotation-step enumerator so the two stay perfectly in sync.
    static int normalize_step(long long raw, int num_slots) {
        long long s = ((raw % num_slots) + num_slots) % num_slots;
        if (s > num_slots / 2) s -= num_slots;
        return static_cast<int>(s);
    }

    // Per-layer rotation step set (deduplicated, sorted). Same BSGS
    // decomposition as `c2s_required_rotation_steps`, but emitted layer
    // by layer so callers can group KSKs at each layer's chain_index.
    static std::vector<std::vector<int>>
    per_layer_rotation_steps(const LinearTransformDiagonals &diags,
                             int num_slots) {
        std::vector<std::vector<int>> out(diags.layers.size());
        for (size_t li = 0; li < diags.layers.size(); ++li) {
            const auto &layer = diags.layers[li];
            const long long ru = static_cast<long long>(layer.rotation_unit);
            const int n1 = layer.n1;
            std::set<int> step_set;
            std::set<int> g_set;
            std::set<int> k_set;
            for (const auto &kv : layer.diagonals) {
                const int d = kv.first;
                int k = ((d % n1) + n1) % n1;
                int g = (d - k) / n1;
                if (k != 0) k_set.insert(k);
                if (g != 0) g_set.insert(g);
            }
            for (int k : k_set) {
                long long s = ((static_cast<long long>(k) * ru) % num_slots + num_slots) % num_slots;
                if (s > num_slots / 2) s -= num_slots;
                if (s != 0) step_set.insert(static_cast<int>(s));
            }
            for (int g : g_set) {
                long long s = ((static_cast<long long>(g) * static_cast<long long>(n1) * ru)
                               % num_slots + num_slots) % num_slots;
                if (s > num_slots / 2) s -= num_slots;
                if (s != 0) step_set.insert(static_cast<int>(s));
            }
            out[li] = std::vector<int>(step_set.begin(), step_set.end());
        }
        return out;
    }

    std::vector<int>
    c2s_required_rotation_steps(const LinearTransformDiagonals &diags,
                                int num_slots) {
        // BSGS evaluator: each diagonal d = g*n1 + k is split into a baby
        // rotation by `k * rotation_unit` and a giant rotation by
        // `g * n1 * rotation_unit`. We therefore emit only those two step
        // sets (the cross-product `d * ru` is no longer required).
        std::set<int> step_set;
        for (const auto &layer : diags.layers) {
            const long long ru = static_cast<long long>(layer.rotation_unit);
            const int n1 = layer.n1;
            std::set<int> g_set;
            std::set<int> k_set;
            for (const auto &kv : layer.diagonals) {
                const int d = kv.first;
                // BSGS decomposition: k = d mod n1 in [0, n1), g = (d-k)/n1.
                int k = ((d % n1) + n1) % n1;
                int g = (d - k) / n1;
                if (k != 0) k_set.insert(k);
                if (g != 0) g_set.insert(g);
            }
            // Baby steps: k * ru for k in observed nonzero ks.
            for (int k : k_set) {
                int s = normalize_step(static_cast<long long>(k) * ru, num_slots);
                if (s != 0) step_set.insert(s);
            }
            // Giant steps: g * n1 * ru for g in observed nonzero gs.
            for (int g : g_set) {
                int s = normalize_step(
                    static_cast<long long>(g) *
                        static_cast<long long>(n1) * ru,
                    num_slots);
                if (s != 0) step_set.insert(s);
            }
        }
        return std::vector<int>(step_set.begin(), step_set.end());
    }

    // Generic naive evaluator used by both apply_c2s_host and apply_s2c_host.
    // The math is identical — only the diagonals differ.
    static std::vector<C64>
    apply_linear_transform_host(const LinearTransformDiagonals &diags,
                                const std::vector<C64> &slot_input,
                                double last_layer_norm) {
        const int num_layers = static_cast<int>(diags.layers.size());
        const int num_slots = static_cast<int>(slot_input.size());

        double natural_product = 1.0;
        for (int s : diags.stages_per_layer) {
            natural_product *= static_cast<double>(1ULL << s);
        }
        const double correction = natural_product / last_layer_norm;
        const double correction_per_layer = std::pow(correction,
                                                     1.0 / static_cast<double>(num_layers));

        std::vector<C64> cur = slot_input;
        for (int layer = 0; layer < num_layers; ++layer) {
            const auto &L = diags.layers[layer];
            const double layer_norm = correction_per_layer /
                                      static_cast<double>(1ULL << diags.stages_per_layer[layer]);
            std::vector<C64> next(num_slots, C64(0.0, 0.0));
            for (const auto &kv : L.diagonals) {
                const int d = kv.first;
                const std::vector<C64> &vals = kv.second;
                const long long R = L.rotation_unit;
                for (int i = 0; i < num_slots; ++i) {
                    long long src_idx_raw = static_cast<long long>(i) +
                                            static_cast<long long>(d) * R;
                    long long j = ((src_idx_raw % num_slots) + num_slots) % num_slots;
                    next[i] += vals[i] * layer_norm * cur[static_cast<size_t>(j)];
                }
            }
            cur = std::move(next);
        }
        return cur;
    }

    std::vector<C64>
    apply_s2c_host(const LinearTransformDiagonals &diags,
                   const std::vector<C64> &slot_input,
                   double last_layer_norm) {
        return apply_linear_transform_host(diags, slot_input, last_layer_norm);
    }

    std::vector<C64>
    apply_c2s_host(const LinearTransformDiagonals &diags,
                   const std::vector<C64> &slot_input,
                   double last_layer_norm) {
        const int num_layers = static_cast<int>(diags.layers.size());
        const int num_slots = static_cast<int>(slot_input.size());

        // Per-lapis distribution of normalization across layers (matches
        // pre_encode_c2s_diags exactly).
        double natural_product = 1.0;
        for (int s : diags.stages_per_layer) {
            natural_product *= static_cast<double>(1ULL << s);
        }
        const double correction = natural_product / last_layer_norm;
        const double correction_per_layer = std::pow(correction,
                                                     1.0 / static_cast<double>(num_layers));

        std::vector<C64> cur = slot_input;
        for (int layer = 0; layer < num_layers; ++layer) {
            const auto &L = diags.layers[layer];
            const double layer_norm = correction_per_layer /
                                      static_cast<double>(1ULL << diags.stages_per_layer[layer]);
            std::vector<C64> next(num_slots, C64(0.0, 0.0));
            for (const auto &kv : L.diagonals) {
                const int d = kv.first;
                const std::vector<C64> &vals = kv.second;
                const long long R = L.rotation_unit;
                for (int i = 0; i < num_slots; ++i) {
                    long long src_idx_raw = static_cast<long long>(i) +
                                            static_cast<long long>(d) * R;
                    long long j = ((src_idx_raw % num_slots) + num_slots) % num_slots;
                    next[i] += vals[i] * layer_norm * cur[static_cast<size_t>(j)];
                }
            }
            cur = std::move(next);
        }
        return cur;
    }

    // ----- Pre-encoding host complex diagonals into PhantomPlaintexts -----

    // Encode one BSGS-pre-rotated complex vector at the given chain index. The
    // CKKS encoder stamps the plaintext's chain_index_ with the encode-target;
    // multiply_plain_inplace requires ct.chain_index() == pt.chain_index().
    static PhantomPlaintext
    encode_complex_diagonal(const PhantomContext &ctx,
                            PhantomCKKSEncoder &encoder,
                            const std::vector<C64> &vals,
                            size_t chain_index,
                            double scale) {
        std::vector<cuDoubleComplex> v(vals.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            v[i] = make_cuDoubleComplex(vals[i].real(), vals[i].imag());
        }
        PhantomPlaintext pt;
        encoder.encode(ctx, v, scale, pt, chain_index);
        return pt;
    }

    // ----- Light-plaintext encode + expand helpers ---------------------------

    // Kernel: convert one tower of unsigned u64 coefficients (already reduced
    // mod q_0, in [0, q_0)) into signed int64 in [-q_0/2, q_0/2). Used at
    // encode time to compress a full-RNS plaintext into a single signed-int
    // tower. Phantom NTT may leave values in [0, 2q_0); we guard with one
    // conditional subtraction first.
    __global__ void light_pt_signed_center_kernel(
            const std::uint64_t *__restrict__ src_tower0,
            std::int64_t *__restrict__ dst_signed,
            std::uint64_t q0,
            std::size_t N) {
        const std::uint64_t half = q0 >> 1;
        for (std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < N;
             tid += blockDim.x * gridDim.x) {
            std::uint64_t v = src_tower0[tid];
            if (v >= q0) v -= q0;
            std::int64_t out;
            if (v >= half) {
                // Centered representative: v - q0, in [-q0/2, 0).
                // q0 < 2^62 in our chain, so the subtraction fits in int64.
                out = -static_cast<std::int64_t>(q0 - v);
            } else {
                out = static_cast<std::int64_t>(v);
            }
            dst_signed[tid] = out;
        }
    }

    // Kernel: expand a single signed-int64 tower across `num_towers` RNS
    // towers, writing each tower's coefficients in coefficient (non-NTT) form
    // ready for forward NTT. For each coefficient c (signed):
    //   tower j gets c mod q_j, with negatives mapped via q_j - ((-c) mod q_j).
    __global__ void light_pt_expand_per_tower_kernel(
            const std::int64_t *__restrict__ src_signed,
            std::uint64_t *__restrict__ dst,  // size num_towers * N
            const DModulus *__restrict__ moduli,
            std::size_t num_towers,
            std::size_t N) {
        const std::size_t total = num_towers * N;
        for (std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < total;
             tid += blockDim.x * gridDim.x) {
            const std::size_t j = tid / N;
            const std::size_t i = tid - j * N;
            const DModulus mod = moduli[j];
            const std::uint64_t qj = mod.value();
            const std::uint64_t mu_hi = mod.const_ratio()[1];
            const std::int64_t c = src_signed[i];
            std::uint64_t out;
            if (c >= 0) {
                out = barrett_reduce_uint64_uint64(static_cast<std::uint64_t>(c), qj, mu_hi);
            } else {
                const std::uint64_t mag = static_cast<std::uint64_t>(-c);
                std::uint64_t mag_mod = barrett_reduce_uint64_uint64(mag, qj, mu_hi);
                out = (mag_mod == 0) ? 0ULL : (qj - mag_mod);
            }
            dst[tid] = out;
        }
    }

    // Int16 variant of light_pt_expand_per_tower_kernel for SingleChainPlaintext
    // storage (coeffs stored as int16 at coeff_scale). Multiplies each coeff by
    // scale_2 before the per-tower mod-q_j reduction to restore the full message
    // scale: a quantized SCP (coeff_scale=2^16) uses scale_2=2^24, a full-scale
    // SCP (coeff_scale=2^40) uses scale_2=1 (unchanged vs the int64 kernel).
    // The product |coeff| * scale_2 (~30 * 2^24 ~ 2^29) fits int64 with margin.
    __global__ void light_pt_expand_per_tower_i16_kernel(
            const std::int16_t *__restrict__ src_signed,
            std::uint64_t *__restrict__ dst,  // size num_towers * N
            const DModulus *__restrict__ moduli,
            std::int64_t scale_2,
            std::size_t num_towers,
            std::size_t N) {
        const std::size_t total = num_towers * N;
        for (std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < total;
             tid += blockDim.x * gridDim.x) {
            const std::size_t j = tid / N;
            const std::size_t i = tid - j * N;
            const DModulus mod = moduli[j];
            const std::uint64_t qj = mod.value();
            const std::uint64_t mu_hi = mod.const_ratio()[1];
            const std::int64_t c =
                static_cast<std::int64_t>(src_signed[i]) * scale_2;
            std::uint64_t out;
            if (c >= 0) {
                out = barrett_reduce_uint64_uint64(static_cast<std::uint64_t>(c), qj, mu_hi);
            } else {
                const std::uint64_t mag = static_cast<std::uint64_t>(-c);
                std::uint64_t mag_mod = barrett_reduce_uint64_uint64(mag, qj, mu_hi);
                out = (mag_mod == 0) ? 0ULL : (qj - mag_mod);
            }
            dst[tid] = out;
        }
    }

    // Encode a complex diagonal into a compact LightPlaintext.
    //
    // Implementation (one-shot, slow path used only at key generation):
    //   1. encode_complex_diagonal -> full-RNS NTT-form PhantomPlaintext at
    //      chain_index.
    //   2. INTT the FIRST tower (q_0) only.
    //   3. Signed-center that tower's coefficients into int64 on device.
    //   4. Free the full-RNS plaintext (returned to caller as RAII drop).
    //
    // Storage for the returned LightPlaintext: N * sizeof(int64_t) bytes
    // (e.g. 512 KB at logN=16) versus ~14 MB for the full plaintext.
    static LightPlaintext
    encode_to_light_plaintext(const PhantomContext &ctx,
                              PhantomCKKSEncoder &encoder,
                              const std::vector<C64> &vals,
                              std::size_t chain_index,
                              double scale) {
        const auto &stream = cudaStreamPerThread;

        // Step 1: build the full-RNS NTT-form plaintext (lives only inside
        // this function).
        PhantomPlaintext full_pt =
            encode_complex_diagonal(ctx, encoder, vals, chain_index, scale);

        const double stored_scale = full_pt.scale();

        const auto &cd = ctx.get_context_data(chain_index);
        const auto &mods = cd.parms().coeff_modulus();
        const std::size_t N = cd.parms().poly_modulus_degree();
        const std::uint64_t q0 = mods.front().value();

        // Step 2: copy tower 0 into a scratch buffer, then INTT it back to
        // coefficient form. We must NOT INTT the original plaintext in place
        // because PhantomPlaintext's storage is owned by the auto-ptr and
        // dropping it back into a partially-NTT'd state is unsafe; instead we
        // copy out then INTT the copy. Tower 0 lives at the start of the buffer.
        auto tower0 = make_cuda_auto_ptr<std::uint64_t>(N, stream);
        cudaMemcpyAsync(tower0.get(), full_pt.data(),
                        N * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
        nwt_2d_radix8_backward_inplace(tower0.get(), ctx.gpu_rns_tables(),
                                       /*coeff_modulus_size=*/1,
                                       /*start_modulus_idx=*/0, stream);

        // Step 3: signed-center the q_0 tower's coefficients into int64.
        LightPlaintext light;
        light.target_chain_index = chain_index;
        light.scale = stored_scale;
        light.coeffs_int64 = make_cuda_auto_ptr<std::int64_t>(N, stream);

        const std::size_t threads = 256;
        const std::size_t blocks = (N + threads - 1) / threads;
        light_pt_signed_center_kernel<<<blocks, threads, 0, stream>>>(
            tower0.get(), light.coeffs_int64.get(), q0, N);

        // full_pt drops at scope exit, freeing its full-RNS device buffer.
        return light;
    }

    // Expand a LightPlaintext to a full-RNS NTT-form PhantomPlaintext at the
    // target chain index baked into `light`. Uses ctx.gpu_rns_tables() for the
    // forward NTT (start_modulus_idx = 0, matching every other plaintext-encode
    // call site in this file).
    static PhantomPlaintext
    expand_light_plaintext(const PhantomContext &ctx,
                           const LightPlaintext &light) {
        const auto &stream = cudaStreamPerThread;
        const std::size_t chain_index = light.target_chain_index;
        const auto &cd = ctx.get_context_data(chain_index);
        const auto &mods = cd.parms().coeff_modulus();
        const std::size_t coeff_modulus_size = mods.size();
        const std::size_t N = cd.parms().poly_modulus_degree();

        // Allocate a fresh PhantomPlaintext sized exactly like a normal
        // encode() output (full key size, matching what multiply_plain_inplace
        // expects via PhantomPlaintext::data()).
        PhantomPlaintext pt;
        pt.set_chain_index(chain_index);
        pt.set_scale(light.scale);
        // PhantomPlaintext::resize sets coeff_modulus_size_/poly_modulus_degree_
        // and allocates the device buffer. Use the active chain's coeff_mod_size
        // (matching the size used by multiply_plain_ntt at this chain index)
        // so the buffer is exactly num_active_towers * N.
        pt.resize(coeff_modulus_size, N, stream);

        // Step 1: per-tower expand from signed int64 -> uint64 mod q_j.
        const DModulus *moduli = ctx.gpu_rns_tables().modulus();
        const std::size_t threads = 256;
        const std::size_t total = coeff_modulus_size * N;
        const std::size_t blocks = (total + threads - 1) / threads;
        light_pt_expand_per_tower_kernel<<<blocks, threads, 0, stream>>>(
            light.coeffs_int64.get(),
            pt.data(),
            moduli,
            coeff_modulus_size,
            N);

        // Step 2: forward NTT each tower in place.
        nwt_2d_radix8_forward_inplace(pt.data(), ctx.gpu_rns_tables(),
                                      coeff_modulus_size, 0, stream);

        return pt;
    }

    // For phantom's 4-section chain layout used in this port, all bootstrap
    // primes are 58-bit and the message scale equals q_msg = ~2^58. The
    // analogue of lapis's `encode_ratio` is therefore identically 1.0; we
    // simply use the per-level prime as the encode scale.
    static double encode_scale_for_layer(const PhantomContext &ctx,
                                         size_t chain_index) {
        const auto &cd = ctx.get_context_data(chain_index);
        const auto &mods = cd.parms().coeff_modulus();
        // The "current level's prime" in lapis is q_list[level]; in phantom,
        // that's the last prime in the modulus chain at this chain_index.
        return static_cast<double>(mods.back().value());
    }

    // Pre-encode any diagonal-set (C2S or S2C) starting at `start_chain_index`,
    // consuming one chain index per layer. Math is identical between the two
    // — only the diagonals and the start level differ.
    //
    // `last_layer_user_scale`: when > 0, the LAST layer is encoded at this
    // scale (typically `user_scale = 2^40`) instead of the chain prime at
    // that level. After multiply+rescale, ct.scale becomes user_scale (down
    // from q_msg). This bakes the post-bootstrap scale-down into the linear
    // transform — see lapis `apply_s2c_level_down_and_sub_raised`.
    static LinearTransformParams
    pre_encode_diagonals(const PhantomContext &ctx,
                         PhantomCKKSEncoder &encoder,
                         const LinearTransformDiagonals &host_diags,
                         double last_layer_norm,
                         size_t start_chain_index,
                         double last_layer_user_scale = 0.0,
                         const std::vector<double> &per_layer_value_norm = {}) {
        LinearTransformParams out;
        out.stages_per_layer = host_diags.stages_per_layer;
        out.n2 = host_diags.n2;
        const int num_layers = static_cast<int>(host_diags.layers.size());
        out.layers.resize(num_layers);

        // Optional per-layer value-domain normalization. When non-empty, must
        // have one entry per layer; each layer's diagonals get an extra
        // multiplicative factor baked in. After multiply+rescale per stage,
        // ct.scale metadata is preserved (still encodes at the chain prime),
        // but the accumulated value shrinks by the cumulative product. This
        // mirrors the_lib's mechanism for absorbing scale gaps between
        // sections of the chain that use different prime bit sizes (e.g.
        // C2S→EvalMod 58→54 boundary on the use17 chain consumes
        // [0.5, 0.5, 0.25] for cumulative 1/16 = 2^-4 value attenuation).
        if (!per_layer_value_norm.empty() &&
            static_cast<int>(per_layer_value_norm.size()) != num_layers) {
            throw std::invalid_argument(
                "pre_encode_diagonals: per_layer_value_norm size must match "
                "number of layers (or be empty for default 1.0)");
        }

        double natural_product = 1.0;
        for (int s : host_diags.stages_per_layer) {
            natural_product *= static_cast<double>(1ULL << s);
        }
        const double correction = natural_product / last_layer_norm;
        const double correction_per_layer = std::pow(correction,
                                                     1.0 / static_cast<double>(num_layers));

        for (int layer = 0; layer < num_layers; ++layer) {
            const auto &L_in = host_diags.layers[layer];
            auto &L_out = out.layers[layer];
            L_out.n1 = L_in.n1;
            L_out.rotation_unit = L_in.rotation_unit;

            const int s = host_diags.stages_per_layer[layer];
            const double value_norm_factor = per_layer_value_norm.empty()
                ? 1.0
                : per_layer_value_norm[layer];
            const double layer_norm = (correction_per_layer /
                                      static_cast<double>(1ULL << s)) *
                                      value_norm_factor;

            const size_t target_chain = start_chain_index + static_cast<size_t>(layer);
            const bool is_last = (layer == num_layers - 1);
            const double encode_scale =
                (is_last && last_layer_user_scale > 0.0)
                    ? last_layer_user_scale
                    : encode_scale_for_layer(ctx, target_chain);

            const int n1_layer = L_in.n1;
            const long long R_layer = static_cast<long long>(L_in.rotation_unit);
            for (const auto &kv : L_in.diagonals) {
                const int diag_idx = kv.first;
                const std::vector<C64> &vals = kv.second;

                // BSGS pre-rotation: decompose d = g*n1 + k with
                //   k = ((d % n1) + n1) % n1   in [0, n1)
                //   g = (d - k) / n1
                // and pre-roll the diagonal by `-g*n1*R` so the runtime
                // evaluator can do baby-step rotation `k*R` first, multiply,
                // then accumulate per-`g` and apply a single giant-step
                // rotation `g*n1*R` at the end.
                //
                // Identity: rot(rot(ct, k*R), g*n1*R) = rot(ct, (g*n1+k)*R)
                //   = rot(ct, d*R), and rot(roll(vals, -g*n1*R), g*n1*R) = vals,
                // so the BSGS evaluator computes the same per-slot products
                // as the naive one — but with `n1 + (#g) - 1` rotations per
                // layer instead of `#diags` rotations.
                const int k_idx = ((diag_idx % n1_layer) + n1_layer) % n1_layer;
                const int g_idx = (diag_idx - k_idx) / n1_layer;
                const long long pre_roll = -static_cast<long long>(g_idx) *
                                           static_cast<long long>(n1_layer) * R_layer;
                std::vector<C64> rotated = roll(vals, pre_roll);

                std::vector<C64> scaled(rotated.size());
                for (size_t i = 0; i < rotated.size(); ++i) {
                    scaled[i] = rotated[i] * layer_norm;
                }

                LightPlaintext light = encode_to_light_plaintext(
                    ctx, encoder, scaled, target_chain, encode_scale);
                L_out.diagonals.emplace(diag_idx, std::move(light));
            }
        }

        return out;
    }

    BootstrapKey
    create_bootstrap_key(const PhantomContext &ctx,
                         PhantomCKKSEncoder &encoder,
                         const PhantomSecretKey &dense_sk,
                         std::size_t sparse_hamming_weight,
                         std::size_t eval_mod_levels,
                         double user_scale,
                         bool split_scale_down,
                         bool use_bootstrap_to_17_levels) {
        BootstrapKey bk;

        // 1. Encapsulation key pair (ModRaise).
        bk.small = create_small_bootstrap_key(ctx, dense_sk, sparse_hamming_weight);

        // 2. Relin key (consumed by Phase 4 EvalMod, but cheap to make now).
        bk.relin_key = const_cast<PhantomSecretKey &>(dense_sk).gen_relinkey(ctx);

        // 3. Galois keys: generate the full bundle ONCE (temporary), then:
        //    a) Clone+truncate per (layer, step) into c2s/s2c per-layer maps.
        //    b) Rebuild user_galois_keys as a MINIMAL bundle (user steps +
        //       conjugation only) so the full bundle can be freed immediately.
        //    Net result: peak GPU memory = full bundle + minimal clone copies,
        //    steady-state = minimal user bundle + per-layer truncated KSKs.

        // 4. C2S host diagonals + GPU pre-encoded plaintexts.
        const auto &top_parms = ctx.get_context_data(0).parms();
        const int log_n = static_cast<int>(arith::get_power_of_two(top_parms.poly_modulus_degree()));
        const int num_slots = 1 << (log_n - 1);

        // Standard C2S normalization (matches lapis test usage).
        std::vector<int> stages_per_layer = {5, 5, 5}; // logN=16-pinned per spec
        if (log_n != 16) {
            throw std::invalid_argument(
                "create_bootstrap_key: this phase pins logN=16 (stages={5,5,5})");
        }
        LinearTransformDiagonals c2s_host = build_c2s_diagonals(log_n, stages_per_layer);
        LinearTransformDiagonals s2c_host = build_s2c_diagonals(log_n, stages_per_layer);

        const size_t first_idx = ctx.get_first_index();
        const size_t num_c2s_layers = c2s_host.layers.size(); // always 3 (butterfly layers)
        // Both use17 and legacy use 3-stage multi-level C2S (1 prime per layer).
        const size_t num_c2s_chain_primes = num_c2s_layers;
        // C2S occupies chain indices [first_idx, first_idx + num_c2s_chain_primes).
        // After C2S the ciphertext lives at first_idx + num_c2s_chain_primes.
        // EvalMod consumes `eval_mod_levels` more, so S2C lives at
        //     first_idx + num_c2s_chain_primes + eval_mod_levels.
        const size_t s2c_start = first_idx + num_c2s_chain_primes + eval_mod_levels;

        // When eval_mod_levels > 0 (full bootstrap): use lapis C2S norm = 2*K*num_slots.
        // With K=28, the per-slot value after C2S + conjugation split is ≈ I + m/K,
        // which lies within EvalMod's operating range. EvalRound = K*ct - EvalMod(ct)
        // extracts K*I; S2C(K*I, norm=1.0) ≈ K*I_coeff; saved - S2C(EvalRound) = m.
        // When eval_mod_levels == 0 (Phase 3 round-trip): use num_slots so that
        // S2C(C2S(z)) = z (identity with S2C norm = 1.0).
        constexpr double K_EVALMOD = 28.0;  // K value for EvalMod K=28 R=3
        const double c2s_last_norm = (eval_mod_levels > 0)
            ? (2.0 * K_EVALMOD * static_cast<double>(num_slots))
            : static_cast<double>(num_slots);

        // C2S encoding: multi-stage for both use17 and legacy.
        // use17: per_layer_value_norm [0.5, 0.5, 0.25] absorbs the 4-bit gap
        //        between the 29-bit C2S primes and 54-bit ER primes.
        // legacy: no value-norm (uniform 29-bit C2S and 58-bit ER).
        {
            const std::vector<double> c2s_value_norm =
                use_bootstrap_to_17_levels
                    ? std::vector<double>{0.5, 0.5, 0.25}
                    : std::vector<double>{};
            bk.c2s = pre_encode_diagonals(
                ctx, encoder, c2s_host,
                /*last_layer_norm=*/c2s_last_norm,
                /*start_chain_index=*/first_idx,
                /*last_layer_user_scale=*/0.0,
                /*per_layer_value_norm=*/c2s_value_norm);
        }

        bk.c2s_chain_primes = num_c2s_chain_primes;

        // S2C with last_layer_norm = 1.0 (lapis convention). Combined with
        // C2S's last_layer_norm = num_slots, the round-trip
        // S2C(C2S(z)) ≈ z (DFT ∘ IDFT = identity).
        //
        // When `user_scale > 0` and `split_scale_down == false` (Phase 4
        // full bootstrap with the baked-scale-down path): encode the LAST
        // S2C layer at user_scale instead of the chain prime. This bakes
        // the q_msg → user_scale scale-down into the S2C transform, so
        // post-S2C ct.scale ≈ user_scale.
        //
        // When `split_scale_down == true`: encode every S2C layer (including
        // the last) at the chain prime, so ct.scale is preserved across the
        // S2C and matches saved's q_msg-aligned scale before subtraction.
        // bootstrap() then performs a single multiply+rescale on the small
        // saved-out residual to land at user_scale (1 extra user level,
        // higher precision).
        // Step 3 of heterogeneous-scale fix: on the use17 chain, EvalMod outputs
        // at scale er_chain_prime (≈2^54) instead of s2c_chain_prime (≈2^58),
        // because we removed the PRE_S2C snap that previously lied about the scale.
        // The baked last-layer scale must be adjusted so the output still lands at
        // user_scale (2^40):
        //   desired: ct_in × last_pt_scale / s2c_last_prime = user_scale
        //   with ct_in = er_chain_prime (not s2c_chain_prime):
        //   last_pt_scale = user_scale × s2c_last_prime / er_chain_prime
        //                 = user_scale × (s2c_chain_prime / er_chain_prime)
        //                 = user_scale × 2^4   (for 58-bit S2C, 54-bit ER)
        // On the legacy uniform chain er_chain_prime == s2c_chain_prime so
        // the correction factor is 1.0 and behaviour is unchanged.
        double s2c_last_layer_user_scale;
        if (split_scale_down) {
            s2c_last_layer_user_scale = 0.0;
        } else if (use_bootstrap_to_17_levels && eval_mod_levels > 0) {
            // use17: ER primes are 54-bit, S2C primes are 58-bit. EvalMod outputs
            // at er_chain_prime (2^54). The baked last-layer S2C scale must compensate:
            //   output_scale = ct_in(2^54) × last_pt_scale / s2c_last_prime(2^58) = user_scale
            //   → last_pt_scale = user_scale × (s2c_chain_prime / er_chain_prime) = user_scale × 2^4
            const double er_chain_prime =
                encode_scale_for_layer(ctx, first_idx + num_c2s_chain_primes);
            const double s2c_chain_prime =
                encode_scale_for_layer(ctx, s2c_start);
            s2c_last_layer_user_scale = user_scale * (s2c_chain_prime / er_chain_prime);
        } else {
            s2c_last_layer_user_scale = user_scale;
        }
        bk.s2c = pre_encode_diagonals(ctx, encoder, s2c_host,
                                      /*last_layer_norm=*/1.0,
                                      /*start_chain_index=*/s2c_start,
                                      /*last_layer_user_scale=*/s2c_last_layer_user_scale);

        // 6. Level-aware Galois KSK partition.
        //
        //    Build a temporary full bundle so that per-layer KSKs can be
        //    cloned from it. After cloning, the full bundle is destroyed and
        //    user_galois_keys is rebuilt as a minimal bundle (user rotation
        //    steps + conjugation only). This avoids retaining all bootstrap
        //    rotation KSKs at full-Q alongside the per-layer truncated copies.
        auto per_layer_c2s_steps = per_layer_rotation_steps(c2s_host, num_slots);
        auto per_layer_s2c_steps = per_layer_rotation_steps(s2c_host, num_slots);

        // Collect the galois_elt indices needed for the full bundle (union of
        // all per-layer steps for C2S and S2C). We only need to generate the
        // KSKs that will be cloned, so use create_galois_keys_for_indices.
        auto &galois_elts = ctx.key_galois_tool_->galois_elts();
        const size_t N = top_parms.poly_modulus_degree();

        // Helper: galois_elt → position in galois_elts vector.
        auto find_elt_idx = [&](uint32_t galois_elt) -> size_t {
            auto it = std::find(galois_elts.begin(), galois_elts.end(), galois_elt);
            if (it == galois_elts.end()) {
                throw std::logic_error(
                    "create_bootstrap_key: step missing from context galois_elts");
            }
            return static_cast<size_t>(it - galois_elts.begin());
        };

        // Partition galois_elts into bootstrap steps (used by C2S/S2C per-layer
        // maps) and user steps (conjugation + user rotations not in C2S/S2C).
        // Bootstrap-step KSKs are generated ONE AT A TIME and immediately
        // truncated to each layer's chain_index — never accumulating more than
        // one full-Q KSK at a time. User-step KSKs are generated as a small
        // bundle for `bk.user_galois_keys`.
        std::set<size_t> bootstrap_indices_set;
        for (const auto &layer_steps : per_layer_c2s_steps) {
            for (int step : layer_steps) {
                bootstrap_indices_set.insert(
                    find_elt_idx(phantom::util::get_elt_from_step(step, N)));
            }
        }
        for (const auto &layer_steps : per_layer_s2c_steps) {
            for (int step : layer_steps) {
                bootstrap_indices_set.insert(
                    find_elt_idx(phantom::util::get_elt_from_step(step, N)));
            }
        }

        // User-step indices: ONLY conjugation (galois_elt = 2N - 1). The
        // bootstrap pipeline reads bk.user_galois_keys exclusively for the
        // post-C2S conjugation rotation (see apply_galois call further
        // down). User-rotation steps are populated later by
        // CKKSEngine::ctor's override path (create_galois_keys_per_level
        // with per-step target chains). Direct callers that need user
        // rotations from `bk.user_galois_keys` must build their own bundle.
        //
        // Building only conjugation here cuts ~8.6 GiB of full-Q transient
        // peak during engine construction (47 non-overlap user-rotation
        // KSKs × ~180 MiB each were allocated then immediately destroyed
        // by the override).
        std::vector<size_t> user_indices;
        user_indices.push_back(find_elt_idx(static_cast<uint32_t>(2 * N - 1)));

        // Canonical-owner principle: for every step used anywhere in C2S/S2C,
        // generate exactly ONE physical KSK at the SHALLOWEST chain at which
        // it is needed. Deeper-chain uses borrow that KSK via a fallback
        // pointer. Phantom's keyswitch kernel drops unused primes at runtime,
        // so a shallower-chain KSK is a superset.
        //
        // Iteration order for canonical assignment:
        //   1. C2S layers, in order (chain = first_idx + li). C2S layers
        //      live at the shallowest bootstrap chains.
        //   2. S2C layers, in order (chain = s2c_start + li). These chains
        //      are deeper than every C2S chain, so any S2C step that
        //      collides with a C2S step inherits the C2S owner.
        // Within each kind we also dedup *across layers of the same kind*:
        // if step S appears in C2S layer 0 and again in C2S layer 1, layer
        // 1's slot delegates to layer 0's owner.
        enum class LayerKind : uint8_t { C2S = 0, S2C = 1 };
        struct CanonicalLoc {
            LayerKind kind;
            size_t layer_idx;
            int step;
        };
        // step (galois_elt_index in fact, but step is unique here) → canonical location.
        std::map<int, CanonicalLoc> canonical;

        auto register_canonical =
            [&](LayerKind kind,
                const std::vector<std::vector<int>> &per_layer_steps) {
                for (size_t li = 0; li < per_layer_steps.size(); ++li) {
                    for (int step : per_layer_steps[li]) {
                        if (canonical.find(step) == canonical.end()) {
                            canonical.emplace(step,
                                              CanonicalLoc{kind, li, step});
                        }
                    }
                }
            };
        register_canonical(LayerKind::C2S, per_layer_c2s_steps); // shallower first
        register_canonical(LayerKind::S2C, per_layer_s2c_steps);

        // Two-pass fill:
        //   Pass 1: walk all (kind, layer, step). At canonical (kind, layer):
        //           generate the KSK and store as `owned`. Otherwise leave
        //           the slot empty for now (fallback will be wired in pass 2).
        //   Pass 2: walk all non-canonical entries and set `fallback` to
        //           point at the canonical owner's `owned`.
        // Two passes are required because pass 1 must complete before any
        // raw pointer into a slot's `owned` is taken — otherwise std::map
        // rehash/insertion could move existing nodes (it doesn't for
        // std::map, but treating slots as stable only after pass 1 is the
        // safer convention).
        // Use resize (not assign) because the map's value type contains a
        // non-copyable PhantomRelinKey; assign(size_t, const T&) would
        // require copy-construction of the empty map.
        bk.c2s_galois_keys.clear();
        bk.c2s_galois_keys.resize(per_layer_c2s_steps.size());
        bk.s2c_galois_keys.clear();
        bk.s2c_galois_keys.resize(per_layer_s2c_steps.size());

        auto layer_chain = [&](LayerKind kind, size_t li) -> size_t {
            if (kind == LayerKind::C2S) {
                // Single-stage: all C2S butterfly layers share first_idx (one
                // 60-bit prime, no per-layer rescale). Multi-stage: each layer
                // consumes one prime → first_idx + li.
                return use_bootstrap_to_17_levels ? first_idx : (first_idx + li);
            }
            return s2c_start + li;
        };
        auto layer_map_ptr = [&](LayerKind kind, size_t li)
            -> std::map<int, PerLayerKSKSlot>* {
            return (kind == LayerKind::C2S) ? &bk.c2s_galois_keys[li]
                                            : &bk.s2c_galois_keys[li];
        };

        // Pass 1: generate exactly one KSK per canonical entry; non-canonical
        // entries get default-constructed slots (empty `owned`, null fallback).
        auto fill_layer_pass1 =
            [&](LayerKind kind,
                const std::vector<std::vector<int>> &per_layer_steps) {
                for (size_t li = 0; li < per_layer_steps.size(); ++li) {
                    auto &out_map = *layer_map_ptr(kind, li);
                    const size_t target_chain = layer_chain(kind, li);
                    for (int step : per_layer_steps[li]) {
                        auto &slot = out_map[step]; // default-construct
                        const auto &loc = canonical.at(step);
                        if (loc.kind == kind && loc.layer_idx == li) {
                            const size_t idx = find_elt_idx(
                                phantom::util::get_elt_from_step(step, N));
                            // Generate a KSK directly at this canonical layer's
                            // chain_index: only `beta_k = ceil(size_Ql/size_P)`
                            // partitions are emitted, matched to the smaller
                            // modulus. Deeper-chain layers borrow this same
                            // KSK at runtime (phantom's keyswitch drops the
                            // unused primes for them).
                            slot.owned =
                                dense_sk.create_one_galois_key(ctx, idx, target_chain);
                        }
                        // else: leave slot empty for now; pass 2 wires fallback.
                    }
                }
            };
        fill_layer_pass1(LayerKind::C2S, per_layer_c2s_steps);
        fill_layer_pass1(LayerKind::S2C, per_layer_s2c_steps);

        // Pass 2: wire fallback pointers from non-canonical slots → canonical
        // slot's `owned`. Now safe because pass 1 has populated all canonical
        // owners.
        auto fill_layer_pass2 =
            [&](LayerKind kind,
                const std::vector<std::vector<int>> &per_layer_steps) {
                for (size_t li = 0; li < per_layer_steps.size(); ++li) {
                    auto &out_map = *layer_map_ptr(kind, li);
                    for (int step : per_layer_steps[li]) {
                        const auto &loc = canonical.at(step);
                        if (loc.kind == kind && loc.layer_idx == li) {
                            continue; // canonical entry — already owns
                        }
                        auto &slot = out_map.at(step);
                        const auto *canon_map = layer_map_ptr(loc.kind, loc.layer_idx);
                        slot.fallback = &canon_map->at(loc.step).owned;
                    }
                }
            };
        fill_layer_pass2(LayerKind::C2S, per_layer_c2s_steps);
        fill_layer_pass2(LayerKind::S2C, per_layer_s2c_steps);

        // Build the minimal user bundle (conjugation + user rotation steps only).
        bk.user_galois_keys =
            dense_sk.create_galois_keys_for_indices(ctx, user_indices);

        return bk;
    }

    // Shared BSGS rotate-multiply-sum-rescale evaluator for any pre-encoded
    // BSGS diagonal stack (C2S or S2C). Iterates layers in order and consumes
    // one chain index per layer.
    //
    // Each diagonal d is decomposed as d = g*n1 + k with k in [0, n1).
    // The encoded plaintext has been pre-rolled by `-g*n1*R` (see
    // pre_encode_diagonals), so the BSGS identity holds:
    //   rot(rot(ct, k*R), g*n1*R) * rot(vals, -g*n1*R)
    //     = rot( rot(ct, k*R) * pt_pre, g*n1*R )
    //     = rot(ct, (g*n1+k)*R) * vals  (per-slot, after the giant rotation).
    //
    // Algorithm per layer:
    //   1. Compute baby[k] = rotate(ct, k*R) for each observed k != 0
    //      (baby[0] = ct itself).
    //   2. For each observed g, accumulate
    //         inner[g] = sum_{k : (g,k) is a diagonal} pt[g*n1+k] * baby[k]
    //      then rotate inner[g] by g*n1*R (giant step, skipped when g == 0
    //      or when the giant step normalizes to 0).
    //   3. Sum the per-g accumulators, rescale.
    //
    // For our DIF butterfly with stages={5,5,5} and n1=16, observed k's span
    // [0,16) and g's typically lie in {-2,-1,0,1}. The total rotation count
    // per layer is (#nonzero k) + (#nonzero g) instead of (#diagonals).
    static void apply_linear_transform_inplace(const PhantomContext &ctx,
                                               PhantomCiphertext &ct,
                                               const LinearTransformParams &params,
                                               const std::vector<std::map<int, PerLayerKSKSlot>> &per_layer_galois,
                                               const char *who,
                                               int layer_start = 0,
                                               int layer_end = -1) {
        const int num_layers = static_cast<int>(params.layers.size());
        if (static_cast<int>(per_layer_galois.size()) != num_layers) {
            throw std::logic_error(
                std::string(who) + ": per-layer galois map count does not match layer count");
        }
        if (layer_end < 0) layer_end = num_layers;
        for (int layer = layer_start; layer < layer_end; ++layer) {
            const auto &L = params.layers[layer];
            if (L.diagonals.empty()) {
                throw std::logic_error(std::string(who) + ": layer has no diagonals");
            }

            const size_t expected_chain = L.diagonals.begin()->second.target_chain_index;
            if (ct.chain_index() != expected_chain) {
                throw std::invalid_argument(
                    std::string(who) + ": ct.chain_index() does not match layer plaintexts");
            }

            const auto &cd = ctx.get_context_data(ct.chain_index());
            const int num_slots = static_cast<int>(cd.parms().poly_modulus_degree() / 2);
            const long long R = static_cast<long long>(L.rotation_unit);
            const int n1 = L.n1;

            const auto &layer_ksks = per_layer_galois[static_cast<size_t>(layer)];
            auto get_layer_ksk = [&](int step) -> const PhantomRelinKey & {
                auto it = layer_ksks.find(step);
                if (it == layer_ksks.end()) {
                    throw std::logic_error(
                        std::string(who) + ": missing KSK for step in layer's per-layer map");
                }
                // Resolve owned-or-fallback: a non-canonical slot delegates
                // to the canonical owner elsewhere in the BootstrapKey
                // (shallower-chain superset KSK shared across all uses).
                return it->second.get();
            };

            // Strategy B: expand ALL LightPlaintexts in this layer to full
            // PhantomPlaintexts up-front. Per-layer peak (~32 expanded × ~14 MB
            // ≈ 450 MB) is bounded and predictable; the expanded map is dropped
            // at the end of the layer iteration. Saves ~1.3 GB of construction-
            // time storage versus keeping every diagonal at full size.
            std::unordered_map<int, PhantomPlaintext> expanded;
            expanded.reserve(L.diagonals.size());
            for (const auto &kv : L.diagonals) {
                expanded.emplace(kv.first, expand_light_plaintext(ctx, kv.second));
            }

            // Group diagonals by their g-bucket. Each bucket holds the (k, pt*)
            // pairs that share the same giant-step rotation amount.
            std::map<int, std::vector<std::pair<int, const PhantomPlaintext *>>> by_g;
            std::set<int> needed_babies; // k in [1, n1) actually used
            for (const auto &kv : L.diagonals) {
                const int d = kv.first;
                int k = ((d % n1) + n1) % n1;
                int g = (d - k) / n1;
                auto it_exp = expanded.find(d);
                if (it_exp == expanded.end()) {
                    throw std::logic_error(
                        std::string(who) + ": expanded plaintext missing for diagonal");
                }
                by_g[g].emplace_back(k, &it_exp->second);
                if (k != 0) needed_babies.insert(k);
            }

            // 1. Baby steps: rotate ct by k*R for each observed nonzero k.
            //    baby[0] aliases ct itself (we don't rotate by 0).
            std::unordered_map<int, PhantomCiphertext> baby;
            for (int k : needed_babies) {
                int step = normalize_step(static_cast<long long>(k) * R, num_slots);
                if (step == 0) {
                    // Degenerate (e.g. k*R == num_slots): the rotation is a
                    // no-op, so reuse ct.
                    continue;
                }
                baby.emplace(k, rotate(ctx, ct, step, get_layer_ksk(step)));
            }
            auto get_baby = [&](int k) -> const PhantomCiphertext & {
                if (k == 0) return ct;
                auto it = baby.find(k);
                if (it != baby.end()) return it->second;
                // k != 0 but k*R reduced to 0: same as ct.
                return ct;
            };

            // 2. Per-g accumulators: inner[g] = sum_k pt[g*n1+k] * baby[k].
            //    Then giant-rotate by g*n1*R and accumulate into layer_out.
            PhantomCiphertext layer_out;
            bool layer_out_init = false;
            for (const auto &g_kv : by_g) {
                const int g = g_kv.first;
                const auto &pairs = g_kv.second;

                PhantomCiphertext inner;
                bool inner_init = false;
                for (const auto &p : pairs) {
                    const int k = p.first;
                    const PhantomPlaintext &pt = *p.second;
                    PhantomCiphertext term = get_baby(k);
                    multiply_plain_inplace(ctx, term, pt);
                    if (!inner_init) {
                        inner = std::move(term);
                        inner_init = true;
                    } else {
                        add_inplace(ctx, inner, term);
                    }
                }
                if (!inner_init) continue;

                // Giant rotation by g*n1*R (skip when g == 0 or step
                // normalizes to 0).
                if (g != 0) {
                    int giant_step = normalize_step(
                        static_cast<long long>(g) *
                            static_cast<long long>(n1) * R,
                        num_slots);
                    if (giant_step != 0) {
                        rotate_inplace(ctx, inner, giant_step, get_layer_ksk(giant_step));
                    }
                }

                if (!layer_out_init) {
                    layer_out = std::move(inner);
                    layer_out_init = true;
                } else {
                    add_inplace(ctx, layer_out, inner);
                }
            }

            if (!layer_out_init) {
                throw std::logic_error(std::string(who) + ": empty layer accumulator");
            }
            ct = std::move(layer_out);
            rescale_to_next_inplace(ctx, ct);
        }
    }

    void apply_c2s_inplace(const PhantomContext &ctx,
                           PhantomCiphertext &ct,
                           const BootstrapKey &bk) {
        apply_linear_transform_inplace(ctx, ct, bk.c2s, bk.c2s_galois_keys,
                                       "apply_c2s_inplace",
                                       /*layer_start=*/0, /*layer_end=*/-1);
    }

    void apply_s2c_inplace(const PhantomContext &ctx,
                           PhantomCiphertext &ct,
                           const BootstrapKey &bk) {
        apply_linear_transform_inplace(ctx, ct, bk.s2c, bk.s2c_galois_keys,
                                       "apply_s2c_inplace");
    }

    // ========================================================================
    // Phase 4: full bootstrap pipeline
    //   scale_up → mod_raise → C2S → eval_round → S2C → final scale-down
    // ========================================================================

    namespace {

        // Per-tower integer scalar multiply on a 2-element CKKS ciphertext (in
        // NTT form). For each tower j, ct[j][i] ← (ct[j][i] · scalar_j) mod q_j.
        // Does not change ct.scale or ct.chain_index — caller manages metadata.
        void multiply_uint_scalars_per_tower(const PhantomContext &ctx,
                                             PhantomCiphertext &ct,
                                             const std::vector<uint64_t> &scalars_per_tower) {
            const auto &cd = ctx.get_context_data(ct.chain_index());
            const auto &parms = cd.parms();
            const size_t poly_degree = parms.poly_modulus_degree();
            const size_t coeff_mod_size = parms.coeff_modulus().size();
            if (scalars_per_tower.size() != coeff_mod_size) {
                throw std::invalid_argument(
                    "multiply_uint_scalars_per_tower: scalar count != coeff_mod_size");
            }
            const auto &stream = cudaStreamPerThread;

            const DModulus *moduli = ctx.gpu_rns_tables().modulus();
            // Per-tower integer multiply: one kernel launch per tower so each
            // tower can have its own scalar (the existing multiply_scalar_rns_poly
            // takes a uniform scalar across all towers, which is wrong here).
            const size_t threads = 256;
            const size_t blocks = (poly_degree + threads - 1) / threads;
            for (size_t poly = 0; poly < ct.size(); ++poly) {
                for (size_t t = 0; t < coeff_mod_size; ++t) {
                    const uint64_t scalar = scalars_per_tower[t];
                    if (scalar == 1) continue;
                    multiply_scalar_per_tower_kernel<<<blocks, threads, 0, stream>>>(
                        ct.data() + poly * coeff_mod_size * poly_degree + t * poly_degree,
                        scalar,
                        moduli + t,
                        poly_degree);
                }
            }
        }

        // Pre-bootstrap scale-up: lift ct.scale from `current_scale` (e.g. 2^40)
        // to q_msg (e.g. 2^58) so mod_raise's tile-encoding aligns.
        //
        // Mirrors lapis `scale_up_for_bootstrap`: ct must be at the level ONE
        // ABOVE the single-prime bottom (2 primes: [q_msg, q_scale]).
        //   c0 = round(q_msg * q_scale / current_scale)  [u128-safe]
        //   ct ← ct * c0   (per-tower modular multiply, same c0 in each tower)
        //   ct ← rescale_to_next(ct)  (drops q_scale, divides scale by q_scale)
        // After rescale: ct is at the bottom (1 prime = q_msg), scale = q_msg.
        //
        // If current_scale == q_msg exactly (chained bootstrap), the scale is
        // already matched — skip the multiply+rescale (saves one level).
        void scale_up_for_bootstrap(const PhantomContext &ctx,
                                    PhantomCiphertext &ct,
                                    double current_scale) {
            const size_t bottom_index = ctx.total_parm_size() - 1;
            const auto &bottom_cd = ctx.get_context_data(bottom_index);
            const uint64_t q_msg = bottom_cd.parms().coeff_modulus()[0].value();
            const double q_msg_d = static_cast<double>(q_msg);

            // If ct is already at the bottom, two valid cases:
            //   (a) scale already == q_msg: chained bootstrap, no-op.
            //   (b) scale == user_scale  : engine path where the user depleted
            //       via rescale-to-next and ct sits at single-prime bottom.
            //       Lift scale by integer-ratio multiply (no rescale needed —
            //       there's no prime left to drop). New scale = user_scale * c0.
            if (ct.chain_index() == bottom_index) {
                if (std::abs(current_scale / q_msg_d - 1.0) < 1e-6) {
                    ct.set_scale(q_msg_d);
                    return;
                }
                // Case (b): in-place lift by c0 = round(q_msg / current_scale).
                const uint64_t cs_u64 = static_cast<uint64_t>(current_scale);
                const bool cs_is_int = cs_u64 > 0 && static_cast<double>(cs_u64) == current_scale;
                uint64_t c0;
                if (cs_is_int) {
                    c0 = (q_msg + cs_u64 / 2) / cs_u64;
                } else {
                    c0 = static_cast<uint64_t>(std::llround(q_msg_d / current_scale));
                }
                if (c0 == 0) {
                    throw std::invalid_argument(
                        "scale_up_for_bootstrap: at bottom but ratio q_msg/current_scale rounds to 0");
                }
                std::vector<uint64_t> scalars(1, c0);  // single tower at bottom
                multiply_uint_scalars_per_tower(ctx, ct, scalars);
                ct.set_scale(current_scale * static_cast<double>(c0));
                return;
            }

            // ct is at level above bottom: 2+ primes, top prime = q_scale.
            const auto &cd = ctx.get_context_data(ct.chain_index());
            const auto &mods = cd.parms().coeff_modulus();
            // q_scale is the top prime at the current level (will be rescaled away).
            const uint64_t q_scale = mods.back().value();
            const double q_scale_d = static_cast<double>(q_scale);

            // c0 = round(q_msg * q_scale / current_scale).
            // Both q_msg and q_scale are ~2^58; current_scale ~ 2^40.
            // c0 ~ 2^76, fits in uint64. Use u128 for the intermediate product.
            const uint64_t cs_u64 = static_cast<uint64_t>(current_scale);
            const bool cs_is_int = cs_u64 > 0 && static_cast<double>(cs_u64) == current_scale;
            const __uint128_t num128 = (__uint128_t)q_msg * (__uint128_t)q_scale;
            uint64_t c0;
            if (cs_is_int) {
                const __uint128_t cs128 = cs_u64;
                c0 = static_cast<uint64_t>((num128 + cs128 / 2) / cs128);
            } else {
                c0 = static_cast<uint64_t>(std::llround(
                    static_cast<double>(num128) / current_scale));
            }
            if (c0 == 0) {
                throw std::invalid_argument(
                    "scale_up_for_bootstrap: computed c0 is zero");
            }

            // Multiply all towers of ct by c0 (mod each prime).
            const size_t num_towers = mods.size();
            std::vector<uint64_t> scalars(num_towers, c0);
            multiply_uint_scalars_per_tower(ctx, ct, scalars);

            // Update scale metadata: new scale = current_scale * c0.
            // After rescale_to_next (which divides by q_scale):
            //   final scale = current_scale * c0 / q_scale ≈ q_msg.
            ct.set_scale(current_scale * static_cast<double>(c0));
            rescale_to_next_inplace(ctx, ct);

            // Snap to exact q_msg to remove floating-point drift.
            ct.set_scale(q_msg_d);
        }

        // Internal: K · ct via per-tower integer multiply (no rescale, no level
        // cost). The encoded message becomes K · m, scale unchanged. Works at
        // any chain index.
        void multiply_int_inplace(const PhantomContext &ctx,
                                  PhantomCiphertext &ct,
                                  uint64_t k) {
            const auto &cd = ctx.get_context_data(ct.chain_index());
            const size_t coeff_mod_size = cd.parms().coeff_modulus().size();
            std::vector<uint64_t> scalars(coeff_mod_size, k);
            multiply_uint_scalars_per_tower(ctx, ct, scalars);
        }

        // Kernel: cyclic shift of poly coefficients by N/2 with negacyclic wrap.
        // For neg=false: result[k] = coeff[(k + N/2) % N] with negation when wrap.
        //   i.e. multiply by x^{N/2} in Z[x]/(x^N+1):
        //   result[k<N/2]  =  coeff[k + N/2]
        //   result[k>=N/2] = -coeff[k - N/2]
        // For neg=true:  result[k<N/2]  = -coeff[k + N/2]
        //                result[k>=N/2] =  coeff[k - N/2]   (= multiply by -x^{N/2})
        // Operates on one tower of N coefficients in [0, q). Caller ensures
        // inputs are in coefficient (non-NTT) form and fully reduced mod q.
        __global__ void shift_poly_N2_kernel(
                const uint64_t * __restrict__ src,
                uint64_t       * __restrict__ dst,
                const DModulus * __restrict__ mod_ptr,
                size_t N,
                bool   negate_wrap_side) {
            const uint64_t q = mod_ptr->value();
            const size_t half = N >> 1;
            for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                 tid < N; tid += blockDim.x * gridDim.x) {
                uint64_t v;
                if (tid < half) {
                    // result[tid] = src[tid + N/2] (no negation if neg=false)
                    v = src[tid + half];
                    if (v >= q) v -= q;
                    if (negate_wrap_side) {
                        // negate: (q - v) % q
                        v = (v == 0) ? 0ULL : (q - v);
                    }
                } else {
                    // result[tid] = -src[tid - N/2] (negate for x^{N/2} multiply)
                    v = src[tid - half];
                    if (v >= q) v -= q;
                    if (!negate_wrap_side) {
                        // default (neg=false): negate the upper half
                        v = (v == 0) ? 0ULL : (q - v);
                    }
                }
                dst[tid] = v;
            }
        }

        // Multiply ct in-place by ±x^{N/2} (= ±i in the CKKS slot sense).
        // neg=false: multiply by +x^{N/2}  (+i in slots)
        // neg=true:  multiply by -x^{N/2}  (-i in slots)
        // Implements lapis's mult_imag_inplace using INTT → poly-shift → NTT.
        void multiply_x_pow_N2_inplace(const PhantomContext &ctx,
                                       PhantomCiphertext &ct,
                                       bool neg,
                                       const cudaStream_t &stream) {
            if (!ct.is_ntt_form()) {
                throw std::invalid_argument("multiply_x_pow_N2_inplace: ct must be in NTT form");
            }
            const auto &cd = ctx.get_context_data(ct.chain_index());
            const size_t coeff_mod_size = cd.parms().coeff_modulus().size();
            const size_t N = cd.parms().poly_modulus_degree();
            const size_t poly_size = ct.size(); // 2 for a standard ciphertext

            // NTT table start index: chain_index maps to which moduli are active.
            // In phantom, the NTT table for the current level starts at index
            // (total_parms - 1 - chain_index) * ... — actually it depends on the
            // chain layout. The simplest approach: use start_modulus_idx = 0
            // (the beginning of the RNS table) which is correct when the chain
            // has dropped from the top down. The current chain_index tells us
            // which primes remain active; their NTT twiddles start at the right
            // place. In mod_raise_extend_modulus the nwt calls use start_idx=0
            // because all top-level moduli are present. Here we need the index
            // corresponding to the active moduli.
            //
            // Phantom stores the ordinary primes in layout:
            //   chain_index = first_idx = 1 → all Q ordinary primes (indices 0..Q-1)
            //   chain_index = first_idx+1    → Q-1 primes (indices 0..Q-2)
            //   chain_index = total-1        → 1 prime (index 0)
            // So the NTT start index = 0 always (primes 0..coeff_mod_size-1).
            const size_t ntt_start_idx = 0;

            const DModulus *moduli = ctx.gpu_rns_tables().modulus();

            const size_t threads = 256;
            const size_t blocks  = (N + threads - 1) / threads;

            for (size_t poly = 0; poly < poly_size; ++poly) {
                uint64_t *poly_ptr = ct.data() + poly * coeff_mod_size * N;

                // Step 1: INTT this polynomial (all towers together).
                nwt_2d_radix8_backward_inplace(poly_ptr, ctx.gpu_rns_tables(),
                                               coeff_mod_size, ntt_start_idx, stream);

                // Step 2: per-tower cyclic shift by N/2.
                // We need a temporary buffer for one tower at a time.
                auto tmp = make_cuda_auto_ptr<uint64_t>(N, stream);
                for (size_t t = 0; t < coeff_mod_size; ++t) {
                    uint64_t *tower_ptr = poly_ptr + t * N;
                    shift_poly_N2_kernel<<<blocks, threads, 0, stream>>>(
                        tower_ptr, tmp.get(),
                        moduli + t,
                        N, neg);
                    cudaMemcpyAsync(tower_ptr, tmp.get(), N * sizeof(uint64_t),
                                   cudaMemcpyDeviceToDevice, stream);
                }

                // Step 3: forward NTT back.
                nwt_2d_radix8_forward_inplace(poly_ptr, ctx.gpu_rns_tables(),
                                              coeff_mod_size, ntt_start_idx, stream);
            }
        }

        // EvalRound = K · ct − EvalMod(ct).  Internal (Phase 4 only).
        PhantomCiphertext eval_round_k16_r3(const PhantomContext &ctx,
                                            PhantomCKKSEncoder &encoder,
                                            const PhantomCiphertext &ct,
                                            const PhantomRelinKey &rk) {
            // K · ct (level-free).
            PhantomCiphertext kct = ct;
            multiply_int_inplace(ctx, kct, /*k=*/16ULL);

            // EvalMod(ct) — consumes 9 levels.
            PhantomCiphertext em = evalmod_k16_r3(ctx, encoder, ct, rk);

            // Align kct to em's chain_index (em is deeper).
            if (kct.chain_index() < em.chain_index()) {
                mod_switch_to_inplace(ctx, kct, em.chain_index());
            }
            // Snap scales to the same value (em's snap target). This matches
            // evalmod's snap_scale convention: both ciphertexts must have
            // matching scale metadata before sub_inplace.
            kct.set_scale(em.scale());

            // result = K·ct − EvalMod(ct).
            sub_inplace(ctx, kct, em);
            return kct;
        }

        // EvalRound = K · ct − EvalMod(ct) for K=28 R=3. Internal (Phase 4 only).
        // Same structure as eval_round_k16_r3 but with K=28 and the degree-49
        // polynomial from the_lib. Same 9-level chain budget.
        //
        // Heterogeneous-scale fix (Step 2): On the use17 chain the ER section
        // uses 54-bit primes, but the CT entering EvalMod carries scale metadata
        // 2^58 (from C2S + conj_split). evalmod_k28_r3 sets
        //   target_scale = ct.scale() = 2^58
        // and snaps after every rescale to 2^58. When the actual rescale prime
        // is 2^54, each rescale produces scale 2^116/2^54 = 2^62, which snap
        // rounds down to 2^58 — hiding 4 bits of value amplification. This
        // compounds exponentially over the 9 EvalMod levels.
        //
        // Fix: set ct.scale to the actual chain prime at the evalmod entry
        // (chain_prime_at(ct.chain_index())) so that target_scale = chain_prime,
        // and after each rescale: target_scale^2 / chain_prime = chain_prime —
        // no drift. On the legacy 58-bit chain, chain_prime ≈ 2^58 ≈ ct.scale()
        // so this is a near-no-op (epsilon difference only).
        PhantomCiphertext eval_round_k28_r3(const PhantomContext &ctx,
                                            PhantomCKKSEncoder &encoder,
                                            const PhantomCiphertext &ct,
                                            const PhantomRelinKey &rk) {
            // Snap ct.scale to the actual chain prime at the evalmod entry so
            // evalmod's internal target_scale matches the prime in every rescale.
            const double er_chain_prime = static_cast<double>(
                ctx.get_context_data(ct.chain_index())
                    .parms().coeff_modulus().back().value());
            PhantomCiphertext ct_snapped = ct;
            ct_snapped.set_scale(er_chain_prime);

            PhantomCiphertext kct = ct_snapped;
            multiply_int_inplace(ctx, kct, /*k=*/28ULL);

            PhantomCiphertext em = evalmod_k28_r3(ctx, encoder, ct_snapped, rk);

            if (kct.chain_index() < em.chain_index()) {
                mod_switch_to_inplace(ctx, kct, em.chain_index());
            }
            kct.set_scale(em.scale());

            sub_inplace(ctx, kct, em);
            return kct;
        }

        // EvalRound = K · ct − EvalMod(ct) for K=28 R=4. Internal (Phase 4 only).
        // Same K=28, but one extra double-angle iteration → 10-level chain budget
        // (6 sine + 4 DA). Higher per-slot precision (~30 bits) than R=3 (~27 bits).
        PhantomCiphertext eval_round_k28_r4(const PhantomContext &ctx,
                                            PhantomCKKSEncoder &encoder,
                                            const PhantomCiphertext &ct,
                                            const PhantomRelinKey &rk) {
            // Same heterogeneous-scale fix as eval_round_k28_r3: snap ct.scale
            // to the actual chain prime at the evalmod entry.
            const double er_chain_prime = static_cast<double>(
                ctx.get_context_data(ct.chain_index())
                    .parms().coeff_modulus().back().value());
            PhantomCiphertext ct_snapped = ct;
            ct_snapped.set_scale(er_chain_prime);

            PhantomCiphertext kct = ct_snapped;
            multiply_int_inplace(ctx, kct, /*k=*/28ULL);

            PhantomCiphertext em = evalmod_k28_r4(ctx, encoder, ct_snapped, rk);

            if (kct.chain_index() < em.chain_index()) {
                mod_switch_to_inplace(ctx, kct, em.chain_index());
            }
            kct.set_scale(em.scale());

            sub_inplace(ctx, kct, em);
            return kct;
        }

    } // namespace

    PhantomCiphertext
    bootstrap(const PhantomContext &ctx,
              PhantomCKKSEncoder &encoder,
              const PhantomCiphertext &ct,
              const BootstrapKey &bk,
              double user_scale,
              bool split_scale_down,
              bool use_bootstrap_to_17_levels,
              int evalmod_r) {
        PhantomCiphertext out = ct;

        // 1. Pre-bootstrap scale-up: ct.scale = user_scale → q_msg.
        //
        // BootstrapTo17Levels chain: q_msg == user_scale == 2^40 (bits[0]
        // IS the user-scale segment's first prime). The_lib's
        // SCALE_UP_RATIO = q_msg / user_scale = 1, so scale_up is a logical
        // no-op. scale_up_for_bootstrap's branch (a) handles this: snaps
        // ct.scale to q_msg without consuming a level.
        scale_up_for_bootstrap(ctx, out, user_scale);

        // 2. Encapsulated mod-raise: bottom → top of chain (chain_index = first_idx).
        mod_raise_inplace(ctx, out, bk.small);

        // 3. Save the mod-raised ct for the final subtraction (lapis "ct_saved").
        //    After S2C(EvalRound(conj_split(C2S(ct)))), the result ≈ K·I coeff form.
        //    saved_ct − qi = (m + K·I) − K·I = m  (up to EvalMod error).
        PhantomCiphertext saved = out;

        // 4. Align to C2S input level (should already be at first_idx after mod_raise).
        const size_t c2s_in_chain =
            bk.c2s.layers[0].diagonals.begin()->second.target_chain_index;
        if (out.chain_index() != c2s_in_chain) {
            mod_switch_to_inplace(ctx, out, c2s_in_chain);
        }

        // 5. C2S: maps coefficient form → slot form.
        //    With last_layer_norm = 2*K*num_slots (lapis ER convention), the output
        //    values are scaled by 1/(2*K*num_slots) relative to natural butterfly norm.
        //    use17: single-stage (3 multiply_plain + 1 rescale of 60-bit prime).
        //    legacy: multi-stage (3 × multiply_plain + rescale per layer).
        apply_c2s_inplace(ctx, out, bk);

        // 6. Conjugation split (lapis "coeff_to_slot_pooled" steps 5–6):
        //    conj(ct) via Galois automorphism x → x^{2N-1}.
        //    real = ct + conj(ct) = 2·Re(slots)
        //    imag = -i·(ct - conj(ct)) = 2·Im(slots)   [via multiply by -x^{N/2}]
        //
        //    Both tracks are needed: I integers from mod_raise have complex slot
        //    images (Im ≠ 0 even for real input m). Dropping the imag track leaves
        //    Im(I_coeff) uncancelled in saved - qi.
        const size_t N = ctx.get_context_data(0).parms().poly_modulus_degree();
        const size_t conj_galois_elt = 2 * N - 1;
        PhantomCiphertext conj_ct = apply_galois(ctx, out, conj_galois_elt, bk.user_galois_keys);

        // imag_pre = out - conj_ct = 2i·Im(slots)
        PhantomCiphertext imag_ct = out;
        sub_inplace(ctx, imag_ct, conj_ct);
        // imag = (-x^{N/2}) · imag_pre = -i · 2i·Im = 2·Im(slots)
        multiply_x_pow_N2_inplace(ctx, imag_ct, /*neg=*/true, cudaStreamPerThread);

        // real = out + conj_ct = 2·Re(slots)
        add_inplace(ctx, out, conj_ct);

        // 7. EvalRound = K·ct − EvalMod(ct) on BOTH real and imag tracks.
        //    Matches lapis bootstrap_evalround_plus_with_r steps 8–9.
        //    R=3 (9 ER levels, ~27-bit poly precision, good on |x| <= ~0.7)
        //    R=4 (10 ER levels, ~30-bit poly precision, good on |x| <= ~1.0)
        PhantomCiphertext real_round, imag_round;
        if (evalmod_r == 4) {
            real_round = eval_round_k28_r4(ctx, encoder, out, bk.relin_key);
            imag_round = eval_round_k28_r4(ctx, encoder, imag_ct, bk.relin_key);
        } else {
            real_round = eval_round_k28_r3(ctx, encoder, out, bk.relin_key);
            imag_round = eval_round_k28_r3(ctx, encoder, imag_ct, bk.relin_key);
        }

        // 8. Recombine real and imag for S2C input.
        //    lapis slot_to_coeff step 1: imag_rot = (+x^{N/2}) · imag_round = +i·imag
        //    combined = real_round + imag_rot
        //    S2C(combined) reconstructs the full coeff-domain I vector.
        multiply_x_pow_N2_inplace(ctx, imag_round, /*neg=*/false, cudaStreamPerThread);
        // Align scales/levels before add (EvalRound on both tracks uses same levels).
        if (real_round.chain_index() != imag_round.chain_index()) {
            if (real_round.chain_index() < imag_round.chain_index()) {
                mod_switch_to_inplace(ctx, real_round, imag_round.chain_index());
            } else {
                mod_switch_to_inplace(ctx, imag_round, real_round.chain_index());
            }
        }
        imag_round.set_scale(real_round.scale());
        add_inplace(ctx, real_round, imag_round);
        out = std::move(real_round);

        // 9. S2C: align to S2C input level (may need a mod_switch if eval_mod_levels
        //    encoded in the BootstrapKey doesn't exactly match EvalRound's actual depth).
        const size_t s2c_in_chain =
            bk.s2c.layers[0].diagonals.begin()->second.target_chain_index;
        if (out.chain_index() < s2c_in_chain) {
            mod_switch_to_inplace(ctx, out, s2c_in_chain);
        } else if (out.chain_index() > s2c_in_chain) {
            throw std::logic_error(
                "bootstrap: ct deeper than S2C start — eval_mod_levels mismatch");
        }
        // Do NOT snap ct.scale here. ct carries its honest EvalMod output scale
        // (≈2^54 on use17, ≈2^58 on legacy) into S2C. multiply_plain_ntt sets
        // new_scale = ct.scale × pt.scale; rescale_to_next divides by the S2C
        // chain prime, so metadata is preserved through all S2C layers.
        apply_s2c_inplace(ctx, out, bk);

        // `out` = qi = S2C(EvalRound(C2S(ct_raised))) ≈ I_coeff (coeff domain).
        const size_t qi_chain = out.chain_index();

        // 9. Bring saved into alignment with qi's level AND scale, then subtract.
        //    saved currently encodes (m + K·I) at scale q_msg (top-of-chain
        //    after mod_raise). qi (= out) encodes K·I_coeff at scale
        //        out.scale() = first_back · user_scale / last_back
        //    where `first_back` is the back prime at S2C's input level (the
        //    snap-in scale) and `last_back` is the back prime at the chain
        //    where S2C's last layer rescaled.
        //
        //    The previous implementation used multiply_plain(1.0 at user_scale)
        //    + rescale, which after rescale leaves saved at scale
        //        q_msg · user_scale / q_back_saved
        //    (where q_back_saved is the back prime at saved's pre-rescale
        //    chain). Then saved.set_scale(out.scale()) hides the residual
        //    factor q_msg/first_back ≠ 1 between the two integers — yielding
        //    per-slot error of order K·|I|·|q_msg − first_back| / first_back
        //    ≈ K·|I| · 2^-29, which at K·|I|≈5000 and 58-bit primes gives
        //    ~9e-6 absolute error (~14 bits, matches observed ~8e-5).
        //
        //    Fix: encode the constant 1.0 at scale
        //        D_exact = round( out.scale() · q_back_saved / q_msg )
        //    computed exactly via u128 integer arithmetic. After
        //    multiply_plain(1.0 at D_exact) + rescale_to_next, saved's scale
        //    becomes
        //        q_msg · D_exact / q_back_saved ≈ out.scale()
        //    (exact up to integer-rounding of D_exact, ≪ 1 ulp at scale
        //    ~2^40), and the integer encodes (m+K·I)·out.scale() with no
        //    spurious q_msg/first_back drift.
        if (std::abs(out.scale() - saved.scale()) > 0.5) {
            const size_t one_above_qi = qi_chain - 1;
            mod_switch_to_inplace(ctx, saved, one_above_qi);

            // Read the prime that the imminent rescale_to_next will drop.
            const auto &one_above_cd = ctx.get_context_data(one_above_qi);
            const uint64_t q_back_saved =
                one_above_cd.parms().coeff_modulus().back().value();

            // Read q_msg (the bottom prime, also saved's "scale prime").
            const size_t bottom_index = ctx.total_parm_size() - 1;
            const uint64_t q_msg =
                ctx.get_context_data(bottom_index).parms().coeff_modulus()[0].value();

            // Read the chain primes that define out.scale() exactly:
            //   non-split: out.scale() = first_back · user_scale / last_back
            //   split:     out.scale() = first_back   (S2C preserved scale)
            const size_t num_s2c_layers = bk.s2c.layers.size();
            const size_t s2c_last_chain = s2c_in_chain + num_s2c_layers - 1;
            const uint64_t first_back =
                ctx.get_context_data(s2c_in_chain).parms().coeff_modulus().back().value();
            const uint64_t last_back =
                ctx.get_context_data(s2c_last_chain).parms().coeff_modulus().back().value();

            // user_scale is a power-of-two double (e.g. 2^40); convert to u64.
            const uint64_t user_scale_u64 = static_cast<uint64_t>(user_scale);
            if (static_cast<double>(user_scale_u64) != user_scale) {
                throw std::invalid_argument(
                    "bootstrap: user_scale must be representable as u64");
            }

            // D_exact_align = (q_back_saved · target_scale) / q_msg,
            // computed in u128 (intermediate ≤ 2^126, fits).
            // - non-split target_scale = first_back · user_scale / last_back
            // - split     target_scale = first_back
            __uint128_t D_int128;
            if (split_scale_down) {
                // Split path: align saved.scale → first_back (= out.scale).
                // D = round(first_back · q_back_saved / q_msg)   (~2^58, fits in u64).
                D_int128 = ((__uint128_t)first_back * (__uint128_t)q_back_saved
                            + (__uint128_t)q_msg / 2) /
                           (__uint128_t)q_msg;
            } else {
                // Non-split: align saved.scale → user_scale (post-baked out).
                // D = (first_back · user_scale · q_back_saved) / (last_back · q_msg).
                // Two u128 steps to avoid 2^174 overflow:
                //   step1 = round(first_back · user_scale / q_msg)
                //   D     = round(step1 · q_back_saved / last_back)
                const __uint128_t step1_num =
                    (__uint128_t)first_back * (__uint128_t)user_scale_u64;
                const __uint128_t step1_den = (__uint128_t)q_msg;
                const __uint128_t step1 = (step1_num + step1_den / 2) / step1_den;
                D_int128 =
                    (step1 * (__uint128_t)q_back_saved + (__uint128_t)last_back / 2) /
                    (__uint128_t)last_back;
            }
            const uint64_t D_exact = static_cast<uint64_t>(D_int128);

            // Per-tower integer multiply by D_exact (mod each tower's prime),
            // followed by rescale_to_next. Bit-exact (no encoder FP).
            const auto &saved_cd = ctx.get_context_data(saved.chain_index());
            const size_t saved_num_towers =
                saved_cd.parms().coeff_modulus().size();
            std::vector<uint64_t> tower_scalars(saved_num_towers, D_exact);
            multiply_uint_scalars_per_tower(ctx, saved, tower_scalars);
            // Bookkeeping: scale becomes q_msg · D_exact (pre-rescale).
            saved.set_scale(saved.scale() * static_cast<double>(D_exact));
            rescale_to_next_inplace(ctx, saved);
            // After rescale: saved.scale = q_msg · D_exact / q_back_saved
            // ≈ out.scale() to <1 ulp. Snap to exact equality so sub_inplace
            // accepts the operands.
            saved.set_scale(out.scale());
        }

        if (saved.chain_index() != qi_chain) {
            mod_switch_to_inplace(ctx, saved, qi_chain);
        }
        // Snap saved.scale to out.scale so sub_inplace's are_close check passes.
        saved.set_scale(out.scale());

        sub_inplace(ctx, saved, out);

        // Post-bootstrap scale-down (split path only). At this point `saved`
        // holds the small residual `m` at scale ≈ first_back (s2c_in chain
        // prime, ~2^58). Bring it down to user_scale via one integer
        // multiply + rescale on the *small* residual rather than the large
        // (m + K·I) — this is what recovers the ~5 bits the_lib's
        // BootstrapTo14Levels compact mode regains by separating the
        // scale_down_ratio division out of the bootstrap pipeline.
        //
        // After multiply by D2 + rescale (drops q_drop_post at qi_chain):
        //   new_scale = saved.scale() · D2 / q_drop_post
        //             = user_scale  (when D2 = q_drop_post · user_scale / saved.scale).
        if (split_scale_down) {
            const uint64_t saved_scale_u64 = static_cast<uint64_t>(saved.scale());
            const auto &qi_cd = ctx.get_context_data(qi_chain);
            const uint64_t q_drop_post =
                qi_cd.parms().coeff_modulus().back().value();

            const uint64_t user_scale_u64 = static_cast<uint64_t>(user_scale);
            if (static_cast<double>(user_scale_u64) != user_scale) {
                throw std::invalid_argument(
                    "bootstrap: user_scale must be representable as u64");
            }

            // D2 = round(q_drop_post · user_scale / saved.scale)  (~user_scale).
            const __uint128_t D2_int128 =
                ((__uint128_t)q_drop_post * (__uint128_t)user_scale_u64
                 + (__uint128_t)saved_scale_u64 / 2) /
                (__uint128_t)saved_scale_u64;
            const uint64_t D2 = static_cast<uint64_t>(D2_int128);

            const auto &saved_cd = ctx.get_context_data(saved.chain_index());
            const size_t saved_num_towers =
                saved_cd.parms().coeff_modulus().size();
            std::vector<uint64_t> tower_scalars(saved_num_towers, D2);
            multiply_uint_scalars_per_tower(ctx, saved, tower_scalars);
            saved.set_scale(saved.scale() * static_cast<double>(D2));
            rescale_to_next_inplace(ctx, saved);
            // Snap to exact user_scale (drift ≪ 1 ulp at scale ~2^40).
            saved.set_scale(user_scale);
        }

        return saved;
    }

    // PROBE-ONLY: see declaration in include/bootstrap.h.
    PhantomCiphertext
    probe_plaintext_storage_mul_rescale(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomCiphertext &ct_in,
            const std::vector<std::complex<double>> &vals,
            std::size_t chain_index,
            double scale,
            int mode) {
        PhantomCiphertext ct = ct_in;  // deep copy
        if (ct.chain_index() != chain_index) {
            mod_switch_to_inplace(ctx, ct, chain_index);
        }

        std::vector<C64> vals_c64(vals.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            vals_c64[i] = C64(vals[i].real(), vals[i].imag());
        }

        if (mode == 0) {
            // light path: tower-0 int64 storage, expand-on-demand.
            LightPlaintext light = encode_to_light_plaintext(
                    ctx, encoder, vals_c64, chain_index, scale);
            PhantomPlaintext expanded = expand_light_plaintext(ctx, light);
            multiply_plain_inplace(ctx, ct, expanded);
            rescale_to_next_inplace(ctx, ct);
            return ct;
        } else {
            // full path: full-RNS NTT-form plaintext encoded directly by CKKS encoder.
            PhantomPlaintext full = encode_complex_diagonal(
                    ctx, encoder, vals_c64, chain_index, scale);
            multiply_plain_inplace(ctx, ct, full);
            rescale_to_next_inplace(ctx, ct);
            return ct;
        }
    }

} // namespace phantom
