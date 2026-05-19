#include "evalmod.h"
#include "evaluate.cuh"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef EVALMOD_STAGE_DEBUG
#include "secretkey.h"
#include <cstdio>
#include <functional>
#include <mutex>
#endif

namespace phantom {

#ifdef EVALMOD_STAGE_DEBUG
// ----------------------------------------------------------------------------
// Stage-probe state (set by the test/probe before calling evalmod_k28_r3).
// Holds the secret key + the analytical reference inputs (one value per slot
// in the first N_PROBE slots). The instrumentation inside sine_chebyshev_k28
// decrypts a clone of each intermediate ciphertext, decodes, computes the
// analytical reference at the same input, and prints
//   bits = -log2(mean_abs_err)  over the first N_PROBE slots.
// ----------------------------------------------------------------------------
namespace {
struct StageProbeState {
    const PhantomSecretKey *sk = nullptr;
    std::vector<double> vals;  // input values at slots [0..N_PROBE-1]
};
StageProbeState g_probe_state;
std::mutex g_probe_mutex;
}  // namespace

void evalmod_set_stage_probe(const PhantomSecretKey *sk,
                             const std::vector<double> &vals) {
    std::lock_guard<std::mutex> lk(g_probe_mutex);
    g_probe_state.sk = sk;
    g_probe_state.vals = vals;
}

void evalmod_clear_stage_probe() {
    std::lock_guard<std::mutex> lk(g_probe_mutex);
    g_probe_state.sk = nullptr;
    g_probe_state.vals.clear();
}
#endif  // EVALMOD_STAGE_DEBUG

// ============================================================================
// Chebyshev coefficients for K=7 sine (lapis evalmod_coeffs.rs).
//   target: cos(7πx/4 - π/16) · (2π)^{-1/8}
//   PS structure: p(x) = low(T_0..T_8) + 2·T_8·high(T_1..T_7)
//   low[0]  = c_0,  low[i] = c_i - c_{16-i} for i=1..7,  low[8] = c_8
//   high[0] = 0 (T_0 unused),  high[j] = c_{8+j} for j=1..7
//   After 3 double-angle iterations → sin(2π·7·x)/(2π).
// ============================================================================
namespace {

constexpr std::array<double, 9> K7_LOW_COEFFS = {
    -5.92365525763352552e-03,   // c_0
    -1.05915002790352347e-01,   // c_1 - c_15
     1.81872438169629558e-01,   // c_2 - c_14
    -7.96115559593863209e-02,   // c_3 - c_13
     6.18237166588616383e-01,   // c_4 - c_12
     9.97204899504430670e-02,   // c_5 - c_11
    -2.85567102345358925e-01,   // c_6 - c_10
    -3.02985739088491718e-02,   // c_7 - c_9
     5.23393518416412837e-02,   // c_8
};

constexpr std::array<double, 8> K7_HIGH_COEFFS = {
     0.0,                       // T_0 (unused — always zero by construction)
     3.49656797384041707e-03,   // c_9
    -5.21322493828673197e-03,   // c_10
    -2.75767647883437550e-04,   // c_11
     3.34517473917776004e-04,   // c_12
     1.47041114332627232e-05,   // c_13
    -1.50752826718614230e-05,   // c_14
    -5.67939380614890243e-07,   // c_15
};

// K=16 R=3 degree-31 Chebyshev (lapis evalmod_coeffs.rs).
//   Target: (2π)^{-1/8} · cos(4πx − π/16); after R=3 DA → sin(2π·16·x)/(2π).
//   PS structure: P(x) = T_16·(T_8·Q_high + Q_low) + (T_8·R_high + R_low)
//   where Q_high, Q_low, R_high, R_low are weighted sums of T_1..T_7
//   (8-element arrays with index 0 holding the T_1 coefficient — there is no
//   constant T_0 term in this factoring).
//   Polynomial precision: 33.9 bits.
constexpr std::array<double, 8> K16_Q_HIGH = {
     1.40933240904887619e-05,  1.50639230276274228e-06,
    -1.94594040092296340e-06, -9.53210705782742233e-08,
     1.13318833226784231e-07,  5.12709912705527410e-09,
    -5.64857600334760771e-09, -2.37565221374504097e-10,
};
constexpr std::array<double, 8> K16_Q_LOW = {
     3.59519354992413262e-02,  6.20731655651990660e-03,
    -1.25289184971230146e-02, -9.32191173092421330e-04,
     1.64251931763363971e-03,  1.07868728320514593e-04,
    -1.68130129335233066e-04, -1.07079565087396969e-05,
};
constexpr std::array<double, 8> K16_R_HIGH = {
    -1.03480520058724892e-01,  9.00051357854676909e-02,
    -8.54845145538975948e-01, -1.80792080525974813e-01,
     7.33614949409451600e-01,  9.95683206882135990e-02,
    -2.78190804126826174e-01, -3.64211934038645019e-02,
};
constexpr std::array<double, 8> K16_R_LOW = {
     1.22772867008426184e-01, -2.97084681220240107e-02,
     4.22982423883966452e-01, -7.97287003613950523e-02,
    -1.10422768440682217e-02,  1.05502625978328088e-01,
     7.22752525999589279e-01,  2.62006578429296658e-02,
};

// K=28 R=3 degree-49 Chebyshev (ported from the_lib bootstrap.cpp).
//   Structure: baby T_1..T_7, giant T_14/T_28, PS basis T_49.
//   sine(x) = (aux_ct · quotient + remainder − T_49), then R=3 DA.
//   aux_ct = ws(baby, aux_coeffs) + T_28
//   quotient = (T_14 + ws(baby, q_coeffs[0])) · ws(baby, q_coeffs[1]) + ws(baby, q_coeffs[2])
//   remainder = same shape, no relin
//   All ws arrays: coeffs[0]=constant, coeffs[1..7]=T_1..T_7 coeff.
//   Total levels: 3 (baby) + 3 (T_14,T_28,T_49) + 3 (PS) = 9. R=3 DA = +3. Total = 9.
constexpr std::array<double, 8> K28_AUX_COEFFS = {
    -0.4674215360340989,   -0.24807246365084598,
    -0.03360736343811264,   0.10485552200765141,
     0.012171628025753911, -0.031034212401993305,
    -0.003985828764410408,  0.0,
};
constexpr std::array<std::array<double, 8>, 3> K28_QUOT_COEFFS = {{
    {-0.7500012154934053,      4.148627473785995e-06,   2.707983311367457e-07,
     -4.324538856989888e-07,  -2.648274421485692e-08,   3.9515438249932484e-08,
      2.4720260690299275e-09,  0.0},
    {-6.511205244110998e-09,  -7.076428529570776e-10,   1.0183620860162692e-09,
      0.0,                     0.0,                      0.0,
      0.0,                     4.0},
    { 0.007868944223231958,    0.0014193060148561195,   -0.003081157124604844,
     -0.00025275745577540515,  0.0005030348400875631,    3.734766164626263e-05,
     -7.780382338014487e-05,   1.0},
}};
constexpr std::array<std::array<double, 8>, 3> K28_REM_COEFFS = {{
    {-1.1324455480972702,  -0.8116709016985189,  -0.20294197180913795,
      0.5290215620326779,  -0.32084902738708476,  0.43458317383580536,
     -0.06303880005650665,  0.0},
    { 0.23316927298939902,  0.12938749185636858,  -0.3042902283756468,
     -0.008967445477730096, -0.6955791340472846,  -0.07562064123442451,
      1.2510854562849574,   2.0},
    { 0.1429148264191405,   0.4544397978460022,   -0.8401926425765848,
     -0.4292803758141986,   -1.495563076527614,    0.3580084476476745,
      1.7971594278092713,   1.0},
}};

constexpr int  EVALMOD_R       = 3;
constexpr int  CHEB_MAX_POWER  = 8;
constexpr int  CHEB_BABY_K28   = 7;

#ifdef EVALMOD_STAGE_DEBUG
// Reference function signature: takes input x ∈ ℝ, returns analytical ref value.
// Used by report_stage to score the decrypted ct against the cleartext stage value.
using StageRefFn = std::function<double(double)>;

// Helper: relinearize a TRIO copy so we can decrypt. No effect on original.
void probe_relinearize_if_needed(const PhantomContext &ctx,
                                 PhantomCiphertext &c,
                                 const PhantomRelinKey &rk) {
    if (c.size() > 2) {
        relinearize_inplace(ctx, c, rk);
    }
}

// Decrypt + decode `ct` (cloning so original is untouched), compute the
// analytical stage value at each input slot via `ref_fn`, then print
// stage name + bits-of-precision (avg over N_PROBE slots, max over same).
void report_stage(const PhantomContext &ctx,
                  PhantomCKKSEncoder &enc,
                  const PhantomCiphertext &ct,
                  const PhantomRelinKey &rk,
                  const char *stage_name,
                  const StageRefFn &ref_fn) {
    std::lock_guard<std::mutex> lk(g_probe_mutex);
    if (g_probe_state.sk == nullptr || g_probe_state.vals.empty()) return;
    const auto &vals = g_probe_state.vals;
    const size_t n = vals.size();

    PhantomCiphertext clone = ct;
    probe_relinearize_if_needed(ctx, clone, rk);

    PhantomPlaintext dec_pt;
    // decrypt() is non-const in the API, but we treat our captured pointer as
    // logically const (we never mutate it) — cast away const at the call site.
    const_cast<PhantomSecretKey *>(g_probe_state.sk)->decrypt(ctx, clone, dec_pt);
    std::vector<double> decoded;
    enc.decode(ctx, dec_pt, decoded);

    double sum_abs = 0.0;
    double max_abs = 0.0;
    double ref0 = 0.0, dec0 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double ref = ref_fn(vals[i]);
        const double err = std::abs(decoded[i] - ref);
        sum_abs += err;
        if (err > max_abs) max_abs = err;
        if (i == 0) { ref0 = ref; dec0 = decoded[0]; }
    }
    const double avg = sum_abs / double(n);
    const double bits_avg = (avg > 0.0) ? -std::log2(avg) : 99.0;
    const double bits_max = (max_abs > 0.0) ? -std::log2(max_abs) : 99.0;
    std::fprintf(stderr,
        "[STAGE %-22s] bits(avg)=%6.2f  bits(max)=%6.2f  avg_err=%.3e  max_err=%.3e  "
        "dec[0]=%+.6e ref[0]=%+.6e  chain=%zu  size=%zu\n",
        stage_name, bits_avg, bits_max, avg, max_abs,
        dec0, ref0, clone.chain_index(), clone.size());
    std::fflush(stderr);
}

// Chebyshev T_k(x) via cos(k·acos(x)) — works for |x| ≤ 1.
double cheb_T(int k, double x) {
    // For |x|<1 (our case after vals/(4K)), cos(k·acos(x)) is the standard form.
    if (x > 1.0) x = 1.0;
    if (x < -1.0) x = -1.0;
    return std::cos(double(k) * std::acos(x));
}

// Weighted-sum reference: coeffs[0] + Σ_{i≥1} coeffs[i] · T_i(x).
double ref_weighted_sum(const double *coeffs, size_t n_coeffs, double x) {
    double s = coeffs[0];
    for (size_t i = 1; i < n_coeffs; ++i) s += coeffs[i] * cheb_T(int(i), x);
    return s;
}
#endif  // EVALMOD_STAGE_DEBUG

// ----------------------------------------------------------------------------
// Tiny helpers wrapping phantom's public ops.
// ----------------------------------------------------------------------------

// Drop a ciphertext to the same chain_index as `target` if it's currently
// fresher (i.e. lower chain_index in phantom's convention where chain_index
// increases with consumption). No-op if already aligned or older.
void align_to(const PhantomContext &ctx, PhantomCiphertext &ct, size_t target_idx) {
    if (ct.chain_index() < target_idx) {
        mod_switch_to_inplace(ctx, ct, target_idx);
    }
}

// Snap a ct's scale to a fixed target. CKKS rescale produces scale = old²/q_i
// where q_i is close to but not exactly 2^Δ, so consecutive ct's end up with
// epsilon-different scales that fail phantom's strict are_close check. Snapping
// to a single canonical target keeps add_inplace/multiply_plain happy without
// changing the underlying ciphertext data — the snap is purely metadata.

#ifdef SNAP_DEBUG
#include <cstdio>
#include <cmath>
static int snap_counter = 0;
static double snap_drift_sum_bits = 0.0;
static double snap_drift_max_bits = 0.0;
#endif

void snap_scale(PhantomCiphertext &ct, double target_scale) {
#ifdef SNAP_DEBUG
    double actual = ct.scale();
    double drift_bits = std::log2(actual / target_scale);
    snap_counter++;
    snap_drift_sum_bits += std::abs(drift_bits);
    if (std::abs(drift_bits) > snap_drift_max_bits)
        snap_drift_max_bits = std::abs(drift_bits);
    std::fprintf(stderr,
        "[SNAP #%02d] chain=%zu  actual=2^%.6f  target=2^%.6f  drift=%+.6f bits\n",
        snap_counter, ct.chain_index(),
        std::log2(actual), std::log2(target_scale), drift_bits);
    std::fflush(stderr);
#endif
    ct.set_scale(target_scale);
}

#ifdef SNAP_DEBUG
// Call once after evalmod_k28_r3 returns to print summary.
static void snap_debug_summary() {
    std::fprintf(stderr,
        "[SNAP_SUMMARY] total=%d  sum_drift=%.6f bits  max_drift=%.6f bits\n",
        snap_counter, snap_drift_sum_bits, snap_drift_max_bits);
    std::fflush(stderr);
}
#endif

void align_pair(const PhantomContext &ctx, PhantomCiphertext &a, PhantomCiphertext &b) {
    const size_t target = std::max(a.chain_index(), b.chain_index());
    align_to(ctx, a, target);
    align_to(ctx, b, target);
}

// ct ← 2·ct  (no level cost).
void double_ct(const PhantomContext &ctx, PhantomCiphertext &ct) {
    PhantomCiphertext tmp = ct;
    add_inplace(ctx, ct, tmp);
}

// Return the actual chain prime (back modulus) at a given chain index.
// This is the correct encode scale for a plaintext that will be multiplied
// against a ciphertext at that chain index then rescaled: using the chain
// prime ensures ct.scale × pt.scale / q_i = ct.scale (no drift).
// On a uniform 58-bit chain: chain_prime_at ≈ 2^58 ≈ target_scale, so this
// is a no-op there. On a 54-bit ER chain: chain_prime ≈ 2^54, which is
// different from target_scale (2^58); using chain_prime keeps rescale honest.
static double chain_prime_at(const PhantomContext &ctx, size_t chain_idx) {
    const auto &mods = ctx.get_context_data(chain_idx).parms().coeff_modulus();
    return static_cast<double>(mods.back().value());
}

// Encode `value` at the same scale as `ref` (ref.scale()), then add to ref.
// add_plain_inplace requires ct.scale == pt.scale strictly, so we match the
// ciphertext metadata scale (which is target_scale after snap). This is a
// level-free scalar add: no rescale follows, so no prime-alignment needed.
// Used for the c_0 constant term and the DA scalar.
// Phantom's CKKS encoder is vector-only, so we replicate the scalar across slots.
void add_scalar_inplace(const PhantomContext &ctx,
                        PhantomCKKSEncoder &enc,
                        PhantomCiphertext &ref,
                        double value) {
    std::vector<double> rep(enc.slot_count(), value);
    PhantomPlaintext pt;
    enc.encode(ctx, rep, ref.scale(), pt, ref.chain_index());
    add_plain_inplace(ctx, ref, pt);
}

// ct ← 2·ct + scalar  (level-free; used inside double-angle).
void rescale_double_add(const PhantomContext &ctx,
                        PhantomCKKSEncoder &enc,
                        PhantomCiphertext &ct,
                        double scalar,
                        double target_scale) {
    rescale_to_next_inplace(ctx, ct);
    snap_scale(ct, target_scale);
    double_ct(ctx, ct);
    add_scalar_inplace(ctx, enc, ct, scalar);
}

// ct ← ct² ; relin
PhantomCiphertext square_and_relin(const PhantomContext &ctx,
                                   const PhantomCiphertext &ct,
                                   const PhantomRelinKey &rk) {
    PhantomCiphertext sq = ct;
    multiply_and_relin_inplace(ctx, sq, ct, rk);
    return sq;
}

// ct ← a × b ; relin.   `a` and `b` must be at the same chain_index/scale.
PhantomCiphertext mul_and_relin(const PhantomContext &ctx,
                                const PhantomCiphertext &a,
                                const PhantomCiphertext &b,
                                const PhantomRelinKey &rk) {
    PhantomCiphertext dst = a;
    multiply_and_relin_inplace(ctx, dst, b, rk);
    return dst;
}

// ct ← a × b ; NO relin. Result is a TRIO (3-component when both inputs DUO,
// or larger when inputs are TRIO). Used for Paterson-Stockmeyer deferred-relin.
// `a` and `b` must be at the same chain_index and scale.
PhantomCiphertext mul_no_relin(const PhantomContext &ctx,
                               const PhantomCiphertext &a,
                               const PhantomCiphertext &b) {
    PhantomCiphertext dst = a;
    multiply_inplace(ctx, dst, b);
    return dst;
}

// ----------------------------------------------------------------------------
// build_chebyshev_basis: returns T_1..T_max_power (1-indexed → result[i-1]).
//
// Recurrences:
//   T_{2n}   = 2 T_n² − 1                  (power-of-2 — uses rescale_add_double(-0.5))
//   T_{a+b}  = 2 T_a T_b − T_{|a−b|}       (general — uses hmult + rescale + double − T_{|a−b|})
//
// The level of T_n grows with n: powers of 2 each consume +1 level. After this
// helper, all returned T_i share the same chain_index (aligned to T_max_power).
// ----------------------------------------------------------------------------
std::vector<PhantomCiphertext>
build_chebyshev_basis(const PhantomContext &ctx,
                      PhantomCKKSEncoder &enc,
                      const PhantomCiphertext &ct,
                      int max_power,
                      const PhantomRelinKey &rk,
                      double target_scale) {
    std::vector<PhantomCiphertext> basis(max_power + 1);
    bool have[max_power + 1] = {false};
    basis[1] = ct;
    snap_scale(basis[1], target_scale);
    have[1] = true;

    for (int n = 2; n <= max_power; ++n) {
        if ((n & (n - 1)) == 0) {
            // Power of 2: T_{2k} = 2·T_k² − 1
            const int half = n >> 1;
            PhantomCiphertext sq = square_and_relin(ctx, basis[half], rk);
            rescale_to_next_inplace(ctx, sq);
            snap_scale(sq, target_scale);
            // sq ← 2·(sq − 0.5) = 2·sq − 1
            add_scalar_inplace(ctx, enc, sq, -0.5);
            double_ct(ctx, sq);
            basis[n] = std::move(sq);
            have[n] = true;
        } else {
            // General: T_{a+b} = 2·T_a·T_b − T_{|a−b|}, with a = highest pow2 ≤ n.
            const int hp2 = 1 << (31 - __builtin_clz(n));
            const int mp  = n - hp2;
            const int sp  = 2 * hp2 - n;

            PhantomCiphertext a = basis[hp2];
            PhantomCiphertext b = basis[mp];
            align_pair(ctx, a, b);

            PhantomCiphertext prod = mul_and_relin(ctx, a, b, rk);
            rescale_to_next_inplace(ctx, prod);
            snap_scale(prod, target_scale);
            double_ct(ctx, prod);

            PhantomCiphertext s = basis[sp];
            align_pair(ctx, prod, s);
            sub_inplace(ctx, prod, s);

            basis[n] = std::move(prod);
            have[n] = true;
        }

        // After completing a power of 2 (or the last index), align lower powers
        // up so all T_i sharing this build phase end up at the same level.
        const bool is_pow2 = (n & (n - 1)) == 0;
        const bool is_last = (n == max_power);
        if (is_pow2 || is_last) {
            const int aligned_point = is_pow2 ? n / 2
                                              : (1 << (31 - __builtin_clz(n)));
            const size_t target_idx = basis[n].chain_index();
            for (int p = aligned_point; p >= 1; --p) {
                if (have[p]) align_to(ctx, basis[p], target_idx);
            }
        }
    }

    // Drop T_0 placeholder.
    std::vector<PhantomCiphertext> out;
    out.reserve(max_power);
    for (int i = 1; i <= max_power; ++i) out.push_back(std::move(basis[i]));
    return out;
}

// ----------------------------------------------------------------------------
// weighted_sum: result = coeffs[0] + Σ_{i ≥ 1} coeffs[i] · basis[i−1]
//
// Lazy-rescale strategy (matches lapis weighted_sum_chebyshev_pre_encoded):
//   - encode each coeffs[i] (i ≥ 1) at basis[0].scale, then multiply_plain
//   - sum all products at scale = baby_scale²  (same chain_index, same scale)
//   - one rescale brings the result back to baby_scale at baby_index+1
//   - finally add coeffs[0] (level-free)
//
// Coefficients with |c| < 1e-50 are skipped so the placeholder K7_HIGH[0]=0
// doesn't generate a useless multiply.
// Consumes 1 level total.
// ----------------------------------------------------------------------------
PhantomCiphertext
weighted_sum_chebyshev(const PhantomContext &ctx,
                       PhantomCKKSEncoder &enc,
                       const std::vector<PhantomCiphertext> &basis,
                       const double *coeffs,
                       size_t n_coeffs,
                       double target_scale) {
    if (n_coeffs == 0)
        throw std::invalid_argument("weighted_sum_chebyshev: empty coeffs");

    const double baby_scale = basis[0].scale();  // post-snap, == target_scale
    const size_t baby_idx   = basis[0].chain_index();
    // Encode plaintext coefficients at the chain prime at baby_idx rather than
    // baby_scale. After multiply_plain_inplace + rescale_to_next_inplace:
    //   result.scale = ct.scale × chain_prime / chain_prime = ct.scale  (exact)
    // On a uniform prime chain chain_prime ≈ baby_scale (no-op in practice).
    // On a 54-bit ER chain, chain_prime ≈ 2^54 while baby_scale (metadata) is
    // 2^58; using baby_scale here caused 2^4 drift per rescale → exponential
    // blow-up across the 9-level EvalMod pipeline.
    const double encode_scale = chain_prime_at(ctx, baby_idx);

    bool first_term = true;
    PhantomCiphertext acc;

    const size_t slots = enc.slot_count();
    for (size_t i = 1; i < n_coeffs; ++i) {
        if (std::abs(coeffs[i]) < 1e-50) continue;

        std::vector<double> rep(slots, coeffs[i]);
        PhantomPlaintext pt;
        enc.encode(ctx, rep, encode_scale, pt, baby_idx);

        PhantomCiphertext term = basis[i - 1];
        multiply_plain_inplace(ctx, term, pt);

        if (first_term) {
            acc = std::move(term);
            first_term = false;
        } else {
            add_inplace(ctx, acc, term);
        }
    }

    if (first_term) {
        throw std::invalid_argument("weighted_sum_chebyshev: all coefficients zero");
    }

    // Single deferred rescale collapses K levels → 1.
    rescale_to_next_inplace(ctx, acc);
    snap_scale(acc, target_scale);

    if (std::abs(coeffs[0]) >= 1e-50) {
        add_scalar_inplace(ctx, enc, acc, coeffs[0]);
    }
    return acc;
}

// ----------------------------------------------------------------------------
// sine_chebyshev_k7: 5-level evaluation of the K=7 polynomial.
//   result = low + 2·T_8·high
//   low  = ws(K7_LOW,  T_1..T_8)              -- 1 level
//   high = ws(K7_HIGH, T_1..T_7)              -- 1 level (parallel)
//   T_8 · high → rescale → double             -- 1 level
//   add low                                    -- 0
// + build_chebyshev_basis(8) = 3 levels       -- prior
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// sine_chebyshev_k16: 6-level evaluation of the K=16 polynomial.
//   P(x) = T_16 · (T_8·Q_high + Q_low) + (T_8·R_high + R_low)
// where the inner sums are weighted_sum over T_1..T_7.
//
// Levels (relative to baby_level after build_chebyshev_basis(8) = 3 levels):
//   T_16 = 2·T_8² − 1                                 baby+1
//   ws Q_high / Q_low / R_high / R_low                baby+1 (parallel)
//   T_8 · Q_high (align, hmult+rescale)               baby+2
//   Q = T_8·Q_high + Q_low                            baby+2 (Q_low aligned up)
//   T_8 · R_high                                      baby+2
//   R = T_8·R_high + R_low                            baby+2
//   T_16 · Q (align T_16 up, hmult+rescale)           baby+3
//   result = T_16·Q + R                               baby+3 (R aligned up)
// → 6 levels total for sine. With R=3 DA, EvalMod = 9 levels.
// ----------------------------------------------------------------------------
// Generic degree-31 sine evaluator with PS factorization
//   P(x) = T_16 · (T_8·Q_high + Q_low) + (T_8·R_high + R_low)
// Used by K=16 R=3 (K=28 R=3 has its own degree-49 PS structure in sine_chebyshev_k28).
PhantomCiphertext
sine_chebyshev_deg31(const PhantomContext &ctx,
                     PhantomCKKSEncoder &enc,
                     const PhantomCiphertext &ct,
                     const PhantomRelinKey &rk,
                     double target_scale,
                     const double *q_high_coeffs,
                     const double *q_low_coeffs,
                     const double *r_high_coeffs,
                     const double *r_low_coeffs) {
    auto baby = build_chebyshev_basis(ctx, enc, ct, CHEB_MAX_POWER, rk, target_scale);

    // T_16 = 2·T_8² − 1.
    PhantomCiphertext t16 = square_and_relin(ctx, baby[7], rk);
    rescale_to_next_inplace(ctx, t16);
    snap_scale(t16, target_scale);
    add_scalar_inplace(ctx, enc, t16, -0.5);
    double_ct(ctx, t16);

    // Inner weighted sums over T_1..T_7 (parallel; each consumes 1 level).
    // Lapis convention: coeffs[0] is the constant term, coeffs[i] (i≥1) is T_i.
    std::vector<PhantomCiphertext> baby7(baby.begin(), baby.begin() + 7);
    auto q_high = weighted_sum_chebyshev(ctx, enc, baby7, q_high_coeffs, 8, target_scale);
    auto q_low  = weighted_sum_chebyshev(ctx, enc, baby7, q_low_coeffs,  8, target_scale);
    auto r_high = weighted_sum_chebyshev(ctx, enc, baby7, r_high_coeffs, 8, target_scale);
    auto r_low  = weighted_sum_chebyshev(ctx, enc, baby7, r_low_coeffs,  8, target_scale);

    // T_8 · Q_high → rescale.
    PhantomCiphertext t8a = baby[7];
    align_pair(ctx, t8a, q_high);
    PhantomCiphertext t8_qh = mul_and_relin(ctx, t8a, q_high, rk);
    rescale_to_next_inplace(ctx, t8_qh);
    snap_scale(t8_qh, target_scale);

    // Q = T_8·Q_high + Q_low.
    align_pair(ctx, t8_qh, q_low);
    add_inplace(ctx, t8_qh, q_low);
    PhantomCiphertext Q = std::move(t8_qh);

    // T_8 · R_high → rescale.
    PhantomCiphertext t8b = baby[7];
    align_pair(ctx, t8b, r_high);
    PhantomCiphertext t8_rh = mul_and_relin(ctx, t8b, r_high, rk);
    rescale_to_next_inplace(ctx, t8_rh);
    snap_scale(t8_rh, target_scale);

    // R = T_8·R_high + R_low.
    align_pair(ctx, t8_rh, r_low);
    add_inplace(ctx, t8_rh, r_low);
    PhantomCiphertext R = std::move(t8_rh);

    // T_16 · Q → rescale.
    align_pair(ctx, t16, Q);
    PhantomCiphertext t16_q = mul_and_relin(ctx, t16, Q, rk);
    rescale_to_next_inplace(ctx, t16_q);
    snap_scale(t16_q, target_scale);

    // result = T_16·Q + R.
    align_pair(ctx, t16_q, R);
    add_inplace(ctx, t16_q, R);
    return t16_q;
}

PhantomCiphertext
sine_chebyshev_k7(const PhantomContext &ctx,
                  PhantomCKKSEncoder &enc,
                  const PhantomCiphertext &ct,
                  const PhantomRelinKey &rk,
                  double target_scale) {
    auto baby = build_chebyshev_basis(ctx, enc, ct, CHEB_MAX_POWER, rk, target_scale);

    // Low part uses T_1..T_8 (slice = baby).
    auto low  = weighted_sum_chebyshev(ctx, enc, baby,
                                       K7_LOW_COEFFS.data(), K7_LOW_COEFFS.size(),
                                       target_scale);

    // High part uses T_1..T_7 only (drop T_8).
    std::vector<PhantomCiphertext> baby_for_high(baby.begin(), baby.begin() + 7);
    auto high = weighted_sum_chebyshev(ctx, enc, baby_for_high,
                                       K7_HIGH_COEFFS.data(), K7_HIGH_COEFFS.size(),
                                       target_scale);

    // T_8 × high → align levels → hmult → rescale → double.
    PhantomCiphertext t8 = baby[7];
    align_pair(ctx, t8, high);
    PhantomCiphertext prod = mul_and_relin(ctx, t8, high, rk);
    rescale_to_next_inplace(ctx, prod);
    snap_scale(prod, target_scale);
    double_ct(ctx, prod);

    // result = low + prod
    align_pair(ctx, prod, low);
    add_inplace(ctx, prod, low);
    return prod;
}

// ----------------------------------------------------------------------------
// sine_chebyshev_k28: degree-49 PS evaluation (ported from the_lib sine()).
//
// Level analysis (depth d = baby chain_index after build_chebyshev_basis(7)):
//   baby  T_1..T_7        : depth d        (3 levels consumed)
//   T_14  = 2·T_7² − 1    : depth d+1
//   ws_*  (aux/Q[i]/R[i]) : depth d+1      (1 rescale; baby NOT pre-aligned)
//   T_28  = 2·T_14² − 1   : depth d+2
//   T_21  = 2·T_14·T_7−T_7: depth d+2      (T_14·T_7 at d+1, rescale +1)
//   T_49  = 2·T_28·T_21−T7: depth d+3
//   aux_ct = ws_aux + T_28 : depth d+2     (align ws_aux down)
//   quotient = (T_14 + Q[0])·Q[1] + Q[2]   : depth d+2  (no level cost
//                                             on the add since T_14, Q[0]
//                                             both at d+1)
//   remainder= (T_14 + R[0])·R[1] + R[2]   : depth d+2
//   result   = aux_ct·quotient + remainder − T_49 : depth d+3
//
// Total sine levels: 3 (baby) + 3 (above) = 6 levels.
// With R=3 DA: total EvalMod = 9 levels (same as K=16 R=3 chain budget).
// ----------------------------------------------------------------------------
PhantomCiphertext
sine_chebyshev_k28(const PhantomContext &ctx,
                   PhantomCKKSEncoder &enc,
                   const PhantomCiphertext &ct,
                   const PhantomRelinKey &rk,
                   double target_scale) {
    // Baby: T_1..T_7  (size 7 vector, baby[i] = T_{i+1}).
    auto baby = build_chebyshev_basis(ctx, enc, ct, CHEB_BABY_K28, rk, target_scale);
    PhantomCiphertext &t7 = baby[6];

#ifdef EVALMOD_STAGE_DEBUG
    // Stage 1: baby basis T_1..T_7
    for (int k = 1; k <= CHEB_BABY_K28; ++k) {
        char name[32];
        std::snprintf(name, sizeof(name), "basis[%d]=T_%d", k, k);
        const int kk = k;
        report_stage(ctx, enc, baby[k - 1], rk, name,
                     [kk](double x) { return cheb_T(kk, x); });
    }
#endif

    // T_14 = 2·T_7² − 1   (depth d+1).
    PhantomCiphertext t14 = square_and_relin(ctx, t7, rk);
    rescale_to_next_inplace(ctx, t14);
    snap_scale(t14, target_scale);
    add_scalar_inplace(ctx, enc, t14, -0.5);
    double_ct(ctx, t14);

#ifdef EVALMOD_STAGE_DEBUG
    report_stage(ctx, enc, t14, rk, "T_14",
                 [](double x) { return cheb_T(14, x); });
#endif

    // T_28 = 2·T_14² − 1   (depth d+2).
    PhantomCiphertext t28 = square_and_relin(ctx, t14, rk);
    rescale_to_next_inplace(ctx, t28);
    snap_scale(t28, target_scale);
    add_scalar_inplace(ctx, enc, t28, -0.5);
    double_ct(ctx, t28);

#ifdef EVALMOD_STAGE_DEBUG
    report_stage(ctx, enc, t28, rk, "T_28",
                 [](double x) { return cheb_T(28, x); });
#endif

    // Weighted sums over baby (depth d) → output at d+1. Critically,
    // we do NOT pre-align baby up to T_14 level; we let ws's internal
    // rescale collapse one baby level → output naturally at T_14's level.
    auto ws_aux = weighted_sum_chebyshev(ctx, enc, baby,
                                         K28_AUX_COEFFS.data(),
                                         K28_AUX_COEFFS.size(), target_scale);
    auto ws_q0  = weighted_sum_chebyshev(ctx, enc, baby,
                                         K28_QUOT_COEFFS[0].data(), 8, target_scale);
    auto ws_q1  = weighted_sum_chebyshev(ctx, enc, baby,
                                         K28_QUOT_COEFFS[1].data(), 8, target_scale);
    auto ws_q2  = weighted_sum_chebyshev(ctx, enc, baby,
                                         K28_QUOT_COEFFS[2].data(), 8, target_scale);
    auto ws_r0  = weighted_sum_chebyshev(ctx, enc, baby,
                                         K28_REM_COEFFS[0].data(),  8, target_scale);
    auto ws_r1  = weighted_sum_chebyshev(ctx, enc, baby,
                                         K28_REM_COEFFS[1].data(),  8, target_scale);
    auto ws_r2  = weighted_sum_chebyshev(ctx, enc, baby,
                                         K28_REM_COEFFS[2].data(),  8, target_scale);

    // T_21 = 2·T_14·T_7 − T_7  (depth d+2). Eager relin (T_14·T_7 → DUO).
    // The_lib pattern: this is `auxiliary_basis` in make_chebyshev_paterson_
    // stockmeyer_basis — built with `multiply(..., rk)` → DUO. We keep it DUO
    // here because we'll multiply it by T_28 next without relin (TRIO carry).
    PhantomCiphertext t14_for_t21 = t14;       // copy, will be reused
    PhantomCiphertext t7_for_t21  = t7;         // copy, baby[6] preserved
    align_pair(ctx, t14_for_t21, t7_for_t21);  // both at d+1
    PhantomCiphertext t21 = mul_and_relin(ctx, t14_for_t21, t7_for_t21, rk);
    rescale_to_next_inplace(ctx, t21);          // d+2
    snap_scale(t21, target_scale);
    double_ct(ctx, t21);                        // 2·T_14·T_7
    {
        PhantomCiphertext t7_a = t7;
        align_to(ctx, t7_a, t21.chain_index()); // d+2
        sub_inplace(ctx, t21, t7_a);            // − T_7
    }

#ifdef EVALMOD_STAGE_DEBUG
    report_stage(ctx, enc, t21, rk, "T_21",
                 [](double x) { return cheb_T(21, x); });
#endif

    // T_49 = 2·T_28·T_21 − T_7  (depth d+3). DEFERRED relin: this matches
    // the_lib's `paterson_stockmeyer_basis` (multiply WITHOUT rk → TRIO).
    // After this, t49 is a 3-component ciphertext; later we'll subtract it
    // from `result` (also TRIO) so size matches.
    PhantomCiphertext t28_for_t49 = t28;
    align_pair(ctx, t28_for_t49, t21);          // both at d+2
    PhantomCiphertext t49 = mul_no_relin(ctx, t28_for_t49, t21);  // TRIO
    rescale_to_next_inplace(ctx, t49);          // d+3
    snap_scale(t49, target_scale);
    double_ct(ctx, t49);
    {
        // t49 is TRIO (size 3), T_7 is DUO (size 2). The relaxed sub_inplace
        // handles size mismatch via overlap-then-tail (TRIO − DUO = TRIO,
        // since the missing T_7 component c2 is implicitly zero).
        PhantomCiphertext t7_b = t7;
        align_to(ctx, t7_b, t49.chain_index()); // d+3
        sub_inplace(ctx, t49, t7_b);            // − T_7  (TRIO − DUO)
    }

#ifdef EVALMOD_STAGE_DEBUG
    report_stage(ctx, enc, t49, rk, "T_49",
                 [](double x) { return cheb_T(49, x); });
#endif

    // aux_ct = ws_aux + T_28   (depth d+2). ws_aux at d+1 → align down.
    // Both DUO (eager-relin upstream). Stays DUO.
    PhantomCiphertext aux_ct = ws_aux;
    PhantomCiphertext t28_for_aux = t28;
    align_pair(ctx, aux_ct, t28_for_aux);       // both at d+2
    add_inplace(ctx, aux_ct, t28_for_aux);

#ifdef EVALMOD_STAGE_DEBUG
    // aux_ct = ws(aux_coeffs, T_1..T_7) + T_28
    report_stage(ctx, enc, aux_ct, rk, "aux_ct",
                 [](double x) {
                     return ref_weighted_sum(K28_AUX_COEFFS.data(),
                                             K28_AUX_COEFFS.size(), x)
                            + cheb_T(28, x);
                 });
#endif

    // quotient = (T_14 + ws_q0) · ws_q1 + ws_q2.
    // The_lib's chebyshev_paterson_stockmeyer WITH rk → eager relin → DUO.
    // We do the same: quotient stays DUO so the later aux·quotient TRIO
    // multiply has matching DUO×DUO inputs.
    PhantomCiphertext t14_plus_q0 = t14;
    add_inplace(ctx, t14_plus_q0, ws_q0);       // d+1
    PhantomCiphertext quotient = mul_and_relin(ctx, t14_plus_q0, ws_q1, rk);
    rescale_to_next_inplace(ctx, quotient);     // d+2
    snap_scale(quotient, target_scale);
    {
        PhantomCiphertext q2 = ws_q2;
        align_to(ctx, q2, quotient.chain_index()); // d+2
        add_inplace(ctx, quotient, q2);
    }

#ifdef EVALMOD_STAGE_DEBUG
    // quotient = (T_14 + ws_q0(x)) · ws_q1(x) + ws_q2(x)
    report_stage(ctx, enc, quotient, rk, "quotient",
                 [](double x) {
                     const double q0 = ref_weighted_sum(K28_QUOT_COEFFS[0].data(), 8, x);
                     const double q1 = ref_weighted_sum(K28_QUOT_COEFFS[1].data(), 8, x);
                     const double q2 = ref_weighted_sum(K28_QUOT_COEFFS[2].data(), 8, x);
                     return (cheb_T(14, x) + q0) * q1 + q2;
                 });
#endif

    // remainder = (T_14 + ws_r0) · ws_r1 + ws_r2.
    // The_lib's chebyshev_paterson_stockmeyer WITHOUT rk → TRIO. We mirror
    // this so the final result-add TRIO+TRIO sizes match.
    PhantomCiphertext t14_plus_r0 = t14;
    add_inplace(ctx, t14_plus_r0, ws_r0);       // d+1
    PhantomCiphertext remainder = mul_no_relin(ctx, t14_plus_r0, ws_r1);  // TRIO
    rescale_to_next_inplace(ctx, remainder);    // d+2
    snap_scale(remainder, target_scale);
    {
        PhantomCiphertext r2 = ws_r2;
        align_to(ctx, r2, remainder.chain_index());
        add_inplace(ctx, remainder, r2);        // TRIO + DUO → TRIO
    }

#ifdef EVALMOD_STAGE_DEBUG
    // remainder = (T_14 + ws_r0(x)) · ws_r1(x) + ws_r2(x)
    report_stage(ctx, enc, remainder, rk, "remainder",
                 [](double x) {
                     const double r0 = ref_weighted_sum(K28_REM_COEFFS[0].data(), 8, x);
                     const double r1 = ref_weighted_sum(K28_REM_COEFFS[1].data(), 8, x);
                     const double r2 = ref_weighted_sum(K28_REM_COEFFS[2].data(), 8, x);
                     return (cheb_T(14, x) + r0) * r1 + r2;
                 });
#endif

    // result = aux_ct · quotient + remainder − T_49   (depth d+3).
    // aux_ct (DUO) · quotient (DUO) WITHOUT relin → TRIO. Then add remainder
    // (TRIO) and subtract T_49 (TRIO). Single relin at the end converts the
    // TRIO result back to DUO. This matches the_lib::sine() exactly:
    // `relinearize(added, relinearization_key)` after all PS-tree operations.
    PhantomCiphertext result = mul_no_relin(ctx, aux_ct, quotient);  // TRIO
    rescale_to_next_inplace(ctx, result);       // d+3
    snap_scale(result, target_scale);

#ifdef EVALMOD_STAGE_DEBUG
    // aux_ct · quotient (TRIO, just after rescale, before remainder/T_49 fold).
    report_stage(ctx, enc, result, rk, "aux*quotient",
                 [](double x) {
                     const double aux = ref_weighted_sum(K28_AUX_COEFFS.data(),
                                                         K28_AUX_COEFFS.size(), x)
                                        + cheb_T(28, x);
                     const double q0 = ref_weighted_sum(K28_QUOT_COEFFS[0].data(), 8, x);
                     const double q1 = ref_weighted_sum(K28_QUOT_COEFFS[1].data(), 8, x);
                     const double q2 = ref_weighted_sum(K28_QUOT_COEFFS[2].data(), 8, x);
                     const double quot = (cheb_T(14, x) + q0) * q1 + q2;
                     return aux * quot;
                 });
#endif
    {
        PhantomCiphertext rem = remainder;
        align_to(ctx, rem, result.chain_index());
        add_inplace(ctx, result, rem);          // TRIO + TRIO
    }
    {
        PhantomCiphertext t49_a = t49;
        align_to(ctx, t49_a, result.chain_index());
        sub_inplace(ctx, result, t49_a);        // TRIO − TRIO
    }

    // Single deferred relinearisation: TRIO → DUO. This is the ONE relin
    // collapsed from what was previously 3 separate relins (T_49, remainder,
    // result), saving 3 × ~2 bits = ~6 bits of key-switching noise.
    relinearize_inplace(ctx, result, rk);

#ifdef EVALMOD_STAGE_DEBUG
    // Final sine polynomial output (before R=3 DA): aux·quot + rem − T_49.
    report_stage(ctx, enc, result, rk, "sine_final",
                 [](double x) {
                     const double aux = ref_weighted_sum(K28_AUX_COEFFS.data(),
                                                         K28_AUX_COEFFS.size(), x)
                                        + cheb_T(28, x);
                     const double q0 = ref_weighted_sum(K28_QUOT_COEFFS[0].data(), 8, x);
                     const double q1 = ref_weighted_sum(K28_QUOT_COEFFS[1].data(), 8, x);
                     const double q2 = ref_weighted_sum(K28_QUOT_COEFFS[2].data(), 8, x);
                     const double quot = (cheb_T(14, x) + q0) * q1 + q2;
                     const double r0 = ref_weighted_sum(K28_REM_COEFFS[0].data(), 8, x);
                     const double r1 = ref_weighted_sum(K28_REM_COEFFS[1].data(), 8, x);
                     const double r2 = ref_weighted_sum(K28_REM_COEFFS[2].data(), 8, x);
                     const double rem  = (cheb_T(14, x) + r0) * r1 + r2;
                     return aux * quot + rem - cheb_T(49, x);
                 });
#endif

    return result;
}

// ----------------------------------------------------------------------------
// apply_double_angle: ct ← 2·ct² − (2π)^{−2^{j−R}}  for j = 1..R.
// Each iteration consumes 1 level via square + rescale.
// ----------------------------------------------------------------------------
void apply_double_angle(const PhantomContext &ctx,
                        PhantomCKKSEncoder &enc,
                        PhantomCiphertext &ct,
                        const PhantomRelinKey &rk,
                        int R,
                        double target_scale) {
    constexpr double TWO_PI = 6.283185307179586;
    for (int j = 1; j <= R; ++j) {
        PhantomCiphertext sq = square_and_relin(ctx, ct, rk);
        const double exponent = std::pow(2.0, double(j - R));
        const double scalar   = -std::pow(TWO_PI, -exponent);
        rescale_double_add(ctx, enc, sq, scalar, target_scale);
        ct = std::move(sq);
    }
}

}  // namespace

// ============================================================================
// Public entries
// ============================================================================
PhantomCiphertext evalmod_k7_r3(const PhantomContext &ctx,
                                PhantomCKKSEncoder &enc,
                                const PhantomCiphertext &ct,
                                const PhantomRelinKey &rk) {
    const double target_scale = ct.scale();
    PhantomCiphertext sine = sine_chebyshev_k7(ctx, enc, ct, rk, target_scale);
    apply_double_angle(ctx, enc, sine, rk, EVALMOD_R, target_scale);
    return sine;
}

PhantomCiphertext evalmod_k16_r3(const PhantomContext &ctx,
                                 PhantomCKKSEncoder &enc,
                                 const PhantomCiphertext &ct,
                                 const PhantomRelinKey &rk) {
    const double target_scale = ct.scale();
    PhantomCiphertext sine = sine_chebyshev_deg31(ctx, enc, ct, rk, target_scale,
                                                  K16_Q_HIGH.data(), K16_Q_LOW.data(),
                                                  K16_R_HIGH.data(), K16_R_LOW.data());
    apply_double_angle(ctx, enc, sine, rk, EVALMOD_R, target_scale);
    return sine;
}

PhantomCiphertext evalmod_k28_r3(const PhantomContext &ctx,
                                 PhantomCKKSEncoder &enc,
                                 const PhantomCiphertext &ct,
                                 const PhantomRelinKey &rk) {
#ifdef SNAP_DEBUG
    snap_counter = 0;
    snap_drift_sum_bits = 0.0;
    snap_drift_max_bits = 0.0;
#endif
    const double target_scale = ct.scale();
    PhantomCiphertext sine = sine_chebyshev_k28(ctx, enc, ct, rk, target_scale);
    apply_double_angle(ctx, enc, sine, rk, EVALMOD_R, target_scale);
#ifdef SNAP_DEBUG
    snap_debug_summary();
#endif
    return sine;
}

// K=28 R=4: same sine polynomial as K=28 R=3 but with one extra DA iteration.
// apply_double_angle's per-iteration scalar = -(2π)^{−2^{j−R}} already takes R
// as a runtime arg, so the only change is R: 3 → 4. Total levels: 6 + 4 = 10.
PhantomCiphertext evalmod_k28_r4(const PhantomContext &ctx,
                                 PhantomCKKSEncoder &enc,
                                 const PhantomCiphertext &ct,
                                 const PhantomRelinKey &rk) {
    const double target_scale = ct.scale();
    PhantomCiphertext sine = sine_chebyshev_k28(ctx, enc, ct, rk, target_scale);
    apply_double_angle(ctx, enc, sine, rk, /*R=*/4, target_scale);
    return sine;
}

}  // namespace phantom
