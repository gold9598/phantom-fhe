#include "evalmod.h"
#include "evaluate.cuh"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace phantom {

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
void snap_scale(PhantomCiphertext &ct, double target_scale) {
    ct.set_scale(target_scale);
}

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

// Encode `value` at the same scale + chain_index as `ref`, then add to ref.
// Used for level-free scalar adds (e.g. the c_0 constant term, the DA scalar).
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

    bool first_term = true;
    PhantomCiphertext acc;

    const size_t slots = enc.slot_count();
    for (size_t i = 1; i < n_coeffs; ++i) {
        if (std::abs(coeffs[i]) < 1e-50) continue;

        std::vector<double> rep(slots, coeffs[i]);
        PhantomPlaintext pt;
        enc.encode(ctx, rep, baby_scale, pt, baby_idx);

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

    // T_14 = 2·T_7² − 1   (depth d+1).
    PhantomCiphertext t14 = square_and_relin(ctx, t7, rk);
    rescale_to_next_inplace(ctx, t14);
    snap_scale(t14, target_scale);
    add_scalar_inplace(ctx, enc, t14, -0.5);
    double_ct(ctx, t14);

    // T_28 = 2·T_14² − 1   (depth d+2).
    PhantomCiphertext t28 = square_and_relin(ctx, t14, rk);
    rescale_to_next_inplace(ctx, t28);
    snap_scale(t28, target_scale);
    add_scalar_inplace(ctx, enc, t28, -0.5);
    double_ct(ctx, t28);

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

    // T_21 = 2·T_14·T_7 − T_7  (depth d+2).
    // T_14 at d+1, T_7 (baby[6]) at d. Align T_7 up to T_14 level.
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

    // T_49 = 2·T_28·T_21 − T_7  (depth d+3).
    PhantomCiphertext t28_for_t49 = t28;
    align_pair(ctx, t28_for_t49, t21);          // both at d+2
    PhantomCiphertext t49 = mul_and_relin(ctx, t28_for_t49, t21, rk);
    rescale_to_next_inplace(ctx, t49);          // d+3
    snap_scale(t49, target_scale);
    double_ct(ctx, t49);
    {
        PhantomCiphertext t7_b = t7;
        align_to(ctx, t7_b, t49.chain_index()); // d+3
        sub_inplace(ctx, t49, t7_b);            // − T_7
    }

    // aux_ct = ws_aux + T_28   (depth d+2). ws_aux at d+1 → align down.
    PhantomCiphertext aux_ct = ws_aux;
    PhantomCiphertext t28_for_aux = t28;
    align_pair(ctx, aux_ct, t28_for_aux);       // both at d+2
    add_inplace(ctx, aux_ct, t28_for_aux);

    // quotient = (T_14 + ws_q0) · ws_q1 + ws_q2.
    // T_14 at d+1, ws_q0 at d+1 → add at d+1 (no level cost).
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

    // remainder = (T_14 + ws_r0) · ws_r1 + ws_r2.
    PhantomCiphertext t14_plus_r0 = t14;
    add_inplace(ctx, t14_plus_r0, ws_r0);       // d+1
    PhantomCiphertext remainder = mul_and_relin(ctx, t14_plus_r0, ws_r1, rk);
    rescale_to_next_inplace(ctx, remainder);    // d+2
    snap_scale(remainder, target_scale);
    {
        PhantomCiphertext r2 = ws_r2;
        align_to(ctx, r2, remainder.chain_index());
        add_inplace(ctx, remainder, r2);
    }

    // result = aux_ct · quotient + remainder − T_49   (depth d+3).
    PhantomCiphertext result = mul_and_relin(ctx, aux_ct, quotient, rk);
    rescale_to_next_inplace(ctx, result);       // d+3
    snap_scale(result, target_scale);
    {
        PhantomCiphertext rem = remainder;
        align_to(ctx, rem, result.chain_index());
        add_inplace(ctx, result, rem);
    }
    {
        PhantomCiphertext t49_a = t49;
        align_to(ctx, t49_a, result.chain_index());
        sub_inplace(ctx, result, t49_a);
    }

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
    const double target_scale = ct.scale();
    PhantomCiphertext sine = sine_chebyshev_k28(ctx, enc, ct, rk, target_scale);
    apply_double_angle(ctx, enc, sine, rk, EVALMOD_R, target_scale);
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
