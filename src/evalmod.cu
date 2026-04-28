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

constexpr int  EVALMOD_R       = 3;
constexpr int  CHEB_MAX_POWER  = 8;

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
PhantomCiphertext
sine_chebyshev_k16(const PhantomContext &ctx,
                   PhantomCKKSEncoder &enc,
                   const PhantomCiphertext &ct,
                   const PhantomRelinKey &rk,
                   double target_scale) {
    auto baby = build_chebyshev_basis(ctx, enc, ct, CHEB_MAX_POWER, rk, target_scale);

    // T_16 = 2·T_8² − 1.
    PhantomCiphertext t16 = square_and_relin(ctx, baby[7], rk);
    rescale_to_next_inplace(ctx, t16);
    snap_scale(t16, target_scale);
    add_scalar_inplace(ctx, enc, t16, -0.5);
    double_ct(ctx, t16);

    // Inner weighted sums over T_1..T_7 (parallel; each consumes 1 level).
    std::vector<PhantomCiphertext> baby7(baby.begin(), baby.begin() + 7);
    auto q_high = weighted_sum_chebyshev(ctx, enc, baby7, K16_Q_HIGH.data(), K16_Q_HIGH.size(), target_scale);
    auto q_low  = weighted_sum_chebyshev(ctx, enc, baby7, K16_Q_LOW .data(), K16_Q_LOW .size(), target_scale);
    auto r_high = weighted_sum_chebyshev(ctx, enc, baby7, K16_R_HIGH.data(), K16_R_HIGH.size(), target_scale);
    auto r_low  = weighted_sum_chebyshev(ctx, enc, baby7, K16_R_LOW .data(), K16_R_LOW .size(), target_scale);

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
    PhantomCiphertext sine = sine_chebyshev_k16(ctx, enc, ct, rk, target_scale);
    apply_double_angle(ctx, enc, sine, rk, EVALMOD_R, target_scale);
    return sine;
}

}  // namespace phantom
