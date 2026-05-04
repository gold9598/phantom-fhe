#pragma once

#include <vector>

#include "ciphertext.h"
#include "ckks.h"
#include "context.cuh"
#include "secretkey.h"

namespace phantom {

    // Evaluate p(x) = a_0 + a_1*x + ... + a_d*x^d on encrypted x via
    // Paterson-Stockmeyer baby-step giant-step.
    //
    // coeffs[i] = a_i. Requires coeffs.size() >= 2 (degree >= 1).
    // For degree d: m = ceil(sqrt(d+1)), l = ceil((d+1)/m).
    // Result chain_index = ct.chain_index() + (m + l - 1).
    PhantomCiphertext eval_polynomial(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomCiphertext &ct,
            const std::vector<double> &coeffs);
}
