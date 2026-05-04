#pragma once

#include <cstddef>
#include <vector>

#include "ciphertext.h"
#include "ckks.h"
#include "context.cuh"
#include "secretkey.h"

namespace phantom {

    struct RmsNormParams {
        std::size_t d_model;
        double epsilon;
        double z_min;
        double z_max;
        std::size_t poly_degree;
    };

    struct RmsNormWeights {
        std::vector<double> g_tiled_real; // length num_slots
        std::vector<double> g;            // raw, length d_model
    };

    // Forward pass: x is encrypted in period-d_model replicated layout.
    // Output: same layout, with rmsnorm(x)*g applied.
    PhantomCiphertext rmsnorm_forward(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            const RmsNormWeights &weights,
            const RmsNormParams &params);

}
