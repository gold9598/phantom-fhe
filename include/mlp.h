#pragma once

#include <cstddef>

#include "bsgs.h"
#include "ciphertext.h"
#include "ckks.h"
#include "context.cuh"
#include "secretkey.h"

namespace phantom {

    struct MlpWeights {
        BsgsDiagonals w_gate;
        BsgsDiagonals w_up;
        BsgsDiagonals w_down;
    };

    PhantomCiphertext mlp_forward(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            const MlpWeights &w);

    // ---- Complex-folded MLP (2x faster matmuls, +5 levels) ----

    struct MlpWeightsComplex {
        ComplexBsgsDiagonals w_gate;       // row-folded: (d_hidden/2 x d_model)
        ComplexBsgsDiagonals w_up;         // row-folded
        ComplexBsgsDiagonals w_down;       // col-folded with conjugate: (d_model x d_hidden/2)
        std::size_t d_model = 0;
        std::size_t d_hidden = 0;
        std::size_t d_pad = 0;             // halved relative to real MLP
    };

    PhantomCiphertext mlp_forward_complex(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const PhantomRelinKey &relin_key,
            const PhantomGaloisKey &galois_key,
            const PhantomCiphertext &x,
            const MlpWeightsComplex &w);

}
