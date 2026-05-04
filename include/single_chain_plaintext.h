#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ckks.h"
#include "context.cuh"
#include "plaintext.h"

namespace phantom {

    // RAII wrapper for pinned host int64 buffer.
    class PinnedHostInt64Buffer {
    public:
        PinnedHostInt64Buffer() = default;
        explicit PinnedHostInt64Buffer(std::size_t n);
        ~PinnedHostInt64Buffer();
        PinnedHostInt64Buffer(const PinnedHostInt64Buffer &other);
        PinnedHostInt64Buffer &operator=(const PinnedHostInt64Buffer &other);
        PinnedHostInt64Buffer(PinnedHostInt64Buffer &&other) noexcept;
        PinnedHostInt64Buffer &operator=(PinnedHostInt64Buffer &&other) noexcept;

        std::int64_t *data() noexcept { return ptr_; }
        const std::int64_t *data() const noexcept { return ptr_; }
        std::size_t size() const noexcept { return n_; }
        std::size_t nbytes() const noexcept { return n_ * sizeof(std::int64_t); }

    private:
        std::int64_t *ptr_ = nullptr;
        std::size_t n_ = 0;
    };

    // Compact, host-pinned, level-agnostic CKKS plaintext.
    // Storage: N * 8 B pinned host (e.g. 256 KB at logN=16).
    struct SingleChainPlaintext {
        PinnedHostInt64Buffer coeffs;   // length N, signed-centered
        double scale = 1.0;
    };

    // Encode a complex slot vector into a SingleChainPlaintext.
    // The integer poly is mod-q_0 signed-centered — valid as a level-agnostic
    // representation as long as |round(coeff*scale)| < q_0/2 (true for CKKS-typical
    // messages).
    SingleChainPlaintext encode_single_chain_plaintext(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const std::vector<std::complex<double>> &slots,
            double scale);

    // Re-tile + forward NTT to produce a full-RNS NTT-form PhantomPlaintext at
    // `target_chain_index`. Caller drops it after the multiply.
    PhantomPlaintext expand_single_chain_to_full(
            const PhantomContext &ctx,
            const SingleChainPlaintext &scp,
            std::size_t target_chain_index);
}
