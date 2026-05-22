#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ckks.h"
#include "context.cuh"
#include "plaintext.h"

namespace phantom {

    // RAII wrapper for pinned host int64 buffer. Used for full-scale SCPs
    // (rmsnorm gammas, masks, merge-bootstrap constants, complex bridges)
    // whose signed-centered coeffs at the 2^40 message scale do not fit int16.
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

    // RAII wrapper for pinned host int16 buffer. Used for quantized IRP weight
    // SCPs: the signed-centered coeffs at coeff_scale=2^16 are ~30, fitting
    // int16 with wide margin; the full 2^40 message scale is restored at expand
    // time by the per-tower scale_2 multiply. int16 (2 B) vs int64 (8 B) shrinks
    // the on-disk IRP cache 4x.
    class PinnedHostInt16Buffer {
    public:
        PinnedHostInt16Buffer() = default;
        explicit PinnedHostInt16Buffer(std::size_t n);
        ~PinnedHostInt16Buffer();
        PinnedHostInt16Buffer(const PinnedHostInt16Buffer &other);
        PinnedHostInt16Buffer &operator=(const PinnedHostInt16Buffer &other);
        PinnedHostInt16Buffer(PinnedHostInt16Buffer &&other) noexcept;
        PinnedHostInt16Buffer &operator=(PinnedHostInt16Buffer &&other) noexcept;

        std::int16_t *data() noexcept { return ptr_; }
        const std::int16_t *data() const noexcept { return ptr_; }
        std::size_t size() const noexcept { return n_; }
        std::size_t nbytes() const noexcept { return n_ * sizeof(std::int16_t); }

    private:
        std::int16_t *ptr_ = nullptr;
        std::size_t n_ = 0;
    };

    // Compact, host-pinned, level-agnostic CKKS plaintext.
    //
    // Adaptive coefficient storage: int16 when the signed-centered coeffs fit
    // (quantized IRP weight SCPs at coeff_scale=2^16, |coeff| ~30 -> 128 KB),
    // int64 otherwise (full-scale SCPs at coeff_scale=2^40 -> 512 KB). The
    // is_int16 flag selects which buffer is populated; size() of the active
    // buffer is the polynomial length N.
    //
    //   scale       — the effective/reported scale of the expanded plaintext
    //                 (the engine user_scale, e.g. 2^40). Downstream multiply
    //                 semantics see this scale, so it is identical for quantized
    //                 and full-scale SCPs.
    //   coeff_scale — the scale the coeffs are quantized at (2^16 for quantized
    //                 IRP weight SCPs, == scale for everything else). scale_2 =
    //                 round(scale / coeff_scale) is applied per-tower in the
    //                 expand kernel to restore the full message scale (==1 when
    //                 coeff_scale == scale).
    struct SingleChainPlaintext {
        PinnedHostInt16Buffer coeffs;       // populated when is_int16
        PinnedHostInt64Buffer coeffs64;     // populated when !is_int16
        bool is_int16 = true;
        double scale = 1.0;
        double coeff_scale = 1.0;

        // Polynomial length N from whichever buffer is active.
        std::size_t N() const {
            return is_int16 ? coeffs.size() : coeffs64.size();
        }
        // Raw coefficient bytes (N * 2 if int16, else N * 8).
        std::size_t nbytes() const {
            return is_int16 ? coeffs.nbytes() : coeffs64.nbytes();
        }
        const void *coeff_ptr() const {
            return is_int16 ? static_cast<const void *>(coeffs.data())
                            : static_cast<const void *>(coeffs64.data());
        }
    };

    // Encode a complex slot vector into a SingleChainPlaintext.
    // The integer poly is mod-q_0 signed-centered — valid as a level-agnostic
    // representation as long as |round(coeff*coeff_scale)| < q_0/2.
    //
    // `scale` is the effective scale stored on the SCP (the value the expanded
    // plaintext reports). `coeff_scale` is the scale used for the actual integer
    // encoding; pass <= 0 to use `scale` (full-scale, scale_2 == 1). The encoder
    // stores the coeffs as int16 when they fit (lossless) and int64 otherwise.
    SingleChainPlaintext encode_single_chain_plaintext(
            const PhantomContext &ctx,
            PhantomCKKSEncoder &encoder,
            const std::vector<std::complex<double>> &slots,
            double scale,
            double coeff_scale = 0.0);

    // Re-tile + forward NTT to produce a full-RNS NTT-form PhantomPlaintext at
    // `target_chain_index`. Caller drops it after the multiply.
    PhantomPlaintext expand_single_chain_to_full(
            const PhantomContext &ctx,
            const SingleChainPlaintext &scp,
            std::size_t target_chain_index);
}
