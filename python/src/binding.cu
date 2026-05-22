#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <cuda_runtime.h>

#include "phantom.h"
#include "attention.h"
#include "bootstrap.h"
#include "bsgs.h"
#include "ckks_engine.h"
#include "linear.h"
#include "mlp.h"
#include "ps.h"
#include "rmsnorm.h"
#include "single_chain_plaintext.h"
#include "softmax.h"

namespace py = pybind11;

PYBIND11_MODULE(pyPhantom, m) {

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::enum_<phantom::scheme_type>(m, "scheme_type")
            .value("none", phantom::scheme_type::none)
            .value("bgv", phantom::scheme_type::bgv)
            .value("bfv", phantom::scheme_type::bfv)
            .value("ckks", phantom::scheme_type::ckks)
            .export_values();

    py::enum_<phantom::mul_tech_type>(m, "mul_tech_type")
            .value("none", phantom::mul_tech_type::none)
            .value("behz", phantom::mul_tech_type::behz)
            .value("hps", phantom::mul_tech_type::hps)
            .value("hps_overq", phantom::mul_tech_type::hps_overq)
            .value("hps_overq_leveled", phantom::mul_tech_type::hps_overq_leveled)
            .export_values();

    py::enum_<phantom::arith::sec_level_type>(m, "sec_level_type")
            .value("none", phantom::arith::sec_level_type::none)
            .value("tc128", phantom::arith::sec_level_type::tc128)
            .value("tc192", phantom::arith::sec_level_type::tc192)
            .value("tc256", phantom::arith::sec_level_type::tc256)
            .export_values();

    py::class_<phantom::arith::Modulus>(m, "modulus")
            .def(py::init<std::uint64_t>());

    m.def("create_coeff_modulus", &phantom::arith::CoeffModulus::Create);

    m.def("create_plain_modulus", &phantom::arith::PlainModulus::Batching);

    py::class_<phantom::EncryptionParameters>(m, "params")
            .def(py::init<phantom::scheme_type>())
            .def("set_mul_tech", &phantom::EncryptionParameters::set_mul_tech)
            .def("set_poly_modulus_degree", &phantom::EncryptionParameters::set_poly_modulus_degree)
            .def("set_special_modulus_size", &phantom::EncryptionParameters::set_special_modulus_size)
            .def("set_galois_elts", &phantom::EncryptionParameters::set_galois_elts)
            .def("set_coeff_modulus", &phantom::EncryptionParameters::set_coeff_modulus)
            .def("set_plain_modulus", &phantom::EncryptionParameters::set_plain_modulus);

    py::class_<phantom::util::cuda_stream_wrapper>(m, "cuda_stream")
            .def(py::init<>());

    py::class_<PhantomContext>(m, "context")
            .def(py::init<phantom::EncryptionParameters &>())
            .def("total_parm_size", &PhantomContext::total_parm_size)
            .def("get_first_index", &PhantomContext::get_first_index);

    py::class_<PhantomSecretKey>(m, "secret_key")
            .def(py::init<>())
            .def(py::init<const PhantomContext &>())
            .def("generate_sparse", &PhantomSecretKey::generate_sparse,
                 py::arg("context"), py::arg("hamming_weight"))
            .def("gen_publickey", &PhantomSecretKey::gen_publickey)
            .def("gen_relinkey", &PhantomSecretKey::gen_relinkey)
            .def("create_galois_keys", &PhantomSecretKey::create_galois_keys)
            .def("create_galois_keys_per_level",
                 &PhantomSecretKey::create_galois_keys_per_level,
                 py::arg("context"), py::arg("indices"), py::arg("target_chain_indices"))
            .def("create_one_galois_key",
                 &PhantomSecretKey::create_one_galois_key,
                 py::arg("context"), py::arg("galois_elt_idx"), py::arg("target_chain_index") = 0)
            .def("encrypt_symmetric",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomSecretKey::encrypt_symmetric, py::const_), py::arg(), py::arg(),
                 py::call_guard<py::gil_scoped_release>())
            .def("decrypt",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &>(
                         &PhantomSecretKey::decrypt), py::arg(), py::arg(),
                 py::call_guard<py::gil_scoped_release>());

    py::class_<PhantomPublicKey>(m, "public_key")
            .def(py::init<>())
            .def("encrypt_asymmetric",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomPublicKey::encrypt_asymmetric), py::arg(), py::arg(),
                 py::call_guard<py::gil_scoped_release>());

    py::class_<PhantomRelinKey>(m, "relin_key")
            .def(py::init<>());

    py::class_<PhantomGaloisKey>(m, "galois_key")
            .def(py::init<>());

    m.def("get_elt_from_step", &phantom::util::get_elt_from_step);

    m.def("get_elts_from_steps", &phantom::util::get_elts_from_steps);

    py::class_<PhantomBatchEncoder>(m, "batch_encoder")
            .def(py::init<const PhantomContext &>())
            .def("slot_count", &PhantomBatchEncoder::slot_count)
            .def("encode",
                 py::overload_cast<const PhantomContext &, const std::vector<uint64_t> &>(
                         &PhantomBatchEncoder::encode, py::const_), py::arg(), py::arg())
            .def("decode",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomBatchEncoder::decode, py::const_), py::arg(), py::arg());

    py::class_<PhantomCKKSEncoder>(m, "ckks_encoder")
            .def(py::init<const PhantomContext &>())
            .def("slot_count", &PhantomCKKSEncoder::slot_count)
            // numpy-array fast path: skips per-element Python iteration. Pybind11
            // dispatches here when caller passes a numpy complex128 array.
            .def("encode_complex_vector",
                 [](PhantomCKKSEncoder &encoder, const PhantomContext &ctx,
                    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> arr,
                    double scale, size_t chain_index) {
                     if (arr.ndim() != 1)
                         throw std::runtime_error(
                             "encode_complex_vector: array must be 1-D");
                     const std::complex<double> *src = arr.data();
                     const std::size_t n = static_cast<std::size_t>(arr.shape(0));
                     std::vector<cuDoubleComplex> values(n);
                     for (std::size_t i = 0; i < n; ++i) {
                         values[i] = make_cuDoubleComplex(src[i].real(), src[i].imag());
                     }
                     py::gil_scoped_release release;
                     return encoder.encode<cuDoubleComplex>(ctx, values, scale, chain_index);
                 },
                 py::arg(), py::arg(), py::arg(), py::arg("chain_index") = 1,
                 "Encode a numpy complex128 1-D array (fast path — no Python "
                 "object iteration).")
            .def("encode_complex_vector",
                 py::overload_cast<const PhantomContext &, const std::vector<cuDoubleComplex> &, double, size_t>(
                         &PhantomCKKSEncoder::encode<cuDoubleComplex>),
                 py::arg(), py::arg(), py::arg(), py::arg("chain_index") = 1,
                 py::call_guard<py::gil_scoped_release>())
            // numpy-array fast path for double-precision real input.
            .def("encode_double_vector",
                 [](PhantomCKKSEncoder &encoder, const PhantomContext &ctx,
                    py::array_t<double, py::array::c_style | py::array::forcecast> arr,
                    double scale, size_t chain_index) {
                     if (arr.ndim() != 1)
                         throw std::runtime_error(
                             "encode_double_vector: array must be 1-D");
                     const double *src = arr.data();
                     const std::size_t n = static_cast<std::size_t>(arr.shape(0));
                     std::vector<double> values(src, src + n);
                     py::gil_scoped_release release;
                     return encoder.encode<double>(ctx, values, scale, chain_index);
                 },
                 py::arg(), py::arg(), py::arg(), py::arg("chain_index") = 1,
                 "Encode a numpy float64 1-D array (fast path — no Python "
                 "object iteration).")
            .def("encode_double_vector",
                 py::overload_cast<const PhantomContext &, const std::vector<double> &, double, size_t>(
                         &PhantomCKKSEncoder::encode<double>),
                 py::arg(), py::arg(), py::arg(),
                 py::arg("chain_index") = 1,
                 py::call_guard<py::gil_scoped_release>())
            .def("decode_complex_vector",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomCKKSEncoder::decode<cuDoubleComplex>),
                 py::arg(), py::arg(),
                 py::call_guard<py::gil_scoped_release>())
            .def("decode_double_vector",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                         &PhantomCKKSEncoder::decode<double>), py::arg(), py::arg(),
                 py::call_guard<py::gil_scoped_release>());

    py::class_<PhantomPlaintext>(m, "plaintext")
            .def(py::init<>());

    py::class_<PhantomCiphertext>(m, "ciphertext")
            .def(py::init<>())
            .def("set_scale", &PhantomCiphertext::set_scale)
            .def("chain_index", [](const PhantomCiphertext &ct) { return ct.chain_index(); })
            .def("scale", [](const PhantomCiphertext &ct) { return ct.scale(); });

    m.def("negate", &phantom::negate, py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("add", &phantom::add, py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("add_plain", &phantom::add_plain, py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("add_many", &phantom::add_many, py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("sub", &phantom::sub, py::arg(), py::arg(), py::arg(), py::arg("negate") = false,
          py::call_guard<py::gil_scoped_release>());

    m.def("sub_plain", &phantom::sub_plain, py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("multiply", &phantom::multiply, py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("multiply_and_relin", &phantom::multiply_and_relin, py::arg(), py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("multiply_plain", &phantom::multiply_plain, py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("relinearize", &phantom::relinearize, py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("rescale_to_next", &phantom::rescale_to_next, py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("mod_switch_to_next",
          py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(&phantom::mod_switch_to_next),
          py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("mod_switch_to_next",
          py::overload_cast<const PhantomContext &, const PhantomCiphertext &>(&phantom::mod_switch_to_next),
          py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("mod_switch_to", py::overload_cast<const PhantomContext &, const PhantomPlaintext &, size_t>(
            &phantom::mod_switch_to), py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("mod_switch_to", py::overload_cast<const PhantomContext &, const PhantomCiphertext &, size_t>(
            &phantom::mod_switch_to), py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("mod_switch_to_inplace",
          py::overload_cast<const PhantomContext &, PhantomCiphertext &, size_t>(
                  &phantom::mod_switch_to_inplace),
          py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("mod_switch_to_inplace",
          py::overload_cast<const PhantomContext &, PhantomPlaintext &, size_t>(
                  &phantom::mod_switch_to_inplace),
          py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("apply_galois",
          py::overload_cast<const PhantomContext &, const PhantomCiphertext &, size_t,
                            const PhantomGaloisKey &>(&phantom::apply_galois),
          py::arg(), py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("rotate",
          py::overload_cast<const PhantomContext &, const PhantomCiphertext &, int,
                            const PhantomGaloisKey &>(&phantom::rotate),
          py::arg(), py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("apply_galois_with_key",
          py::overload_cast<const PhantomContext &, const PhantomCiphertext &, size_t,
                            const PhantomRelinKey &>(&phantom::apply_galois),
          py::arg(), py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("rotate_with_key",
          py::overload_cast<const PhantomContext &, const PhantomCiphertext &, int,
                            const PhantomRelinKey &>(&phantom::rotate),
          py::arg(), py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("hoisting", &phantom::hoisting, py::arg(), py::arg(), py::arg(), py::arg(),
          py::call_guard<py::gil_scoped_release>());

    m.def("hoist_rotations", &phantom::hoist_rotations,
          py::arg("context"), py::arg("ct"), py::arg("glk"), py::arg("steps"),
          py::call_guard<py::gil_scoped_release>());

    // ===== CKKS bootstrap (Phase 6) =====
    py::class_<phantom::SmallBootstrapKey>(m, "small_bootstrap_key");

    py::class_<phantom::BootstrapKey>(m, "bootstrap_key");

    // Compute the union of Galois elements needed by C2S + S2C for the given
    // log_n and stages_per_layer, plus the conjugation element (2N-1).
    // Returns a list of uint32_t galois elements suitable for
    // params.set_galois_elts(). Default stages=[5,5,5] for logN=16.
    m.def("bootstrap_required_galois_elts",
          [](int log_n, std::vector<int> stages_per_layer) -> std::vector<uint32_t> {
              const size_t N = size_t(1) << log_n;
              const int num_slots = static_cast<int>(N >> 1);
              auto c2s_h = phantom::build_c2s_diagonals(log_n, stages_per_layer);
              auto s2c_h = phantom::build_s2c_diagonals(log_n, stages_per_layer);
              auto c2s_steps = phantom::c2s_required_rotation_steps(c2s_h, num_slots);
              auto s2c_steps = phantom::c2s_required_rotation_steps(s2c_h, num_slots);
              std::vector<int> all_steps = c2s_steps;
              all_steps.insert(all_steps.end(), s2c_steps.begin(), s2c_steps.end());
              std::sort(all_steps.begin(), all_steps.end());
              all_steps.erase(std::unique(all_steps.begin(), all_steps.end()), all_steps.end());
              auto elts = phantom::util::get_elts_from_steps(all_steps, N);
              elts.push_back(static_cast<uint32_t>(2 * N - 1)); // conjugation
              return elts;
          },
          py::arg("log_n"),
          py::arg("stages_per_layer") = std::vector<int>{5, 5, 5});

    m.def("create_bootstrap_key", &phantom::create_bootstrap_key,
          py::arg("context"),
          py::arg("encoder"),
          py::arg("dense_sk"),
          py::arg("sparse_hamming_weight") = 128,
          py::arg("eval_mod_levels") = 0,
          py::arg("user_scale") = 0.0,
          py::arg("split_scale_down") = false);

    m.def("bootstrap", &phantom::bootstrap,
          py::arg("context"),
          py::arg("encoder"),
          py::arg("ct"),
          py::arg("bk"),
          py::arg("user_scale"),
          py::arg("split_scale_down") = false,
          py::call_guard<py::gil_scoped_release>());

    // ===== CKKSEngine: user-facing facade with bootstrap =====
    py::class_<phantom::CKKSEngineConfig>(m, "ckks_engine_config")
            .def(py::init<>())
            .def_readwrite("log_n", &phantom::CKKSEngineConfig::log_n)
            .def_readwrite("user_scale", &phantom::CKKSEngineConfig::user_scale)
            .def_readwrite("num_scale_levels", &phantom::CKKSEngineConfig::num_scale_levels)
            .def_readwrite("sparse_hw", &phantom::CKKSEngineConfig::sparse_hw)
            .def_readwrite("num_special_primes", &phantom::CKKSEngineConfig::num_special_primes)
            .def_readwrite("include_user_rotations", &phantom::CKKSEngineConfig::include_user_rotations)
            .def_readwrite("user_rotation_steps", &phantom::CKKSEngineConfig::user_rotation_steps)
            .def_readwrite("user_rotation_target_chain_indices",
                           &phantom::CKKSEngineConfig::user_rotation_target_chain_indices)
            .def_readwrite("split_scale_down",
                           &phantom::CKKSEngineConfig::split_scale_down)
            .def_readwrite("build_two_scale_arrays",
                           &phantom::CKKSEngineConfig::build_two_scale_arrays)
            .def_readwrite("use_bootstrap_to_17_levels",
                           &phantom::CKKSEngineConfig::use_bootstrap_to_17_levels);

    py::class_<phantom::CKKSEngine>(m, "ckks_engine")
            .def(py::init<const phantom::CKKSEngineConfig &>(), py::arg("config"))
            .def("slot_count", &phantom::CKKSEngine::slot_count)
            .def("user_scale", &phantom::CKKSEngine::user_scale)
            .def("max_user_level", &phantom::CKKSEngine::max_user_level)
            .def("user_level", &phantom::CKKSEngine::user_level)
            .def("user_level_chain_index", &phantom::CKKSEngine::user_level_chain_index)
            .def("encrypt", &phantom::CKKSEngine::encrypt,
                 py::call_guard<py::gil_scoped_release>())
            .def("decrypt_decode", &phantom::CKKSEngine::decrypt_decode,
                 py::call_guard<py::gil_scoped_release>())
            .def("bootstrap_inplace", &phantom::CKKSEngine::bootstrap_inplace,
                 py::call_guard<py::gil_scoped_release>())
            .def("context", &phantom::CKKSEngine::context, py::return_value_policy::reference_internal)
            .def("encoder", &phantom::CKKSEngine::mutable_encoder, py::return_value_policy::reference_internal)
            .def("secret_key", &phantom::CKKSEngine::mutable_secret_key, py::return_value_policy::reference_internal)
            .def("relin_key", &phantom::CKKSEngine::relin_key, py::return_value_policy::reference_internal)
            .def("galois_key", &phantom::CKKSEngine::galois_key, py::return_value_policy::reference_internal)
            .def("scale_array_size", &phantom::CKKSEngine::scale_array_size)
            // long double → double (precision-lossy but adequate for inspection).
            .def("ckks_scale_at",
                 [](const phantom::CKKSEngine &e, std::size_t idx) {
                     return static_cast<double>(e.ckks_scale_at(idx));
                 })
            .def("ckks_rescaled_scale_at",
                 [](const phantom::CKKSEngine &e, std::size_t idx) {
                     return static_cast<double>(e.ckks_rescaled_scale_at(idx));
                 });

    // ===== CUDA device control =====
    m.def("set_cuda_device", [](int dev) {
        auto err = cudaSetDevice(dev);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                                     cudaGetErrorString(err));
    }, py::arg("device_id"),
       "Set the active CUDA device for the calling thread. Required before "
       "constructing a CKKSEngine on a non-default GPU and at the start of each "
       "worker thread in multi-GPU sweeps.");

    m.def("get_cuda_device_count", []() -> int {
        int count = 0;
        auto err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess) return 0;
        return count;
    }, "Return the number of CUDA devices visible to this process.");

    // ===== Single-chain (host-pinned, level-agnostic) plaintext =====
    py::class_<phantom::SingleChainPlaintext>(m, "single_chain_plaintext")
            .def_property_readonly("scale",
                                   [](const phantom::SingleChainPlaintext &p) { return p.scale; })
            .def_property_readonly("nbytes",
                                   [](const phantom::SingleChainPlaintext &p) { return p.coeffs.nbytes(); })
            .def_property_readonly("N",
                                   [](const phantom::SingleChainPlaintext &p) { return p.coeffs.size(); })
            // Raw int64 coeffs as py::bytes. Releases the GIL during the
            // copy so concurrent workers can serialize cache entries in
            // parallel; the SCP buffer is in pinned host memory and is
            // safe to read while other threads run.
            .def("coeffs_bytes",
                 [](const phantom::SingleChainPlaintext &p) {
                     return py::bytes(reinterpret_cast<const char *>(p.coeffs.data()),
                                      p.coeffs.nbytes());
                 },
                 "Return the SCP's int64 coefficient buffer as raw bytes "
                 "(length N * 8). Pair with scp_from_bytes to round-trip "
                 "an SCP to/from disk.")
            .def("get_scale",
                 [](const phantom::SingleChainPlaintext &p) { return p.scale; },
                 "Return the SCP's encoding scale.");

    // numpy-array fast path for encode_single_chain_plaintext: pybind11
    // dispatches here when callers pass a numpy complex128 array directly.
    // Avoids the GIL-held per-element marshalling of `.tolist()` -> std::vector
    // of std::complex (~50 ms per call at N=65536). With 32 layers * 256 SCPs
    // per Wq IRP pre-encode worker, this saves ~13 s of pure GIL-held time
    // per worker, unblocking concurrent encode threads.
    m.def("encode_single_chain_plaintext",
          [](const PhantomContext &ctx, PhantomCKKSEncoder &encoder,
             py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> arr,
             double scale) {
              if (arr.ndim() != 1)
                  throw std::runtime_error(
                      "encode_single_chain_plaintext: array must be 1-D");
              const std::complex<double> *src = arr.data();
              const std::size_t n = static_cast<std::size_t>(arr.shape(0));
              std::vector<std::complex<double>> slots(src, src + n);
              py::gil_scoped_release release;
              return phantom::encode_single_chain_plaintext(ctx, encoder, slots, scale);
          },
          py::arg("context"), py::arg("encoder"), py::arg("slots"), py::arg("scale"),
          "Encode a numpy complex128 1-D array into a SingleChainPlaintext. "
          "Faster than the list-based overload — no Python object iteration.");

    // numpy-array fast path with real (float64) input. Promotes to complex
    // inside the C++ boundary while the GIL is held only briefly.
    m.def("encode_single_chain_plaintext",
          [](const PhantomContext &ctx, PhantomCKKSEncoder &encoder,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr,
             double scale) {
              if (arr.ndim() != 1)
                  throw std::runtime_error(
                      "encode_single_chain_plaintext: array must be 1-D");
              const double *src = arr.data();
              const std::size_t n = static_cast<std::size_t>(arr.shape(0));
              std::vector<std::complex<double>> slots(n);
              for (std::size_t i = 0; i < n; ++i) {
                  slots[i] = std::complex<double>(src[i], 0.0);
              }
              py::gil_scoped_release release;
              return phantom::encode_single_chain_plaintext(ctx, encoder, slots, scale);
          },
          py::arg("context"), py::arg("encoder"), py::arg("slots"), py::arg("scale"),
          "Encode a numpy float64 1-D array into a SingleChainPlaintext (real "
          "values promoted to complex). Faster than the list-based overload.");

    m.def("encode_single_chain_plaintext",
          [](const PhantomContext &ctx, PhantomCKKSEncoder &encoder,
             std::vector<std::complex<double>> slots, double scale) {
              // `slots` is already a C++ vector at this point (pybind11
              // unpacked the Python list during argument marshalling), so
              // we no longer need the GIL for the NTT + encode work below.
              // Releasing here lets concurrent worker threads encode in
              // parallel during the Wq IRP pre-encode phase (32 layers x
              // 256 SCPs = 8,192 calls per example).
              py::gil_scoped_release release;
              return phantom::encode_single_chain_plaintext(ctx, encoder, slots, scale);
          },
          py::arg("context"), py::arg("encoder"), py::arg("slots"), py::arg("scale"));

    // Reconstruct a SingleChainPlaintext from raw coeff bytes + scale.
    // Allocates pinned host memory and memcpy's the bytes in. Used by the
    // disk-persistent IRP cache to avoid re-encoding plaintexts across
    // process restarts. No target_chain_index needed: SCPs are
    // level-agnostic, the chain index is supplied at expand-time only.
    m.def("scp_from_bytes",
          [](py::bytes data, double scale, std::size_t N) {
              // Extract the bytes into a std::string while we still hold
              // the GIL — converting py::bytes -> std::string touches the
              // Python object.
              std::string s(data);
              if (s.size() != N * sizeof(std::int64_t)) {
                  throw std::runtime_error(
                      "scp_from_bytes: byte length mismatch (got " +
                      std::to_string(s.size()) + ", expected " +
                      std::to_string(N * sizeof(std::int64_t)) + ")");
              }
              // The pinned-host allocation + memcpy are pure C++; release
              // the GIL so concurrent workers can deserialize SCPs in
              // parallel during the disk-cache load.
              py::gil_scoped_release release;
              phantom::SingleChainPlaintext out;
              out.scale = scale;
              out.coeffs = phantom::PinnedHostInt64Buffer(N);
              std::memcpy(out.coeffs.data(), s.data(), s.size());
              return out;
          },
          py::arg("data"), py::arg("scale"), py::arg("N"),
          "Reconstruct a SingleChainPlaintext from raw int64 coefficient "
          "bytes (length N * 8) and a scale. Inverse of coeffs_bytes().");

    m.def("expand_single_chain_to_full", &phantom::expand_single_chain_to_full,
          py::arg("context"), py::arg("scp"), py::arg("target_chain_index"),
          py::call_guard<py::gil_scoped_release>());

    // ===== FD-packed matrix-vector multiply =====
    m.def("inner_sum", &phantom::inner_sum,
          py::arg("context"), py::arg("galois_key"), py::arg("ct"),
          py::arg("block_size"),
          py::call_guard<py::gil_scoped_release>());

    m.def("replicate", &phantom::replicate,
          py::arg("context"), py::arg("galois_key"), py::arg("ct"),
          py::arg("period"), py::arg("num_slots"),
          py::call_guard<py::gil_scoped_release>());

    // ===== BSGS-diagonal matrix-vector multiply =====
    py::class_<phantom::BsgsDiagonals>(m, "bsgs_diagonals")
            .def_property_readonly("d_pad",
                                   [](const phantom::BsgsDiagonals &d) { return d.d_pad; })
            .def_property_readonly("baby_steps",
                                   [](const phantom::BsgsDiagonals &d) { return d.baby_steps; })
            .def_property_readonly("giant_steps",
                                   [](const phantom::BsgsDiagonals &d) { return d.giant_steps; });

    m.def("pre_encode_bsgs_diagonals",
          [](const PhantomContext &ctx, PhantomCKKSEncoder &encoder,
             const std::vector<double> &matrix,
             std::size_t num_rows, std::size_t num_cols,
             std::size_t d_pad, std::size_t baby_steps, double scale) {
              return phantom::pre_encode_bsgs_diagonals(
                      ctx, encoder, matrix, num_rows, num_cols, d_pad, baby_steps, scale);
          },
          py::arg("context"), py::arg("encoder"), py::arg("matrix"),
          py::arg("num_rows"), py::arg("num_cols"),
          py::arg("d_pad"), py::arg("baby_steps"), py::arg("scale"),
          py::return_value_policy::move);

    m.def("bsgs_required_steps", &phantom::bsgs_required_steps, py::arg("baby_steps"));

    m.def("fused_mac_accumulate", &phantom::fused_mac_accumulate,
          py::arg("ctx"), py::arg("babies"), py::arg("plaintexts"),
          py::call_guard<py::gil_scoped_release>());

    m.def("bsgs_matmul_preencoded", &phantom::bsgs_matmul_preencoded,
          py::arg("context"), py::arg("galois_key"), py::arg("x"), py::arg("diags"),
          py::call_guard<py::gil_scoped_release>());

    m.def("compute_bsgs_babies", &phantom::compute_bsgs_babies,
          py::arg("context"), py::arg("galois_key"), py::arg("x"), py::arg("baby_steps"),
          py::return_value_policy::move,
          py::call_guard<py::gil_scoped_release>());

    m.def("bsgs_apply_giants_with_babies", &phantom::bsgs_apply_giants_with_babies,
          py::arg("context"), py::arg("galois_key"), py::arg("babies"), py::arg("diags"),
          py::call_guard<py::gil_scoped_release>());

    py::enum_<phantom::ComplexFoldMode>(m, "complex_fold_mode")
        .value("Rows", phantom::ComplexFoldMode::Rows)
        .value("ColsConj", phantom::ComplexFoldMode::ColsConj);

    m.def("pre_encode_bsgs_diagonals_complex",
          [](const PhantomContext &ctx, PhantomCKKSEncoder &encoder,
             const std::vector<double> &matrix,
             std::size_t num_rows, std::size_t num_cols,
             std::size_t d_pad, std::size_t baby_steps, double scale,
             phantom::ComplexFoldMode fold_mode) {
              return phantom::pre_encode_bsgs_diagonals_complex(
                      ctx, encoder, matrix, num_rows, num_cols, d_pad, baby_steps, scale, fold_mode);
          },
          py::arg("ctx"), py::arg("encoder"), py::arg("matrix"),
          py::arg("num_rows"), py::arg("num_cols"),
          py::arg("d_pad"), py::arg("baby_steps"), py::arg("scale"),
          py::arg("fold_mode"),
          py::return_value_policy::move);

    // ===== Paterson-Stockmeyer polynomial evaluation =====
    m.def("eval_polynomial", &phantom::eval_polynomial,
          py::arg("context"), py::arg("encoder"), py::arg("relin_key"),
          py::arg("ct"), py::arg("coeffs"),
          py::call_guard<py::gil_scoped_release>());

    // ===== RMSNorm =====
    py::class_<phantom::RmsNormParams>(m, "rmsnorm_params")
            .def(py::init<>())
            .def_readwrite("d_model", &phantom::RmsNormParams::d_model)
            .def_readwrite("epsilon", &phantom::RmsNormParams::epsilon)
            .def_readwrite("z_min", &phantom::RmsNormParams::z_min)
            .def_readwrite("z_max", &phantom::RmsNormParams::z_max)
            .def_readwrite("poly_degree", &phantom::RmsNormParams::poly_degree);

    py::class_<phantom::RmsNormWeights>(m, "rmsnorm_weights")
            .def(py::init<>())
            .def(py::init([](const std::vector<double> &g_tiled_real,
                             const std::vector<double> &g) {
                     phantom::RmsNormWeights w;
                     w.g_tiled_real = g_tiled_real;
                     w.g = g;
                     return w;
                 }),
                 py::arg("g_tiled_real"), py::arg("g"))
            .def_property_readonly("g_tiled_real",
                                   [](const phantom::RmsNormWeights &w) { return w.g_tiled_real; })
            .def_property_readonly("g",
                                   [](const phantom::RmsNormWeights &w) { return w.g; });

    m.def("rmsnorm_forward", &phantom::rmsnorm_forward,
          py::arg("context"), py::arg("encoder"), py::arg("relin_key"),
          py::arg("galois_key"), py::arg("x"), py::arg("weights"), py::arg("params"),
          py::call_guard<py::gil_scoped_release>());

    // ===== Softmax =====
    m.def("ps_exp_init", &phantom::ps_exp_init,
          py::arg("context"), py::arg("encoder"), py::arg("relin_key"),
          py::arg("scores"), py::arg("num_tokens"), py::arg("num_squarings"),
          py::arg("extra_scale"),
          py::call_guard<py::gil_scoped_release>());

    m.def("square_iterations_inplace", &phantom::square_iterations_inplace,
          py::arg("context"), py::arg("relin_key"), py::arg("ct"),
          py::arg("num_squarings"),
          py::call_guard<py::gil_scoped_release>());

    m.def("square_iterations_damped_inplace", &phantom::square_iterations_damped_inplace,
          py::arg("context"), py::arg("encoder"), py::arg("relin_key"),
          py::arg("ct"), py::arg("damps"),
          py::call_guard<py::gil_scoped_release>());

    m.def("softmax_correct", &phantom::softmax_correct,
          py::arg("context"), py::arg("encoder"), py::arg("relin_key"),
          py::arg("e_ct"), py::arg("a_ct"), py::arg("iters"),
          py::call_guard<py::gil_scoped_release>());

    m.def("finalize_softmax", &phantom::finalize_softmax,
          py::arg("context"), py::arg("encoder"), py::arg("relin_key"),
          py::arg("galois_key"), py::arg("e_ct"), py::arg("num_tokens"),
          py::arg("stride"), py::arg("iters"),
          py::call_guard<py::gil_scoped_release>());

    // ===== SwiGLU MLP =====
    py::class_<phantom::MlpWeights>(m, "mlp_weights")
            .def(py::init<>())
            .def(py::init([](phantom::BsgsDiagonals w_gate,
                             phantom::BsgsDiagonals w_up,
                             phantom::BsgsDiagonals w_down) {
                     phantom::MlpWeights w;
                     w.w_gate = std::move(w_gate);
                     w.w_up = std::move(w_up);
                     w.w_down = std::move(w_down);
                     return w;
                 }),
                 py::arg("w_gate"), py::arg("w_up"), py::arg("w_down"));

    m.def("mlp_forward", &phantom::mlp_forward,
          py::arg("context"), py::arg("encoder"),
          py::arg("relin_key"), py::arg("galois_key"),
          py::arg("x"), py::arg("w"),
          py::call_guard<py::gil_scoped_release>());

    // ===== Complex-folded MLP (2x faster matmuls via complex slot packing) =====
    py::class_<phantom::ComplexBsgsDiagonals>(m, "complex_bsgs_diagonals");

    m.def("bsgs_apply_giants_with_babies_complex",
          &phantom::bsgs_apply_giants_with_babies_complex,
          py::arg("context"), py::arg("galois_key"),
          py::arg("babies"), py::arg("diags"),
          py::call_guard<py::gil_scoped_release>());

    py::class_<phantom::MlpWeightsComplex>(m, "mlp_weights_complex")
            .def(py::init<>())
            .def(py::init([](phantom::ComplexBsgsDiagonals w_gate,
                             phantom::ComplexBsgsDiagonals w_up,
                             phantom::ComplexBsgsDiagonals w_down,
                             std::size_t d_model,
                             std::size_t d_hidden,
                             std::size_t d_pad) {
                     phantom::MlpWeightsComplex w;
                     w.w_gate = std::move(w_gate);
                     w.w_up = std::move(w_up);
                     w.w_down = std::move(w_down);
                     w.d_model = d_model;
                     w.d_hidden = d_hidden;
                     w.d_pad = d_pad;
                     return w;
                 }),
                 py::arg("w_gate"), py::arg("w_up"), py::arg("w_down"),
                 py::arg("d_model"), py::arg("d_hidden"), py::arg("d_pad"))
            .def_property_readonly("d_model",
                                   [](const phantom::MlpWeightsComplex &w) { return w.d_model; })
            .def_property_readonly("d_hidden",
                                   [](const phantom::MlpWeightsComplex &w) { return w.d_hidden; })
            .def_property_readonly("d_pad",
                                   [](const phantom::MlpWeightsComplex &w) { return w.d_pad; });

    m.def("mlp_forward_complex", &phantom::mlp_forward_complex,
          py::arg("context"), py::arg("encoder"),
          py::arg("relin_key"), py::arg("galois_key"),
          py::arg("x"), py::arg("w"),
          py::call_guard<py::gil_scoped_release>());

    // ===== Attention: QK^T =====
    m.def("compute_qkt", &phantom::compute_qkt,
          py::arg("context"), py::arg("relin_key"), py::arg("galois_key"),
          py::arg("q"), py::arg("packed_k"), py::arg("d_head"),
          py::call_guard<py::gil_scoped_release>());

    // ===== Attention: score × V =====
    m.def("score_times_v", &phantom::score_times_v,
          py::arg("context"), py::arg("relin_key"), py::arg("galois_key"),
          py::arg("score_cts"), py::arg("v_cts"), py::arg("mask_pt"),
          py::arg("d_head"), py::arg("d_total"), py::arg("positions_per_ct"),
          py::call_guard<py::gil_scoped_release>());
}
