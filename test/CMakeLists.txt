add_executable(ntt_test ntt_test.cu)
target_link_libraries(ntt_test Phantom)
target_include_directories(ntt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(ntt_perf ntt_perf.cu)
target_link_libraries(ntt_perf Phantom)
target_include_directories(ntt_perf PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(hmult_perf hmult_perf.cu)
target_link_libraries(hmult_perf Phantom)
target_include_directories(hmult_perf PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(ks_matmul_perf ks_matmul_perf.cu)
target_link_libraries(ks_matmul_perf Phantom)
target_include_directories(ks_matmul_perf PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(evalmod_test evalmod_test.cu)
target_link_libraries(evalmod_test Phantom)
target_include_directories(evalmod_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(bootstrap_test bootstrap_test.cu)
target_link_libraries(bootstrap_test Phantom)
target_include_directories(bootstrap_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(ckks_engine_test ckks_engine_test.cu)
target_link_libraries(ckks_engine_test Phantom)
target_include_directories(ckks_engine_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

# find CUDAToolkit to enable cuFFT
find_package(CUDAToolkit REQUIRED)

add_executable(fft_test fft_test.cu)
target_link_libraries(fft_test PUBLIC Phantom CUDA::cufft)
target_include_directories(fft_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
