cmake_minimum_required(VERSION 3.18)
project(DiffTensorTests LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures (SM 89 and above)
set(CMAKE_CUDA_ARCHITECTURES "89;90")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

# CUDA kernel library (static)
add_library(diff_tensor_cuda STATIC
    create_diff_tensor.cu
)

target_compile_options(diff_tensor_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        --expt-relaxed-constexpr
        -Xcompiler -fPIC
    >
)

target_include_directories(diff_tensor_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# C++ wrapper library (static)
add_library(diff_tensor_wrapper STATIC
    create_diff_tensor_wrapper.cpp
)

target_link_libraries(diff_tensor_wrapper PUBLIC
    diff_tensor_cuda
    CUDA::cudart
)

target_include_directories(diff_tensor_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# GTest executable
add_executable(diff_tensor_tests
    create_diff_tensor_test.cpp
    test_main.cpp
)

target_link_libraries(diff_tensor_tests
    diff_tensor_wrapper
    diff_tensor_cuda
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
)

# Enable testing
enable_testing()
add_test(NAME DiffTensorTests COMMAND diff_tensor_tests)

# Set runtime path for finding CUDA libraries
set_target_properties(diff_tensor_tests PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
)