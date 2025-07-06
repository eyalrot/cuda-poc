cmake_minimum_required(VERSION 3.18)
project(cuda_gaussian_threshold LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architectures (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES "89;90")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS "-O3 -use_fast_math --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math --ptxas-options=-v")

# Create library for the Gaussian threshold kernel
add_library(gaussian_threshold_kernel STATIC
    kernels/gaussian_threshold_kernel.cu
    host/gaussian_threshold_launcher.cu
)

# Set properties for the library
target_include_directories(gaussian_threshold_kernel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/host
)

# Link CUDA runtime
target_link_libraries(gaussian_threshold_kernel PUBLIC
    CUDA::cudart
)

# Create test executable
add_executable(test_gaussian_threshold
    tests/test_gaussian_threshold.cpp
)

# Link test with kernel library
target_link_libraries(test_gaussian_threshold PRIVATE
    gaussian_threshold_kernel
)

# Enable testing
enable_testing()
add_test(NAME GaussianThresholdTest COMMAND test_gaussian_threshold)