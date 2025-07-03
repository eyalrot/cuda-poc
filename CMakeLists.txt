cmake_minimum_required(VERSION 3.18)
project(cuda_kernel_optimizer LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture settings for SM 89/90 (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES "89;90")

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "-O3 -use_fast_math --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math --ptxas-options=-v")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/host)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Source files for the static library
set(KERNEL_SOURCES
    kernels/filters.cu
    kernels/fused_ops.cu
    kernels/morphology.cu
)

set(HOST_SOURCES
    host/launchers.cu
)

# Create static library from CUDA kernels and host code
add_library(cuda_image_kernels STATIC
    ${KERNEL_SOURCES}
    ${HOST_SOURCES}
)

# Set properties for the static library
set_target_properties(cuda_image_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Specify target compile options
target_compile_options(cuda_image_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        --ptxas-options=-v
        --extended-lambda
        --expt-relaxed-constexpr
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
        -mtune=native
    >
)

# Find GTest (optional)
find_package(GTest QUIET)

if(GTest_FOUND)
    message(STATUS "GTest found, building tests")
    
    # Test executables
    add_executable(test_filters tests/test_filters.cpp tests/test_common.cpp)
    target_link_libraries(test_filters cuda_image_kernels GTest::gtest GTest::gtest_main)
    
    add_executable(test_fused_ops tests/test_fused_ops.cpp tests/test_common.cpp)
    target_link_libraries(test_fused_ops cuda_image_kernels GTest::gtest GTest::gtest_main)
    
    add_executable(test_morphology tests/test_morphology.cpp tests/test_common.cpp)
    target_link_libraries(test_morphology cuda_image_kernels GTest::gtest GTest::gtest_main)
    
    # Enable testing
    enable_testing()
    add_test(NAME FilterTests COMMAND test_filters)
    add_test(NAME FusedOpsTests COMMAND test_fused_ops)
    add_test(NAME MorphologyTests COMMAND test_morphology)
else()
    message(STATUS "GTest not found, skipping tests")
    message(STATUS "To enable tests, install GTest:")
    message(STATUS "  sudo apt-get install libgtest-dev")
    message(STATUS "  cd /usr/src/gtest && sudo cmake . && sudo make && sudo cp lib/*.a /usr/lib")
endif()

# Debug information
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

# Simple test example
add_executable(simple_test examples/simple_test.cpp)
target_link_libraries(simple_test cuda_image_kernels)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set runtime output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)