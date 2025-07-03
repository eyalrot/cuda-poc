/*
 * CUDA Kernel Optimizer Agent - Test Framework Common Header
 * 
 * Common utilities and base classes for testing CUDA kernels.
 * Provides standardized test infrastructure for all kernel types.
 * 
 * Target: CUDA 12.9, SM 89/90 architectures
 * Build: Executable tests linking to static library
 */

#pragma once

#include "../host/launchers.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

// Test configuration constants
const float DEFAULT_THRESHOLD = 1e-5f;
const int DEFAULT_ITERATIONS = 10;
const int WARMUP_ITERATIONS = 3;

// Test data types
enum class TestDataType {
    UINT8 = 0,
    UINT16 = 1,
    FLOAT32 = 2
};

// Test result structure
struct TestResult {
    bool passed;
    float max_error;
    float avg_error;
    double execution_time_ms;
    std::string error_message;
    
    TestResult() : passed(false), max_error(0.0f), avg_error(0.0f), 
                   execution_time_ms(0.0), error_message("") {}
};

// Performance benchmark result
struct BenchmarkResult {
    double min_time_ms;
    double max_time_ms;
    double avg_time_ms;
    double stddev_ms;
    size_t iterations;
    
    BenchmarkResult() : min_time_ms(0.0), max_time_ms(0.0), avg_time_ms(0.0),
                        stddev_ms(0.0), iterations(0) {}
};

// Base test class for all kernel tests
class CudaKernelTest : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;
    
    // Test data generation
    template<typename T>
    std::vector<T> generate_random_data(size_t size, T min_val, T max_val);
    
    template<typename T>
    std::vector<T> generate_gradient_data(int height, int width, int channels);
    
    template<typename T>
    std::vector<T> generate_checkerboard_data(int height, int width, int channels);
    
    // Test data loading/saving
    template<typename T>
    bool load_test_data(const std::string& filename, std::vector<T>& data, 
                       BinaryHeader& header);
    
    template<typename T>
    bool save_test_data(const std::string& filename, const std::vector<T>& data,
                       const BinaryHeader& header);
    
    // Error calculation
    template<typename T>
    float calculate_max_error(const std::vector<T>& result, 
                             const std::vector<T>& expected);
    
    template<typename T>
    float calculate_avg_error(const std::vector<T>& result, 
                             const std::vector<T>& expected);
    
    template<typename T>
    float calculate_relative_error(const std::vector<T>& result, 
                                  const std::vector<T>& expected);
    
    // Performance benchmarking
    template<typename KernelFunc>
    BenchmarkResult benchmark_kernel(KernelFunc kernel_func, 
                                   size_t iterations = DEFAULT_ITERATIONS);
    
    // Memory management helpers
    template<typename T>
    T* allocate_device_memory(size_t size);
    
    template<typename T>
    void free_device_memory(T* ptr);
    
    template<typename T>
    void copy_to_device(T* d_ptr, const std::vector<T>& h_data);
    
    template<typename T>
    void copy_from_device(std::vector<T>& h_data, const T* d_ptr);
    
    // Test validation
    template<typename T>
    TestResult validate_kernel_output(
        const std::vector<T>& cuda_result,
        const std::vector<T>& expected_result,
        float threshold = DEFAULT_THRESHOLD
    );
    
    // Device information
    void print_device_info();
    bool check_device_capability(int major, int minor);
    
    // Random number generator
    std::mt19937 rng_;
    
private:
    // Device properties
    cudaDeviceProp device_prop_;
    bool device_initialized_;
};

// Specialized test class for filter operations
class FilterKernelTest : public CudaKernelTest {
protected:
    void SetUp() override;
    
    // Filter-specific test data
    template<typename T>
    std::vector<T> generate_noisy_image(int height, int width, int channels, 
                                       float noise_level = 0.1f);
    
    template<typename T>
    std::vector<T> generate_edge_image(int height, int width, int channels);
    
    // Reference implementations for validation
    template<typename T>
    std::vector<T> reference_gaussian_blur(const std::vector<T>& input,
                                          int height, int width, int channels,
                                          float sigma, int kernel_size);
    
    template<typename T>
    std::vector<T> reference_box_filter(const std::vector<T>& input,
                                       int height, int width, int channels,
                                       int kernel_size);
    
    template<typename T>
    std::vector<T> reference_median_filter(const std::vector<T>& input,
                                          int height, int width, int channels,
                                          int kernel_size);
    
    template<typename T>
    std::vector<T> reference_bilateral_filter(const std::vector<T>& input,
                                             int height, int width, int channels,
                                             float sigma_space, float sigma_color,
                                             int kernel_size);
};

// Specialized test class for morphological operations
class MorphologyKernelTest : public CudaKernelTest {
protected:
    void SetUp() override;
    
    // Morphology-specific test data
    template<typename T>
    std::vector<T> generate_binary_image(int height, int width, int channels,
                                        float object_ratio = 0.3f);
    
    template<typename T>
    std::vector<T> generate_shapes_image(int height, int width, int channels);
    
    // Structuring element generation
    std::vector<uint8_t> generate_disk_structuring_element(int radius);
    std::vector<uint8_t> generate_square_structuring_element(int size);
    std::vector<uint8_t> generate_cross_structuring_element(int size);
    
    // Reference implementations
    template<typename T>
    std::vector<T> reference_erosion(const std::vector<T>& input,
                                    int height, int width, int channels,
                                    const std::vector<uint8_t>& se, int se_size);
    
    template<typename T>
    std::vector<T> reference_dilation(const std::vector<T>& input,
                                     int height, int width, int channels,
                                     const std::vector<uint8_t>& se, int se_size);
};

// Specialized test class for fused operations
class FusedKernelTest : public CudaKernelTest {
protected:
    void SetUp() override;
    
    // Fused operation testing
    template<typename T>
    TestResult test_fused_vs_separate(
        const std::vector<T>& input,
        int height, int width, int channels,
        float threshold = DEFAULT_THRESHOLD
    );
    
    // Performance comparison
    template<typename T>
    void compare_fused_vs_separate_performance(
        const std::vector<T>& input,
        int height, int width, int channels,
        size_t iterations = DEFAULT_ITERATIONS
    );
};

// Test parameter structures
struct ImageDimensions {
    int height;
    int width;
    int channels;
    
    ImageDimensions(int h, int w, int c) : height(h), width(w), channels(c) {}
};

struct FilterParameters {
    float sigma;
    int kernel_size;
    float threshold;
    
    FilterParameters(float s, int k, float t = 0.0f) 
        : sigma(s), kernel_size(k), threshold(t) {}
};

struct MorphologyParameters {
    int kernel_size;
    int se_type; // 0=disk, 1=square, 2=cross
    
    MorphologyParameters(int k, int t) : kernel_size(k), se_type(t) {}
};

// Parameterized test helpers
class ParameterizedImageTest : public CudaKernelTest,
                              public ::testing::WithParamInterface<ImageDimensions> {
protected:
    ImageDimensions GetImageDimensions() const { return GetParam(); }
};

class ParameterizedFilterTest : public FilterKernelTest,
                               public ::testing::WithParamInterface<
                                   std::tuple<ImageDimensions, FilterParameters>> {
protected:
    ImageDimensions GetImageDimensions() const { return std::get<0>(GetParam()); }
    FilterParameters GetFilterParameters() const { return std::get<1>(GetParam()); }
};

class ParameterizedMorphologyTest : public MorphologyKernelTest,
                                   public ::testing::WithParamInterface<
                                       std::tuple<ImageDimensions, MorphologyParameters>> {
protected:
    ImageDimensions GetImageDimensions() const { return std::get<0>(GetParam()); }
    MorphologyParameters GetMorphologyParameters() const { return std::get<1>(GetParam()); }
};

// Utility macros for test assertions
#define ASSERT_CUDA_SUCCESS(call) \
    do { \
        cudaError_t error = call; \
        ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error); \
    } while(0)

#define EXPECT_CUDA_SUCCESS(call) \
    do { \
        cudaError_t error = call; \
        EXPECT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error); \
    } while(0)

#define ASSERT_KERNEL_ACCURACY(result, expected, threshold) \
    do { \
        TestResult test_result = validate_kernel_output(result, expected, threshold); \
        ASSERT_TRUE(test_result.passed) \
            << "Kernel accuracy test failed: " << test_result.error_message \
            << " (max error: " << test_result.max_error << ")"; \
    } while(0)

#define EXPECT_KERNEL_ACCURACY(result, expected, threshold) \
    do { \
        TestResult test_result = validate_kernel_output(result, expected, threshold); \
        EXPECT_TRUE(test_result.passed) \
            << "Kernel accuracy test failed: " << test_result.error_message \
            << " (max error: " << test_result.max_error << ")"; \
    } while(0)

// Test data file paths
const std::string TEST_DATA_DIR = "test_data/";
const std::string TEST_INPUT_DIR = TEST_DATA_DIR + "input/";
const std::string TEST_EXPECTED_DIR = TEST_DATA_DIR + "expected/";
const std::string TEST_OUTPUT_DIR = TEST_DATA_DIR + "output/";

// Common test dimensions
const std::vector<ImageDimensions> COMMON_TEST_DIMENSIONS = {
    ImageDimensions(64, 64, 1),      // Small square, single channel
    ImageDimensions(128, 128, 3),    // Medium square, RGB
    ImageDimensions(256, 256, 1),    // Large square, single channel
    ImageDimensions(512, 512, 4),    // Large square, RGBA
    ImageDimensions(720, 1280, 3),   // HD dimensions
    ImageDimensions(100, 150, 1),    // Non-square dimensions
    ImageDimensions(33, 17, 5)       // Odd dimensions, 5 channels
};

// Common filter parameters
const std::vector<FilterParameters> COMMON_FILTER_PARAMS = {
    FilterParameters(1.0f, 5),       // Light blur
    FilterParameters(2.0f, 9),       // Medium blur
    FilterParameters(3.0f, 13),      // Heavy blur
    FilterParameters(0.5f, 3),       // Minimal blur
    FilterParameters(4.0f, 17)       // Very heavy blur
};

// Common morphology parameters
const std::vector<MorphologyParameters> COMMON_MORPH_PARAMS = {
    MorphologyParameters(3, 0),      // Small disk
    MorphologyParameters(5, 1),      // Medium square
    MorphologyParameters(7, 2),      // Large cross
    MorphologyParameters(9, 0),      // Large disk
    MorphologyParameters(11, 1)      // Very large square
};

// Template implementations
template<typename KernelFunc>
inline BenchmarkResult CudaKernelTest::benchmark_kernel(KernelFunc kernel_func, 
                                               size_t iterations) {
    BenchmarkResult result;
    result.iterations = iterations;
    
    // Warm-up runs
    for (int i = 0; i < 5; ++i) {
        kernel_func();
    }
    cudaDeviceSynchronize();
    
    // Timing runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::vector<float> times;
    times.reserve(iterations);
    
    for (size_t i = 0; i < iterations; ++i) {
        cudaEventRecord(start);
        kernel_func();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        times.push_back(elapsed_ms);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Calculate statistics
    result.min_time_ms = *std::min_element(times.begin(), times.end());
    result.max_time_ms = *std::max_element(times.begin(), times.end());
    result.avg_time_ms = std::accumulate(times.begin(), times.end(), 0.0f) / iterations;
    
    // Calculate standard deviation
    float variance = 0;
    for (float time : times) {
        variance += (time - result.avg_time_ms) * (time - result.avg_time_ms);
    }
    result.stddev_ms = std::sqrt(variance / iterations);
    
    return result;
}