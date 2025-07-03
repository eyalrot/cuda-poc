/*
 * CUDA Kernel Optimizer Agent - Filter Kernel Tests
 * 
 * Comprehensive tests for filter kernels including:
 * - Gaussian blur
 * - Box filter
 * - Median filter
 * - Bilateral filter
 * 
 * Tests cover accuracy, performance, and edge cases.
 * Build: Executable tests linking to static library
 */

#include "test_common.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cmath>

class FilterKernelTestSuite : public FilterKernelTest {
protected:
    void SetUp() override {
        FilterKernelTest::SetUp();
        
        // Initialize test data
        test_height_ = 128;
        test_width_ = 128;
        test_channels_ = 3;
        
        // Generate test input data
        input_data_float_ = generate_noisy_image<float>(
            test_height_, test_width_, test_channels_, 0.1f);
        input_data_uint8_ = generate_noisy_image<uint8_t>(
            test_height_, test_width_, test_channels_, 0.1f);
        
        // Allocate device memory
        d_input_float_ = allocate_device_memory<float>(input_data_float_.size());
        d_output_float_ = allocate_device_memory<float>(input_data_float_.size());
        d_input_uint8_ = allocate_device_memory<uint8_t>(input_data_uint8_.size());
        d_output_uint8_ = allocate_device_memory<uint8_t>(input_data_uint8_.size());
        
        // Copy input data to device
        copy_to_device(d_input_float_, input_data_float_);
        copy_to_device(d_input_uint8_, input_data_uint8_);
    }
    
    void TearDown() override {
        free_device_memory(d_input_float_);
        free_device_memory(d_output_float_);
        free_device_memory(d_input_uint8_);
        free_device_memory(d_output_uint8_);
        
        FilterKernelTest::TearDown();
    }
    
    // Test data
    int test_height_, test_width_, test_channels_;
    std::vector<float> input_data_float_;
    std::vector<uint8_t> input_data_uint8_;
    
    // Device pointers
    float* d_input_float_;
    float* d_output_float_;
    uint8_t* d_input_uint8_;
    uint8_t* d_output_uint8_;
};

// Test Gaussian blur kernel accuracy
TEST_F(FilterKernelTestSuite, GaussianBlurAccuracy) {
    const float sigma = 2.0f;
    const int kernel_size = 9;
    const float threshold = 1e-4f;
    
    // Test with float data
    FilterOps::launch_gaussian_blur<float>(
        d_input_float_, d_output_float_,
        test_channels_, test_height_, test_width_,
        sigma, kernel_size, false
    );
    
    std::vector<float> cuda_result(input_data_float_.size());
    copy_from_device(cuda_result, d_output_float_);
    
    // Generate reference result
    std::vector<float> expected_result = reference_gaussian_blur<float>(
        input_data_float_, test_height_, test_width_, test_channels_,
        sigma, kernel_size
    );
    
    ASSERT_KERNEL_ACCURACY(cuda_result, expected_result, threshold);
}

// Test box filter kernel accuracy
TEST_F(FilterKernelTestSuite, BoxFilterAccuracy) {
    const int kernel_size = 5;
    const float threshold = 1e-5f;
    
    // Test with uint8 data
    FilterOps::launch_box_filter<uint8_t>(
        d_input_uint8_, d_output_uint8_,
        test_channels_, test_height_, test_width_,
        kernel_size, false
    );
    
    std::vector<uint8_t> cuda_result(input_data_uint8_.size());
    copy_from_device(cuda_result, d_output_uint8_);
    
    // Generate reference result
    std::vector<uint8_t> expected_result = reference_box_filter<uint8_t>(
        input_data_uint8_, test_height_, test_width_, test_channels_,
        kernel_size
    );
    
    ASSERT_KERNEL_ACCURACY(cuda_result, expected_result, threshold);
}

// Test median filter kernel accuracy
TEST_F(FilterKernelTestSuite, MedianFilterAccuracy) {
    const int kernel_size = 5;
    const float threshold = 1e-5f;
    
    // Test with float data
    FilterOps::launch_median_filter<float>(
        d_input_float_, d_output_float_,
        test_channels_, test_height_, test_width_,
        kernel_size, false
    );
    
    std::vector<float> cuda_result(input_data_float_.size());
    copy_from_device(cuda_result, d_output_float_);
    
    // Generate reference result
    std::vector<float> expected_result = reference_median_filter<float>(
        input_data_float_, test_height_, test_width_, test_channels_,
        kernel_size
    );
    
    ASSERT_KERNEL_ACCURACY(cuda_result, expected_result, threshold);
}

// Test bilateral filter kernel accuracy
TEST_F(FilterKernelTestSuite, BilateralFilterAccuracy) {
    const float sigma_space = 2.0f;
    const float sigma_color = 0.1f;
    const int kernel_size = 7;
    const float threshold = 1e-3f;
    
    // Test with float data
    FilterOps::launch_bilateral_filter<float>(
        d_input_float_, d_output_float_,
        test_channels_, test_height_, test_width_,
        sigma_space, sigma_color, kernel_size, false
    );
    
    std::vector<float> cuda_result(input_data_float_.size());
    copy_from_device(cuda_result, d_output_float_);
    
    // Generate reference result
    std::vector<float> expected_result = reference_bilateral_filter<float>(
        input_data_float_, test_height_, test_width_, test_channels_,
        sigma_space, sigma_color, kernel_size
    );
    
    ASSERT_KERNEL_ACCURACY(cuda_result, expected_result, threshold);
}

// Performance benchmarks for filter kernels
TEST_F(FilterKernelTestSuite, GaussianBlurPerformance) {
    const float sigma = 2.0f;
    const int kernel_size = 9;
    const size_t iterations = 100;
    
    auto kernel_func = [&]() {
        FilterOps::launch_gaussian_blur<float>(
            d_input_float_, d_output_float_,
            test_channels_, test_height_, test_width_,
            sigma, kernel_size, false
        );
    };
    
    BenchmarkResult result = benchmark_kernel(kernel_func, iterations);
    
    std::cout << "Gaussian Blur Performance:" << std::endl;
    std::cout << "  Average time: " << result.avg_time_ms << " ms" << std::endl;
    std::cout << "  Min time: " << result.min_time_ms << " ms" << std::endl;
    std::cout << "  Max time: " << result.max_time_ms << " ms" << std::endl;
    std::cout << "  Std dev: " << result.stddev_ms << " ms" << std::endl;
    
    // Performance expectations (adjust based on hardware)
    EXPECT_LT(result.avg_time_ms, 5.0) << "Gaussian blur too slow";
}

TEST_F(FilterKernelTestSuite, BoxFilterPerformance) {
    const int kernel_size = 5;
    const size_t iterations = 100;
    
    auto kernel_func = [&]() {
        FilterOps::launch_box_filter<uint8_t>(
            d_input_uint8_, d_output_uint8_,
            test_channels_, test_height_, test_width_,
            kernel_size, false
        );
    };
    
    BenchmarkResult result = benchmark_kernel(kernel_func, iterations);
    
    std::cout << "Box Filter Performance:" << std::endl;
    std::cout << "  Average time: " << result.avg_time_ms << " ms" << std::endl;
    
    // Box filter should be faster than Gaussian blur
    EXPECT_LT(result.avg_time_ms, 3.0) << "Box filter too slow";
}

// Parameterized tests for different image dimensions
INSTANTIATE_TEST_SUITE_P(
    DifferentDimensions,
    ParameterizedFilterTest,
    ::testing::Combine(
        ::testing::ValuesIn(COMMON_TEST_DIMENSIONS),
        ::testing::ValuesIn(COMMON_FILTER_PARAMS)
    )
);

TEST_P(ParameterizedFilterTest, GaussianBlurParameterized) {
    auto dims = GetImageDimensions();
    auto params = GetFilterParameters();
    
    // Generate test data
    std::vector<float> input_data = generate_noisy_image<float>(
        dims.height, dims.width, dims.channels, 0.1f);
    
    // Allocate device memory
    float* d_input = allocate_device_memory<float>(input_data.size());
    float* d_output = allocate_device_memory<float>(input_data.size());
    
    copy_to_device(d_input, input_data);
    
    // Run kernel
    FilterOps::launch_gaussian_blur<float>(
        d_input, d_output,
        dims.channels, dims.height, dims.width,
        params.sigma, params.kernel_size, false
    );
    
    std::vector<float> cuda_result(input_data.size());
    copy_from_device(cuda_result, d_output);
    
    // Generate reference result
    std::vector<float> expected_result = reference_gaussian_blur<float>(
        input_data, dims.height, dims.width, dims.channels,
        params.sigma, params.kernel_size
    );
    
    ASSERT_KERNEL_ACCURACY(cuda_result, expected_result, 1e-4f);
    
    free_device_memory(d_input);
    free_device_memory(d_output);
}

// Edge case tests
TEST_F(FilterKernelTestSuite, EdgeCases) {
    // Test with very small images
    const int small_height = 3;
    const int small_width = 3;
    const int small_channels = 1;
    
    std::vector<float> small_input = generate_random_data<float>(
        small_height * small_width * small_channels, 0.0f, 1.0f);
    
    float* d_small_input = allocate_device_memory<float>(small_input.size());
    float* d_small_output = allocate_device_memory<float>(small_input.size());
    
    copy_to_device(d_small_input, small_input);
    
    // Test with kernel size larger than image
    const int large_kernel_size = 5;
    const float sigma = 1.0f;
    
    // This should not crash
    EXPECT_NO_THROW({
        FilterOps::launch_gaussian_blur<float>(
            d_small_input, d_small_output,
            small_channels, small_height, small_width,
            sigma, large_kernel_size, false
        );
    });
    
    free_device_memory(d_small_input);
    free_device_memory(d_small_output);
}

// Test boundary conditions
TEST_F(FilterKernelTestSuite, BoundaryConditions) {
    // Create image with known pattern at edges
    std::vector<float> edge_input = generate_checkerboard_data<float>(
        test_height_, test_width_, test_channels_);
    
    float* d_edge_input = allocate_device_memory<float>(edge_input.size());
    float* d_edge_output = allocate_device_memory<float>(edge_input.size());
    
    copy_to_device(d_edge_input, edge_input);
    
    const float sigma = 1.0f;
    const int kernel_size = 5;
    
    FilterOps::launch_gaussian_blur<float>(
        d_edge_input, d_edge_output,
        test_channels_, test_height_, test_width_,
        sigma, kernel_size, false
    );
    
    std::vector<float> cuda_result(edge_input.size());
    copy_from_device(cuda_result, d_edge_output);
    
    // Verify that edge pixels are handled correctly
    // Check that no NaN or infinite values are produced
    for (size_t i = 0; i < cuda_result.size(); ++i) {
        EXPECT_TRUE(std::isfinite(cuda_result[i])) 
            << "Non-finite value at index " << i;
    }
    
    free_device_memory(d_edge_input);
    free_device_memory(d_edge_output);
}

// Test with different data types
TEST_F(FilterKernelTestSuite, DataTypeConsistency) {
    const float sigma = 1.0f;
    const int kernel_size = 5;
    
    // Test with uint16_t
    std::vector<uint16_t> input_uint16 = generate_noisy_image<uint16_t>(
        test_height_, test_width_, test_channels_, 0.1f);
    
    uint16_t* d_input_uint16 = allocate_device_memory<uint16_t>(input_uint16.size());
    uint16_t* d_output_uint16 = allocate_device_memory<uint16_t>(input_uint16.size());
    
    copy_to_device(d_input_uint16, input_uint16);
    
    FilterOps::launch_gaussian_blur<uint16_t>(
        d_input_uint16, d_output_uint16,
        test_channels_, test_height_, test_width_,
        sigma, kernel_size, false
    );
    
    std::vector<uint16_t> cuda_result(input_uint16.size());
    copy_from_device(cuda_result, d_output_uint16);
    
    // Verify that output is within expected range
    for (size_t i = 0; i < cuda_result.size(); ++i) {
        EXPECT_LE(cuda_result[i], 65535) << "Value out of range at index " << i;
    }
    
    free_device_memory(d_input_uint16);
    free_device_memory(d_output_uint16);
}

// Memory bandwidth utilization test
TEST_F(FilterKernelTestSuite, MemoryBandwidthUtilization) {
    const float sigma = 2.0f;
    const int kernel_size = 9;
    const size_t iterations = 50;
    
    // Large image for bandwidth testing
    const int large_height = 2048;
    const int large_width = 2048;
    const int large_channels = 3;
    
    std::vector<float> large_input = generate_random_data<float>(
        large_height * large_width * large_channels, 0.0f, 1.0f);
    
    float* d_large_input = allocate_device_memory<float>(large_input.size());
    float* d_large_output = allocate_device_memory<float>(large_input.size());
    
    copy_to_device(d_large_input, large_input);
    
    auto kernel_func = [&]() {
        FilterOps::launch_gaussian_blur<float>(
            d_large_input, d_large_output,
            large_channels, large_height, large_width,
            sigma, kernel_size, false
        );
    };
    
    BenchmarkResult result = benchmark_kernel(kernel_func, iterations);
    
    // Calculate theoretical memory bandwidth
    size_t bytes_per_kernel = large_input.size() * sizeof(float) * 2; // read + write
    double bandwidth_gb_s = (bytes_per_kernel * 1e-9) / (result.avg_time_ms * 1e-3);
    
    std::cout << "Memory bandwidth utilization: " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // Expect reasonable bandwidth utilization (adjust based on hardware)
    EXPECT_GT(bandwidth_gb_s, 100.0) << "Low memory bandwidth utilization";
    
    free_device_memory(d_large_input);
    free_device_memory(d_large_output);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}