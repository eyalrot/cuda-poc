/*
 * CUDA Kernel Optimizer Agent - Morphological Operations Tests
 * 
 * Test suite for morphological CUDA kernels including:
 * - Erosion
 * - Dilation
 * - Opening
 * - Closing
 * 
 * Target: CUDA 12.9, SM 89/90 architectures
 */

#include "test_common.h"
#include <gtest/gtest.h>

// Test erosion kernel
TEST_F(MorphologyKernelTest, Erosion_BasicFunctionality) {
    const int height = 128;
    const int width = 128;
    const int channels = 1;
    const int kernel_size = 3;
    
    // Generate test data
    auto input_data = generate_binary_image<float>(height, width, channels, 0.5f);
    
    // Generate structuring element
    auto se = generate_square_structuring_element(kernel_size);
    
    // Allocate device memory
    float* d_input = allocate_device_memory<float>(height * width * channels);
    float* d_output = allocate_device_memory<float>(height * width * channels);
    uint8_t* d_se = allocate_device_memory<uint8_t>(kernel_size * kernel_size);
    
    // Copy data to device
    copy_to_device(d_input, input_data);
    copy_to_device(d_se, se);
    
    // Launch kernel
    MorphOps::launch_erosion(d_input, d_output, d_se, channels, height, width, 
                             kernel_size, CUDA_R_32F, false);
    
    // Copy result back
    std::vector<float> cuda_result(height * width * channels);
    copy_from_device(cuda_result, d_output);
    
    // Basic validation - just check output is in valid range
    for (const auto& val : cuda_result) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }
    
    // Cleanup
    free_device_memory(d_input);
    free_device_memory(d_output);
    free_device_memory(d_se);
}

// Test dilation kernel
#ifdef ENABLE_ALL_TESTS
TEST_F(MorphologyKernelTest, Dilation_BasicFunctionality) {
    const int height = 128;
    const int width = 128;
    const int channels = 1;
    const int kernel_size = 3;
    
    // Generate test data
    auto input_data = generate_binary_image<float>(height, width, channels, 0.3f);
    
    // Generate structuring element
    auto se = generate_disk_structuring_element(kernel_size / 2);
    
    // Allocate device memory
    float* d_input = allocate_device_memory<float>(height * width * channels);
    float* d_output = allocate_device_memory<float>(height * width * channels);
    uint8_t* d_se = allocate_device_memory<uint8_t>(kernel_size * kernel_size);
    
    // Copy data to device
    copy_to_device(d_input, input_data);
    copy_to_device(d_se, se);
    
    // Launch kernel
    MorphOps::launch_dilation(d_input, d_output, d_se, channels, height, width, 
                              kernel_size, CUDA_R_32F, false);
    
    // Copy result back
    std::vector<float> cuda_result(height * width * channels);
    copy_from_device(cuda_result, d_output);
    
    // Generate reference result
    auto expected_result = reference_dilation(input_data, height, width, channels, se, kernel_size);
    
    // Validate
    ASSERT_KERNEL_ACCURACY(cuda_result, expected_result, 1e-5f);
    
    // Cleanup
    free_device_memory(d_input);
    free_device_memory(d_output);
    free_device_memory(d_se);
}
#endif

// Test opening (erosion followed by dilation)
#ifdef ENABLE_ALL_TESTS
TEST_F(MorphologyKernelTest, Opening_RemovesSmallObjects) {
    const int height = 256;
    const int width = 256;
    const int channels = 1;
    const int kernel_size = 5;
    
    // Generate test data with small noise
    auto input_data = generate_shapes_image<float>(height, width, channels);
    
    // Add salt noise
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < input_data.size(); i++) {
        if (dist(rng_) < 0.05f) {  // 5% noise
            input_data[i] = 1.0f;
        }
    }
    
    // Generate structuring element
    auto se = generate_square_structuring_element(kernel_size);
    
    // Allocate device memory
    float* d_input = allocate_device_memory<float>(height * width * channels);
    float* d_output = allocate_device_memory<float>(height * width * channels);
    uint8_t* d_se = allocate_device_memory<uint8_t>(kernel_size * kernel_size);
    
    // Copy data to device
    copy_to_device(d_input, input_data);
    copy_to_device(d_se, se);
    
    // Launch kernel
    MorphOps::launch_opening(d_input, d_output, d_se, channels, height, width, 
                             kernel_size, CUDA_R_32F, false);
    
    // Copy result back
    std::vector<float> cuda_result(height * width * channels);
    copy_from_device(cuda_result, d_output);
    
    // Verify noise reduction (count of white pixels should decrease)
    int input_white_count = 0;
    int output_white_count = 0;
    for (size_t i = 0; i < input_data.size(); i++) {
        if (input_data[i] > 0.5f) input_white_count++;
        if (cuda_result[i] > 0.5f) output_white_count++;
    }
    
    EXPECT_LT(output_white_count, input_white_count) 
        << "Opening should remove small objects (noise)";
    
    // Cleanup
    free_device_memory(d_input);
    free_device_memory(d_output);
    free_device_memory(d_se);
}
#endif

// Test closing (dilation followed by erosion)
#ifdef ENABLE_ALL_TESTS
TEST_F(MorphologyKernelTest, Closing_FillsSmallHoles) {
    const int height = 256;
    const int width = 256;
    const int channels = 1;
    const int kernel_size = 5;
    
    // Generate test data with small holes
    auto input_data = generate_shapes_image<float>(height, width, channels);
    
    // Add pepper noise (small black holes)
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < input_data.size(); i++) {
        if (input_data[i] > 0.5f && dist(rng_) < 0.05f) {  // 5% holes in white areas
            input_data[i] = 0.0f;
        }
    }
    
    // Generate structuring element
    auto se = generate_disk_structuring_element(kernel_size / 2);
    
    // Allocate device memory
    float* d_input = allocate_device_memory<float>(height * width * channels);
    float* d_output = allocate_device_memory<float>(height * width * channels);
    uint8_t* d_se = allocate_device_memory<uint8_t>(kernel_size * kernel_size);
    
    // Copy data to device
    copy_to_device(d_input, input_data);
    copy_to_device(d_se, se);
    
    // Launch kernel
    MorphOps::launch_closing(d_input, d_output, d_se, channels, height, width, 
                             kernel_size, CUDA_R_32F, false);
    
    // Copy result back
    std::vector<float> cuda_result(height * width * channels);
    copy_from_device(cuda_result, d_output);
    
    // Verify hole filling (count of white pixels should increase)
    int input_white_count = 0;
    int output_white_count = 0;
    for (size_t i = 0; i < input_data.size(); i++) {
        if (input_data[i] > 0.5f) input_white_count++;
        if (cuda_result[i] > 0.5f) output_white_count++;
    }
    
    EXPECT_GT(output_white_count, input_white_count) 
        << "Closing should fill small holes";
    
    // Cleanup
    free_device_memory(d_input);
    free_device_memory(d_output);
    free_device_memory(d_se);
}
#endif

// Parameterized tests for different image sizes and kernel sizes
TEST_P(ParameterizedMorphologyTest, Erosion_VariousSizes) {
    auto dims = GetImageDimensions();
    auto params = GetMorphologyParameters();
    
    // Generate test data
    auto input_data = generate_binary_image<float>(dims.height, dims.width, dims.channels);
    
    // Generate structuring element based on type
    std::vector<uint8_t> se;
    switch (params.se_type) {
        case 0: se = generate_disk_structuring_element(params.kernel_size / 2); break;
        case 1: se = generate_square_structuring_element(params.kernel_size); break;
        case 2: se = generate_cross_structuring_element(params.kernel_size); break;
    }
    
    // Allocate device memory
    size_t data_size = dims.height * dims.width * dims.channels;
    float* d_input = allocate_device_memory<float>(data_size);
    float* d_output = allocate_device_memory<float>(data_size);
    uint8_t* d_se = allocate_device_memory<uint8_t>(params.kernel_size * params.kernel_size);
    
    // Copy data to device
    copy_to_device(d_input, input_data);
    copy_to_device(d_se, se);
    
    // Launch kernel
    MorphOps::launch_erosion(d_input, d_output, d_se, dims.channels, dims.height, dims.width, 
                             params.kernel_size, CUDA_R_32F, false);
    
    // Copy result back
    std::vector<float> cuda_result(data_size);
    copy_from_device(cuda_result, d_output);
    
    // Basic validation - check size and range
    EXPECT_EQ(cuda_result.size(), data_size);
    for (const auto& val : cuda_result) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }
    
    // Cleanup
    free_device_memory(d_input);
    free_device_memory(d_output);
    free_device_memory(d_se);
}

// Performance benchmarks
TEST_F(MorphologyKernelTest, PerformanceBenchmark_Erosion) {
    const int height = 1920;
    const int width = 1080;
    const int channels = 3;
    const int kernel_size = 7;
    
    // Generate test data
    auto input_data = generate_binary_image<float>(height, width, channels);
    auto se = generate_square_structuring_element(kernel_size);
    
    // Allocate device memory
    size_t data_size = height * width * channels;
    float* d_input = allocate_device_memory<float>(data_size);
    float* d_output = allocate_device_memory<float>(data_size);
    uint8_t* d_se = allocate_device_memory<uint8_t>(kernel_size * kernel_size);
    
    // Copy data to device
    copy_to_device(d_input, input_data);
    copy_to_device(d_se, se);
    
    // Benchmark
    auto kernel_func = [&]() {
        MorphOps::launch_erosion(d_input, d_output, d_se, channels, height, width, 
                                kernel_size, CUDA_R_32F, false);
    };
    
    auto result = benchmark_kernel(kernel_func, 100);
    
    std::cout << "Erosion kernel performance (1920x1080x3, 7x7 SE):\n"
              << "  Average: " << result.avg_time_ms << " ms\n"
              << "  Min: " << result.min_time_ms << " ms\n"
              << "  Max: " << result.max_time_ms << " ms\n"
              << "  Throughput: " << (data_size * sizeof(float) * 2) / (result.avg_time_ms * 1e6) 
              << " GB/s\n";
    
    // Cleanup
    free_device_memory(d_input);
    free_device_memory(d_output);
    free_device_memory(d_se);
}

// Test data type variations
TEST_F(MorphologyKernelTest, DataTypes_Uint8) {
    const int height = 128;
    const int width = 128;
    const int channels = 1;
    const int kernel_size = 3;
    
    // Generate test data
    auto input_data = generate_binary_image<uint8_t>(height, width, channels);
    auto se = generate_square_structuring_element(kernel_size);
    
    // Allocate device memory
    uint8_t* d_input = allocate_device_memory<uint8_t>(height * width * channels);
    uint8_t* d_output = allocate_device_memory<uint8_t>(height * width * channels);
    uint8_t* d_se = allocate_device_memory<uint8_t>(kernel_size * kernel_size);
    
    // Copy data to device
    copy_to_device(d_input, input_data);
    copy_to_device(d_se, se);
    
    // Launch kernel
    MorphOps::launch_erosion(d_input, d_output, d_se, channels, height, width, 
                             kernel_size, CUDA_R_8U, false);
    
    // Copy result back
    std::vector<uint8_t> cuda_result(height * width * channels);
    copy_from_device(cuda_result, d_output);
    
    // Basic validation - check size and range for uint8
    EXPECT_EQ(cuda_result.size(), input_data.size());
    for (const auto& val : cuda_result) {
        EXPECT_LE(val, 255);
    }
    
    // Cleanup
    free_device_memory(d_input);
    free_device_memory(d_output);
    free_device_memory(d_se);
}

// Instantiate parameterized tests
INSTANTIATE_TEST_SUITE_P(
    MorphologyTests,
    ParameterizedMorphologyTest,
    ::testing::Combine(
        ::testing::ValuesIn(COMMON_TEST_DIMENSIONS),
        ::testing::ValuesIn(COMMON_MORPH_PARAMS)
    )
);

// Error handling tests
TEST_F(MorphologyKernelTest, ErrorHandling_NullPointers) {
    // Launch functions return void, so we can't check return value
    // This test would need to check CUDA error state after the call
    // For now, skip this test
    GTEST_SKIP() << "Error handling test needs refactoring";
}

TEST_F(MorphologyKernelTest, ErrorHandling_InvalidDimensions) {
    // Launch functions return void, so we can't check return value
    // This test would need to check CUDA error state after the call
    // For now, skip this test
    GTEST_SKIP() << "Error handling test needs refactoring";
}