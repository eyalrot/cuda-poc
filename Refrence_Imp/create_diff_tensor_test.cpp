#include <gtest/gtest.h>
#include "create_diff_tensor.h"
#include "create_diff_tensor_wrapper.cpp"
#include <vector>
#include <cmath>
#include <random>

class DiffTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        cudaSetDevice(0);
    }

    void TearDown() override {
        // Reset CUDA device
        cudaDeviceReset();
    }

    // Helper function to generate random half precision values
    std::vector<__half> generateRandomInput(int batch, int channels, int height, int width) {
        std::vector<__half> data(batch * channels * height * width);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = __float2half(dis(gen));
        }
        return data;
    }

    // Helper function to verify output correctness
    bool verifyOutput(const std::vector<__half>& input,
                     const std::vector<__half>& output0,
                     const std::vector<__half>& output1,
                     const std::vector<__half>& output2,
                     int batch, int channels, int height, int width,
                     float tolerance = 1e-3f) {
        int image_size = height * width;
        
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < image_size; ++i) {
                int base = b * channels * image_size;
                int out_idx = b * image_size + i;
                
                // Get input values
                float in0 = __half2float(input[base + 0 * image_size + i]);
                float in1 = __half2float(input[base + 1 * image_size + i]);
                float in2 = __half2float(input[base + 2 * image_size + i]);
                
                // Calculate expected values
                float expected0 = in0 - in1;
                float expected1 = in0 - in2;
                float expected2 = in1 - in2;
                
                // Get actual values
                float actual0 = __half2float(output0[out_idx]);
                float actual1 = __half2float(output1[out_idx]);
                float actual2 = __half2float(output2[out_idx]);
                
                // Check tolerance
                if (std::abs(expected0 - actual0) > tolerance ||
                    std::abs(expected1 - actual1) > tolerance ||
                    std::abs(expected2 - actual2) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
};

// Test basic functionality with small input
TEST_F(DiffTensorTest, BasicFunctionality) {
    DiffTensorWrapper::TensorDims dims = {2, 3, 4, 4}; // Small test case
    
    // Generate random input
    auto input_data = generateRandomInput(dims.batch, dims.channels, dims.height, dims.width);
    
    // Allocate device memory
    void* d_input = DiffTensorWrapper::allocateInputTensor(dims);
    void* d_output0 = DiffTensorWrapper::allocateOutputTensor(dims);
    void* d_output1 = DiffTensorWrapper::allocateOutputTensor(dims);
    void* d_output2 = DiffTensorWrapper::allocateOutputTensor(dims);
    
    // Initialize input
    DiffTensorWrapper::initializeInputTensor(d_input, dims, input_data);
    
    // Execute kernel
    DiffTensorWrapper::executeDiffTensor(d_input, d_output0, d_output1, d_output2, dims);
    
    // Copy outputs to host
    std::vector<__half> output0, output1, output2;
    DiffTensorWrapper::copyOutputToHost(output0, d_output0, dims);
    DiffTensorWrapper::copyOutputToHost(output1, d_output1, dims);
    DiffTensorWrapper::copyOutputToHost(output2, d_output2, dims);
    
    // Verify results
    ASSERT_TRUE(verifyOutput(input_data, output0, output1, output2, 
                           dims.batch, dims.channels, dims.height, dims.width));
    
    // Cleanup
    DiffTensorWrapper::freeTensor(d_input);
    DiffTensorWrapper::freeTensor(d_output0);
    DiffTensorWrapper::freeTensor(d_output1);
    DiffTensorWrapper::freeTensor(d_output2);
}

// Test with larger, more realistic dimensions
TEST_F(DiffTensorTest, LargeDimensions) {
    DiffTensorWrapper::TensorDims dims = {100, 3, 1008, 176};
    
    // Allocate device memory
    void* d_input = DiffTensorWrapper::allocateInputTensor(dims);
    void* d_output0 = DiffTensorWrapper::allocateOutputTensor(dims);
    void* d_output1 = DiffTensorWrapper::allocateOutputTensor(dims);
    void* d_output2 = DiffTensorWrapper::allocateOutputTensor(dims);
    
    ASSERT_NE(d_input, nullptr);
    ASSERT_NE(d_output0, nullptr);
    ASSERT_NE(d_output1, nullptr);
    ASSERT_NE(d_output2, nullptr);
    
    // Initialize with zeros
    size_t input_size = dims.batch * dims.channels * dims.height * dims.width * sizeof(__half);
    cudaMemset(d_input, 0, input_size);
    
    // Execute kernel
    DiffTensorWrapper::executeDiffTensor(d_input, d_output0, d_output1, d_output2, dims);
    
    // Verify no CUDA errors
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    // Cleanup
    DiffTensorWrapper::freeTensor(d_input);
    DiffTensorWrapper::freeTensor(d_output0);
    DiffTensorWrapper::freeTensor(d_output1);
    DiffTensorWrapper::freeTensor(d_output2);
}

// Test performance benchmarks
TEST_F(DiffTensorTest, PerformanceBenchmark) {
    // Test batch size 100
    benchmark(100, 176, 1008);
    SUCCEED(); // Benchmark test always passes if it completes
    
    // Test batch size 1000
    benchmark(1000, 176, 1008);
    SUCCEED();
}

// Test edge cases
TEST_F(DiffTensorTest, EdgeCases) {
    // Test minimum dimensions
    DiffTensorWrapper::TensorDims dims = {1, 3, 1, 1};
    
    std::vector<__half> input_data = {
        __float2half(0.5f), __float2half(0.3f), __float2half(0.2f)
    };
    
    void* d_input = DiffTensorWrapper::allocateInputTensor(dims);
    void* d_output0 = DiffTensorWrapper::allocateOutputTensor(dims);
    void* d_output1 = DiffTensorWrapper::allocateOutputTensor(dims);
    void* d_output2 = DiffTensorWrapper::allocateOutputTensor(dims);
    
    DiffTensorWrapper::initializeInputTensor(d_input, dims, input_data);
    DiffTensorWrapper::executeDiffTensor(d_input, d_output0, d_output1, d_output2, dims);
    
    std::vector<__half> output0, output1, output2;
    DiffTensorWrapper::copyOutputToHost(output0, d_output0, dims);
    DiffTensorWrapper::copyOutputToHost(output1, d_output1, dims);
    DiffTensorWrapper::copyOutputToHost(output2, d_output2, dims);
    
    // Verify specific values
    EXPECT_NEAR(__half2float(output0[0]), 0.2f, 1e-3f); // 0.5 - 0.3
    EXPECT_NEAR(__half2float(output1[0]), 0.3f, 1e-3f); // 0.5 - 0.2
    EXPECT_NEAR(__half2float(output2[0]), 0.1f, 1e-3f); // 0.3 - 0.2
    
    DiffTensorWrapper::freeTensor(d_input);
    DiffTensorWrapper::freeTensor(d_output0);
    DiffTensorWrapper::freeTensor(d_output1);
    DiffTensorWrapper::freeTensor(d_output2);
}

// Test with streams
TEST_F(DiffTensorTest, StreamExecution) {
    DiffTensorWrapper::TensorDims dims = {10, 3, 100, 100};
    
    // Create stream
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    
    void* d_input = DiffTensorWrapper::allocateInputTensor(dims);
    void* d_output0 = DiffTensorWrapper::allocateOutputTensor(dims);
    void* d_output1 = DiffTensorWrapper::allocateOutputTensor(dims);
    void* d_output2 = DiffTensorWrapper::allocateOutputTensor(dims);
    
    // Execute with stream
    DiffTensorWrapper::executeDiffTensor(d_input, d_output0, d_output1, d_output2, dims, stream);
    
    // Synchronize stream
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    
    // Cleanup
    DiffTensorWrapper::freeTensor(d_input);
    DiffTensorWrapper::freeTensor(d_output0);
    DiffTensorWrapper::freeTensor(d_output1);
    DiffTensorWrapper::freeTensor(d_output2);
    cudaStreamDestroy(stream);
}