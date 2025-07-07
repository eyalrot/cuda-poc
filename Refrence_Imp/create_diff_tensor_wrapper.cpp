#include "create_diff_tensor.h"
#include <vector>
#include <memory>

// Wrapper class for diff tensor operations
class DiffTensorWrapper {
public:
    struct TensorDims {
        int batch;
        int channels;
        int height;
        int width;
    };

    // Allocate device memory for input tensor
    static void* allocateInputTensor(const TensorDims& dims) {
        size_t size = dims.batch * dims.channels * dims.height * dims.width * sizeof(__half);
        void* ptr = nullptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }

    // Allocate device memory for output tensor
    static void* allocateOutputTensor(const TensorDims& dims) {
        size_t size = dims.batch * dims.height * dims.width * sizeof(__half);
        void* ptr = nullptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }

    // Free device memory
    static void freeTensor(void* ptr) {
        cudaFree(ptr);
    }

    // Initialize input tensor with test data
    static void initializeInputTensor(void* d_input, const TensorDims& dims, 
                                    const std::vector<__half>& host_data) {
        size_t size = dims.batch * dims.channels * dims.height * dims.width * sizeof(__half);
        cudaMemcpy(d_input, host_data.data(), size, cudaMemcpyHostToDevice);
    }

    // Copy output tensor to host
    static void copyOutputToHost(std::vector<__half>& host_data, 
                               const void* d_output, const TensorDims& dims) {
        size_t size = dims.batch * dims.height * dims.width * sizeof(__half);
        host_data.resize(dims.batch * dims.height * dims.width);
        cudaMemcpy(host_data.data(), d_output, size, cudaMemcpyDeviceToHost);
    }

    // Execute diff tensor kernel
    static void executeDiffTensor(const void* d_input, void* d_output0, 
                                void* d_output1, void* d_output2,
                                const TensorDims& dims, cudaStream_t stream = 0) {
        launch_diff_tensors(static_cast<const __half*>(d_input),
                          static_cast<__half*>(d_output0),
                          static_cast<__half*>(d_output1),
                          static_cast<__half*>(d_output2),
                          dims.width, dims.height, dims.batch, stream);
    }
};