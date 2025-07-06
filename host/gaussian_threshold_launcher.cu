/*
 * Host launcher for Gaussian threshold kernel
 * Provides C++ interface for the optimized CUDA kernel
 */

#include "gaussian_threshold_launcher.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Forward declarations of kernels
__global__ void gaussian_threshold_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
);

__global__ void gaussian_threshold_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
);

namespace cuda_image {

void launch_gaussian_threshold(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    bool use_shared_memory
) {
    // Configure kernel launch parameters
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    if (use_shared_memory) {
        // Calculate shared memory size
        const int RADIUS = 2;
        const int shared_w = BLOCK_SIZE + 2 * RADIUS;
        const int shared_h = BLOCK_SIZE + 2 * RADIUS;
        size_t shared_mem_size = shared_w * shared_h * sizeof(float);
        
        gaussian_threshold_shared_kernel<<<grid, block, shared_mem_size>>>(
            d_input, d_output, width, height
        );
    } else {
        gaussian_threshold_kernel<<<grid, block>>>(
            d_input, d_output, width, height
        );
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void process_image_cuda(
    const float* h_input,
    float* h_output,
    int width,
    int height,
    bool use_shared_memory
) {
    const size_t size = width * height * sizeof(float);
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    launch_gaussian_threshold(d_input, d_output, width, height, use_shared_memory);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

} // namespace cuda_image