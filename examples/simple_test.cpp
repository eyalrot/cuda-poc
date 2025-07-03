/*
 * Simple test program to verify CUDA kernel build system
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../host/launchers.h"

int main() {
    std::cout << "CUDA Kernel Optimizer - Build System Test\n";
    std::cout << "=========================================\n";
    
    // Check CUDA device
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceCount failed: " << cudaGetErrorString(error) << "\n";
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA devices found\n";
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)\n";
    
    // Get device properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "\nDevice " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
        std::cout << "  SM Count: " << prop.multiProcessorCount << "\n";
    }
    
    // Test simple allocation
    const int width = 256;
    const int height = 256;
    const int channels = 3;
    const size_t size = width * height * channels;
    
    std::cout << "\nTesting memory allocation...\n";
    
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    error = cudaMalloc(&d_input, size * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Error: Input allocation failed: " << cudaGetErrorString(error) << "\n";
        return 1;
    }
    
    error = cudaMalloc(&d_output, size * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Error: Output allocation failed: " << cudaGetErrorString(error) << "\n";
        cudaFree(d_input);
        return 1;
    }
    
    std::cout << "Memory allocation successful\n";
    
    // Create test data
    std::vector<float> h_input(size);
    for (size_t i = 0; i < size; i++) {
        h_input[i] = static_cast<float>(i % 256) / 255.0f;
    }
    
    // Copy to device
    error = cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Error: Memory copy failed: " << cudaGetErrorString(error) << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    
    std::cout << "Memory copy successful\n";
    
    // Test a simple kernel launch
    std::cout << "\nTesting Gaussian blur kernel...\n";
    try {
        FilterOps::launch_gaussian_blur(d_input, d_output, channels, height, width, 
                                       2.0f, CUDA_R_32F, false);
        
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "Error: Kernel execution failed: " << cudaGetErrorString(error) << "\n";
            cudaFree(d_input);
            cudaFree(d_output);
            return 1;
        }
        
        std::cout << "Kernel execution successful\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    
    // Copy result back
    std::vector<float> h_output(size);
    error = cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "Error: Result copy failed: " << cudaGetErrorString(error) << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    
    std::cout << "Result copy successful\n";
    
    // Basic validation
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum += h_output[i];
    }
    std::cout << "Output sum: " << sum << " (should be non-zero)\n";
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    std::cout << "\nBuild system test completed successfully!\n";
    return 0;
}