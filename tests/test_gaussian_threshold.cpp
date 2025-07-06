#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "../host/gaussian_threshold_launcher.h"

// Simple CPU implementation for verification
void gaussian_threshold_cpu(
    const float* input,
    float* output,
    int width,
    int height
) {
    // 5x5 Gaussian kernel (normalized by 256)
    const float kernel[5][5] = {
        {1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f},
        {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
        {6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f, 6.0f/256.0f},
        {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
        {1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f}
    };
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            
            // Apply 5x5 Gaussian filter with zero padding
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    const int px = x + kx;
                    const int py = y + ky;
                    
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        sum += input[py * width + px] * kernel[ky + 2][kx + 2];
                    }
                }
            }
            
            // Apply threshold
            output[y * width + x] = (sum > 0.5f) ? sum : 0.0f;
        }
    }
}

bool compare_results(const float* cpu_result, const float* gpu_result, 
                    int size, float tolerance = 1e-5f) {
    int differences = 0;
    float max_diff = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        float diff = std::abs(cpu_result[i] - gpu_result[i]);
        if (diff > tolerance) {
            differences++;
            max_diff = std::max(max_diff, diff);
        }
    }
    
    if (differences > 0) {
        std::cout << "Found " << differences << " differences (max: " 
                  << max_diff << ")" << std::endl;
        return false;
    }
    return true;
}

int main() {
    // Test parameters
    const int width = 512;
    const int height = 512;
    const int size = width * height;
    
    // Allocate memory
    std::vector<float> h_input(size);
    std::vector<float> h_output_cpu(size);
    std::vector<float> h_output_gpu(size);
    std::vector<float> h_output_gpu_shared(size);
    
    // Initialize input with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < size; ++i) {
        h_input[i] = dis(gen);
    }
    
    std::cout << "Testing Gaussian Threshold Kernel" << std::endl;
    std::cout << "Image size: " << width << "x" << height << std::endl;
    
    // CPU reference
    std::cout << "\nComputing CPU reference..." << std::endl;
    gaussian_threshold_cpu(h_input.data(), h_output_cpu.data(), width, height);
    
    // GPU version (without shared memory)
    std::cout << "Running GPU kernel (global memory)..." << std::endl;
    cuda_image::process_image_cuda(h_input.data(), h_output_gpu.data(), 
                                   width, height, false);
    
    // GPU version (with shared memory)
    std::cout << "Running GPU kernel (shared memory)..." << std::endl;
    cuda_image::process_image_cuda(h_input.data(), h_output_gpu_shared.data(), 
                                   width, height, true);
    
    // Verify results
    std::cout << "\nVerifying results..." << std::endl;
    bool global_correct = compare_results(h_output_cpu.data(), h_output_gpu.data(), size);
    bool shared_correct = compare_results(h_output_cpu.data(), h_output_gpu_shared.data(), size);
    
    std::cout << "Global memory version: " << (global_correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Shared memory version: " << (shared_correct ? "PASSED" : "FAILED") << std::endl;
    
    // Print sample output values
    std::cout << "\nSample values (first 10):" << std::endl;
    std::cout << "Input\t\tCPU\t\tGPU" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_input[i] << "\t" 
                  << h_output_cpu[i] << "\t" 
                  << h_output_gpu_shared[i] << std::endl;
    }
    
    return (global_correct && shared_correct) ? 0 : 1;
}