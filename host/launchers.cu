/*
 * CUDA Kernel Optimizer Agent - Host Launcher Implementation
 * 
 * Implementation of host-side wrapper functions for launching CUDA kernels.
 * Data type: float32 only (as specified in Python file)
 * 
 * Target: CUDA 12.9, SM 89/90 architectures
 */

#include "launchers.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

// External kernel declarations (float32 only)
extern __global__ void gaussian_blur_kernel(
    const float* input, float* output,
    int channels, int height, int width,
    float sigma, int kernel_size
);

extern __global__ void box_filter_kernel(
    const float* input, float* output,
    int channels, int height, int width,
    int kernel_size
);

extern __global__ void median_filter_kernel(
    const float* input, float* output,
    int channels, int height, int width,
    int kernel_size
);

extern __global__ void bilateral_filter_kernel(
    const float* input, float* output,
    int channels, int height, int width,
    float sigma_space, float sigma_color,
    int kernel_size
);

extern __global__ void fused_blur_sobel_threshold_kernel(
    const float* input, uint8_t* output,
    int channels, int height, int width,
    float sigma, float threshold
);


extern __global__ void fused_bilateral_histeq_kernel(
    const float* input, float* output,
    const int* histogram, const float* cdf,
    int channels, int height, int width,
    float sigma_space, float sigma_color,
    int kernel_size, int num_bins
);

extern __global__ void fused_gaussian_pyramid_kernel(
    const float* input,
    float* level1, float* level2, float* level3,
    int channels, int height, int width,
    float sigma1, float sigma2, float sigma3
);

extern __global__ void fused_erosion_dilation_kernel(
    const float* input,
    float* eroded, float* dilated, float* gradient,
    const uint8_t* structuring_element,
    int channels, int height, int width,
    int kernel_size
);

extern __global__ void fused_conv_activation_kernel(
    const float* input, float* output,
    const float* kernel_weights,
    int channels, int height, int width,
    int kernel_size, int activation_type
);

extern __global__ void erosion_kernel(
    const float* input, float* output,
    const uint8_t* structuring_element,
    int channels, int height, int width,
    int kernel_size
);

extern __global__ void dilation_kernel(
    const float* input, float* output,
    const uint8_t* structuring_element,
    int channels, int height, int width,
    int kernel_size
);

extern __global__ void opening_kernel(
    const float* input, float* output,
    float* temp_buffer,
    const uint8_t* structuring_element,
    int channels, int height, int width,
    int kernel_size
);

extern __global__ void closing_kernel(
    const float* input, float* output,
    float* temp_buffer,
    const uint8_t* structuring_element,
    int channels, int height, int width,
    int kernel_size
);

extern __global__ void morphological_gradient_kernel(
    const float* input, float* output,
    const uint8_t* structuring_element,
    int channels, int height, int width,
    int kernel_size
);

extern __global__ void top_hat_kernel(
    const float* input, float* output,
    float* temp_buffer,
    const uint8_t* structuring_element,
    int channels, int height, int width,
    int kernel_size
);

extern __global__ void black_hat_kernel(
    const float* input, float* output,
    float* temp_buffer,
    const uint8_t* structuring_element,
    int channels, int height, int width,
    int kernel_size
);

// Filter operations implementation
namespace FilterOps {

void launch_gaussian_blur(
    const void* d_input,
    void* d_output,
    int channels, int height, int width,
    float sigma,
    cudaDataType_t dtype,
    bool use_unified_memory
) {
    // Only support float32
    if (dtype != CUDA_R_32F) {
        throw std::runtime_error("Only float32 is supported");
    }
    
    const float* input = static_cast<const float*>(d_input);
    float* output = static_cast<float*>(d_output);
    
    // Calculate kernel size based on sigma
    int kernel_size = static_cast<int>(ceil(6.0f * sigma)) | 1;
    kernel_size = std::min(kernel_size, 17); // Max 17x17
    
    // Configure kernel launch
    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);
    
    // Calculate shared memory size
    int tile_width = block.x + kernel_size - 1;
    int tile_height = block.y + kernel_size - 1;
    size_t shared_mem_size = tile_width * tile_height * sizeof(float);
    
    // Launch kernel
    gaussian_blur_kernel<<<grid, block, shared_mem_size>>>(
        input, output, channels, height, width, sigma, kernel_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    if (!use_unified_memory) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void launch_box_filter(
    const void* d_input,
    void* d_output,
    int channels, int height, int width,
    int kernel_size,
    cudaDataType_t dtype,
    bool use_unified_memory
) {
    if (dtype != CUDA_R_32F) {
        throw std::runtime_error("Only float32 is supported");
    }
    
    const float* input = static_cast<const float*>(d_input);
    float* output = static_cast<float*>(d_output);
    
    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);
    
    int tile_width = block.x + kernel_size - 1;
    int tile_height = block.y + kernel_size - 1;
    size_t shared_mem_size = tile_width * tile_height * sizeof(float);
    
    box_filter_kernel<<<grid, block, shared_mem_size>>>(
        input, output, channels, height, width, kernel_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    if (!use_unified_memory) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void launch_median_filter(
    const void* d_input,
    void* d_output,
    int channels, int height, int width,
    int kernel_size,
    cudaDataType_t dtype,
    bool use_unified_memory
) {
    if (dtype != CUDA_R_32F) {
        throw std::runtime_error("Only float32 is supported");
    }
    
    const float* input = static_cast<const float*>(d_input);
    float* output = static_cast<float*>(d_output);
    
    dim3 block(16, 16); // Smaller for median filter
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    median_filter_kernel<<<grid, block>>>(
        input, output, channels, height, width, kernel_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    if (!use_unified_memory) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void launch_bilateral_filter(
    const void* d_input,
    void* d_output,
    int channels, int height, int width,
    float sigma_space, float sigma_color,
    int kernel_size,
    cudaDataType_t dtype,
    bool use_unified_memory
) {
    if (dtype != CUDA_R_32F) {
        throw std::runtime_error("Only float32 is supported");
    }
    
    const float* input = static_cast<const float*>(d_input);
    float* output = static_cast<float*>(d_output);
    
    dim3 block(16, 16); // Smaller for bilateral filter
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    bilateral_filter_kernel<<<grid, block>>>(
        input, output, channels, height, width, 
        sigma_space, sigma_color, kernel_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    if (!use_unified_memory) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

} // namespace FilterOps

// Fused operations implementation
namespace FusedOps {

void launch_fused_blur_sobel_threshold(
    const void* d_input,
    void* d_output,
    int channels, int height, int width,
    float sigma, float threshold,
    cudaDataType_t dtype,
    bool use_unified_memory
) {
    if (dtype != CUDA_R_32F) {
        throw std::runtime_error("Only float32 is supported");
    }
    
    const float* input = static_cast<const float*>(d_input);
    uint8_t* output = static_cast<uint8_t*>(d_output);
    
    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);
    
    // Extended shared memory for blur + sobel
    int kernel_size = static_cast<int>(ceil(6.0f * sigma)) | 1;
    int tile_width = block.x + kernel_size + 1; // +1 for sobel
    int tile_height = block.y + kernel_size + 1;
    size_t shared_mem_size = tile_width * tile_height * sizeof(float);
    
    fused_blur_sobel_threshold_kernel<<<grid, block, shared_mem_size>>>(
        input, output, channels, height, width, sigma, threshold
    );
    
    CUDA_CHECK(cudaGetLastError());
    if (!use_unified_memory) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Add other fused operations implementations...

} // namespace FusedOps

// Morphological operations implementation
namespace MorphOps {

void launch_erosion(
    const void* d_input,
    void* d_output,
    const uint8_t* d_structuring_element,
    int channels, int height, int width,
    int kernel_size,
    cudaDataType_t dtype,
    bool use_unified_memory
) {
    if (dtype != CUDA_R_32F) {
        throw std::runtime_error("Only float32 is supported");
    }
    
    const float* input = static_cast<const float*>(d_input);
    float* output = static_cast<float*>(d_output);
    
    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);
    
    int tile_width = block.x + kernel_size - 1;
    int tile_height = block.y + kernel_size - 1;
    size_t shared_mem_size = tile_width * tile_height * sizeof(float);
    
    erosion_kernel<<<grid, block, shared_mem_size>>>(
        input, output, d_structuring_element,
        channels, height, width, kernel_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    if (!use_unified_memory) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Add other morphological operations implementations...

} // namespace MorphOps

// Utility functions implementation
namespace Utils {

void check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line 
                  << " - " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

dim3 calculate_block_size(int width, int height, int max_threads) {
    int block_x = 32;
    int block_y = max_threads / block_x;
    return dim3(block_x, block_y);
}

dim3 calculate_grid_size(int width, int height, dim3 block_size) {
    return dim3(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );
}

size_t calculate_shared_memory_size(
    int block_width, int block_height,
    int halo_radius, int data_type_size
) {
    int tile_width = block_width + 2 * halo_radius;
    int tile_height = block_height + 2 * halo_radius;
    return tile_width * tile_height * data_type_size;
}

cudaDeviceProp get_device_properties(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    return prop;
}

bool load_binary_file(
    const std::string& filename,
    void** data,
    BinaryHeader& header
) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read header
    file.read(reinterpret_cast<char*>(&header), sizeof(BinaryHeader));
    
    // Only support float32 (dtype = 0)
    if (header.dtype != 0) {
        std::cerr << "Error: Only float32 (dtype=0) is supported, got dtype=" 
                  << header.dtype << std::endl;
        return false;
    }
    
    // Allocate and read data
    size_t size = header.height * header.width * header.channels * sizeof(float);
    *data = malloc(size);
    file.read(reinterpret_cast<char*>(*data), size);
    
    return file.good();
}

bool save_binary_file(
    const std::string& filename,
    const void* data,
    const BinaryHeader& header
) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write header
    file.write(reinterpret_cast<const char*>(&header), sizeof(BinaryHeader));
    
    // Write data
    size_t size = header.height * header.width * header.channels * sizeof(float);
    file.write(reinterpret_cast<const char*>(data), size);
    
    return file.good();
}

bool compare_float_arrays(
    const float* array1,
    const float* array2,
    size_t size,
    float threshold
) {
    for (size_t i = 0; i < size; i++) {
        float diff = std::abs(array1[i] - array2[i]);
        if (diff > threshold) {
            std::cerr << "Mismatch at index " << i << ": " 
                      << array1[i] << " vs " << array2[i] 
                      << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace Utils