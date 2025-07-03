/*
 * CUDA Kernel Optimizer Agent - Host Launcher Headers
 * 
 * Host-side wrapper functions for launching CUDA kernels.
 * Data type: float32 only (as specified in Python file)
 * 
 * Target: CUDA 12.9, SM 89/90 architectures
 * Build: Static library with executable tests
 */

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <cstdint>

// Forward declarations
struct BinaryHeader {
    int32_t height;
    int32_t width;
    int32_t channels;
    int32_t dtype;  // 0 = float32 (as specified in Python)
};

// Filter operations - float32 only
namespace FilterOps {
    void launch_gaussian_blur(
        const void* d_input,
        void* d_output,
        int channels, int height, int width,
        float sigma,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_box_filter(
        const void* d_input,
        void* d_output,
        int channels, int height, int width,
        int kernel_size,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_median_filter(
        const void* d_input,
        void* d_output,
        int channels, int height, int width,
        int kernel_size,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_bilateral_filter(
        const void* d_input,
        void* d_output,
        int channels, int height, int width,
        float sigma_space, float sigma_color,
        int kernel_size,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
}

// Fused operations - float32 only
namespace FusedOps {
    void launch_fused_blur_sobel_threshold(
        const void* d_input,
        void* d_output,
        int channels, int height, int width,
        float sigma, float threshold,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_fused_bilateral_histeq(
        const void* d_input,
        void* d_output,
        const int* d_histogram,
        const float* d_cdf,
        int channels, int height, int width,
        float sigma_space, float sigma_color,
        int kernel_size, int num_bins,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_fused_gaussian_pyramid(
        const void* d_input,
        void* d_level1, void* d_level2, void* d_level3,
        int channels, int height, int width,
        float sigma1, float sigma2, float sigma3,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_fused_erosion_dilation(
        const void* d_input,
        void* d_eroded, void* d_dilated, void* d_gradient,
        const uint8_t* d_structuring_element,
        int channels, int height, int width,
        int kernel_size,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_fused_conv_activation(
        const void* d_input,
        void* d_output,
        const void* d_kernel,
        int channels, int height, int width,
        int kernel_size, int activation_type,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
}

// Morphological operations - float32 only
namespace MorphOps {
    void launch_erosion(
        const void* d_input,
        void* d_output,
        const uint8_t* d_structuring_element,
        int channels, int height, int width,
        int kernel_size,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_dilation(
        const void* d_input,
        void* d_output,
        const uint8_t* d_structuring_element,
        int channels, int height, int width,
        int kernel_size,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_opening(
        const void* d_input,
        void* d_output,
        const uint8_t* d_structuring_element,
        int channels, int height, int width,
        int kernel_size,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_closing(
        const void* d_input,
        void* d_output,
        const uint8_t* d_structuring_element,
        int channels, int height, int width,
        int kernel_size,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_morphological_gradient(
        const void* d_input,
        void* d_output,
        const uint8_t* d_structuring_element,
        int channels, int height, int width,
        int kernel_size,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_top_hat(
        const void* d_input,
        void* d_output,
        const uint8_t* d_structuring_element,
        int channels, int height, int width,
        int kernel_size,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
    
    void launch_black_hat(
        const void* d_input,
        void* d_output,
        const uint8_t* d_structuring_element,
        int channels, int height, int width,
        int kernel_size,
        cudaDataType_t dtype,
        bool use_unified_memory = false
    );
}

// Utility functions
namespace Utils {
    // Load binary file with header
    bool load_binary_file(
        const std::string& filename,
        void** data,
        BinaryHeader& header
    );
    
    // Save binary file with header
    bool save_binary_file(
        const std::string& filename,
        const void* data,
        const BinaryHeader& header
    );
    
    // Compare two float arrays with threshold
    bool compare_float_arrays(
        const float* array1,
        const float* array2,
        size_t size,
        float threshold
    );
    
    // Calculate optimal block and grid dimensions
    dim3 calculate_block_size(int width, int height, int max_threads = 1024);
    dim3 calculate_grid_size(int width, int height, dim3 block_size);
    
    // Calculate shared memory requirements
    size_t calculate_shared_memory_size(
        int block_width, int block_height,
        int halo_radius, int data_type_size
    );
    
    // Get device properties
    cudaDeviceProp get_device_properties(int device_id = 0);
    
    // Check CUDA errors
    void check_cuda_error(cudaError_t error, const char* file, int line);
}

// Macro for CUDA error checking
#define CUDA_CHECK(call) Utils::check_cuda_error(call, __FILE__, __LINE__)