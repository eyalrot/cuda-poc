/*
 * CUDA Kernel Optimizer Agent - Fused Operations
 * 
 * High-performance fused kernels that combine multiple operations
 * to maximize memory bandwidth utilization and minimize kernel launches.
 * 
 * Fused operations include:
 * - Gaussian blur + Sobel + threshold
 * - Bilateral filter + histogram equalization
 * - Multi-scale Gaussian pyramid
 * - Erosion + dilation (morphological gradient)
 * 
 * Target: CUDA 12.9, SM 89/90 architectures
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdint>

// Fused Gaussian blur + Sobel + threshold kernel
// Optimized for edge detection pipelines
// Data type: float32 only
__global__ void fused_blur_sobel_threshold_kernel(
    const float* __restrict__ input,    // CHW layout
    uint8_t* __restrict__ output,   // CHW layout
    int channels, int height, int width,
    float sigma, float threshold
) {
    extern __shared__ float shared_tile[];
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Sobel kernels
    const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    // Shared memory configuration: 34x34 tile for 32x32 output
    const int tile_width = blockDim.x + 4;
    const int tile_height = blockDim.y + 4;
    const float inv_sigma_sq = 1.0f / (2.0f * sigma * sigma);
    
    for (int c = 0; c < channels; c++) {
        const int idx = c * height * width + y * width + x;
        
        // Load tile with halo for both blur and sobel
        const int tile_x = tx + 2;
        const int tile_y = ty + 2;
        
        // Load center pixel
        shared_tile[tile_y * tile_width + tile_x] = input[idx];
        
        // Load halo pixels with boundary handling
        if (tx < 2) {
            int src_x = max(0, x - 2);
            shared_tile[tile_y * tile_width + tx] = 
                input[c * height * width + y * width + src_x];
        }
        if (tx >= blockDim.x - 2) {
            int src_x = min(width - 1, x + 2);
            shared_tile[tile_y * tile_width + tile_x + 2] = 
                input[c * height * width + y * width + src_x];
        }
        if (ty < 2) {
            int src_y = max(0, y - 2);
            shared_tile[ty * tile_width + tile_x] = 
                input[c * height * width + src_y * width + x];
        }
        if (ty >= blockDim.y - 2) {
            int src_y = min(height - 1, y + 2);
            shared_tile[(tile_y + 2) * tile_width + tile_x] = 
                input[c * height * width + src_y * width + x];
        }
        
        __syncthreads();
        
        // Apply Gaussian blur with 5x5 kernel
        float blurred = 0.0f;
        float weight_sum = 0.0f;
        
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                float weight = expf(-(kx * kx + ky * ky) * inv_sigma_sq);
                int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                blurred += weight * shared_tile[src_idx];
                weight_sum += weight;
            }
        }
        blurred /= weight_sum;
        
        // Update shared memory with blurred values
        shared_tile[tile_y * tile_width + tile_x] = blurred;
        
        __syncthreads();
        
        // Apply Sobel edge detection
        float grad_x = 0.0f, grad_y = 0.0f;
        
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                float pixel_val = shared_tile[src_idx];
                grad_x += sobel_x[ky + 1][kx + 1] * pixel_val;
                grad_y += sobel_y[ky + 1][kx + 1] * pixel_val;
            }
        }
        
        float magnitude = sqrtf(grad_x * grad_x + grad_y * grad_y);
        
        // Apply threshold
        output[idx] = (magnitude > threshold) ? 255 : 0;
        
        __syncthreads();
    }
}

// Fused bilateral filter + histogram equalization
// Data type: float32 only
__global__ void fused_bilateral_histeq_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ histogram,
    const float* __restrict__ cdf,
    int channels, int height, int width,
    float sigma_space, float sigma_color,
    int kernel_size, int num_bins
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int radius = kernel_size / 2;
    const float inv_sigma_space_sq = 1.0f / (2.0f * sigma_space * sigma_space);
    const float inv_sigma_color_sq = 1.0f / (2.0f * sigma_color * sigma_color);
    
    for (int c = 0; c < channels; c++) {
        const int idx = c * height * width + y * width + x;
        const float center_val = input[idx];
        
        // Apply bilateral filter
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int src_y = min(max(0, y + ky), height - 1);
                int src_x = min(max(0, x + kx), width - 1);
                int src_idx = c * height * width + src_y * width + src_x;
                
                float neighbor_val = input[src_idx];
                
                float spatial_weight = expf(-(kx * kx + ky * ky) * inv_sigma_space_sq);
                float color_diff = center_val - neighbor_val;
                float color_weight = expf(-(color_diff * color_diff) * inv_sigma_color_sq);
                
                float total_weight = spatial_weight * color_weight;
                sum += total_weight * neighbor_val;
                weight_sum += total_weight;
            }
        }
        
        float filtered_val = sum / weight_sum;
        
        // Apply histogram equalization
        int bin_idx = min(num_bins - 1, (int)(filtered_val * num_bins));
        float equalized_val = cdf[c * num_bins + bin_idx];
        
        output[idx] = equalized_val;
    }
}

// Fused multi-scale Gaussian pyramid (3 levels)
template<typename T>
__global__ void fused_gaussian_pyramid_kernel(
    const T* __restrict__ input,
    T* __restrict__ level1,    // 1/2 scale
    T* __restrict__ level2,    // 1/4 scale
    T* __restrict__ level3,    // 1/8 scale
    int channels, int height, int width,
    float sigma1, float sigma2, float sigma3
) {
    extern __shared__ float shared_tile[];
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int tile_width = blockDim.x + 8; // Extended halo for multiple scales
    const int tile_x = tx + 4;
    const int tile_y = ty + 4;
    
    for (int c = 0; c < channels; c++) {
        const int idx = c * height * width + y * width + x;
        
        // Load tile with extended halo
        shared_tile[tile_y * tile_width + tile_x] = input[idx];
        
        // Load halo pixels
        if (tx < 4) {
            int src_x = max(0, x - 4);
            shared_tile[tile_y * tile_width + tx] = 
                input[c * height * width + y * width + src_x];
        }
        if (tx >= blockDim.x - 4) {
            int src_x = min(width - 1, x + 4);
            shared_tile[tile_y * tile_width + tile_x + 4] = 
                input[c * height * width + y * width + src_x];
        }
        
        __syncthreads();
        
        // Generate level 1 (1/2 scale)
        if (x % 2 == 0 && y % 2 == 0) {
            float sum = 0.0f, weight_sum = 0.0f;
            float inv_sigma1_sq = 1.0f / (2.0f * sigma1 * sigma1);
            
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    float weight = expf(-(kx * kx + ky * ky) * inv_sigma1_sq);
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    sum += weight * shared_tile[src_idx];
                    weight_sum += weight;
                }
            }
            
            int level1_idx = c * (height/2) * (width/2) + (y/2) * (width/2) + (x/2);
            level1[level1_idx] = sum / weight_sum;
        }
        
        // Generate level 2 (1/4 scale)
        if (x % 4 == 0 && y % 4 == 0) {
            float sum = 0.0f, weight_sum = 0.0f;
            float inv_sigma2_sq = 1.0f / (2.0f * sigma2 * sigma2);
            
            for (int ky = -3; ky <= 3; ky++) {
                for (int kx = -3; kx <= 3; kx++) {
                    float weight = expf(-(kx * kx + ky * ky) * inv_sigma2_sq);
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    sum += weight * shared_tile[src_idx];
                    weight_sum += weight;
                }
            }
            
            int level2_idx = c * (height/4) * (width/4) + (y/4) * (width/4) + (x/4);
            level2[level2_idx] = sum / weight_sum;
        }
        
        // Generate level 3 (1/8 scale)
        if (x % 8 == 0 && y % 8 == 0) {
            float sum = 0.0f, weight_sum = 0.0f;
            float inv_sigma3_sq = 1.0f / (2.0f * sigma3 * sigma3);
            
            for (int ky = -4; ky <= 4; ky++) {
                for (int kx = -4; kx <= 4; kx++) {
                    float weight = expf(-(kx * kx + ky * ky) * inv_sigma3_sq);
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    sum += weight * shared_tile[src_idx];
                    weight_sum += weight;
                }
            }
            
            int level3_idx = c * (height/8) * (width/8) + (y/8) * (width/8) + (x/8);
            level3[level3_idx] = sum / weight_sum;
        }
        
        __syncthreads();
    }
}

// Fused erosion + dilation (morphological gradient)
// Data type: float32 only
__global__ void fused_erosion_dilation_kernel(
    const float* __restrict__ input,
    float* __restrict__ eroded,
    float* __restrict__ dilated,
    float* __restrict__ gradient,
    int channels, int height, int width,
    int kernel_size
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int radius = kernel_size / 2;
    
    for (int c = 0; c < channels; c++) {
        const int idx = c * height * width + y * width + x;
        
        float min_val = input[idx];
        float max_val = input[idx];
        
        // Find min and max in neighborhood
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int src_y = min(max(0, y + ky), height - 1);
                int src_x = min(max(0, x + kx), width - 1);
                int src_idx = c * height * width + src_y * width + src_x;
                
                float val = input[src_idx];
                min_val = min(min_val, val);
                max_val = max(max_val, val);
            }
        }
        
        eroded[idx] = min_val;
        dilated[idx] = max_val;
        gradient[idx] = max_val - min_val;
    }
}

// Fused convolution + activation (ReLU/Sigmoid)
// Data type: float32 only
__global__ void fused_conv_activation_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ kernel,
    int channels, int height, int width,
    int kernel_size, int activation_type
) {
    extern __shared__ float shared_tile[];
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int radius = kernel_size / 2;
    const int tile_width = blockDim.x + 2 * radius;
    const int tile_x = tx + radius;
    const int tile_y = ty + radius;
    
    for (int c = 0; c < channels; c++) {
        const int idx = c * height * width + y * width + x;
        
        // Load tile with halo
        shared_tile[tile_y * tile_width + tile_x] = input[idx];
        
        // Load halo pixels
        if (tx < radius) {
            int src_x = max(0, x - radius);
            shared_tile[tile_y * tile_width + tx] = 
                input[c * height * width + y * width + src_x];
        }
        if (tx >= blockDim.x - radius) {
            int src_x = min(width - 1, x + radius);
            shared_tile[tile_y * tile_width + tile_x + radius] = 
                input[c * height * width + y * width + src_x];
        }
        
        __syncthreads();
        
        // Apply convolution
        float sum = 0;
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int kernel_idx = (ky + radius) * kernel_size + (kx + radius);
                int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                sum += kernel[kernel_idx] * shared_tile[src_idx];
            }
        }
        
        // Apply activation function
        float result;
        switch (activation_type) {
            case 0: // ReLU
                result = fmaxf(0.0f, sum);
                break;
            case 1: // Sigmoid
                result = 1.0f / (1.0f + expf(-sum));
                break;
            default:
                result = sum;
        }
        
        output[idx] = result;
        __syncthreads();
    }
}

