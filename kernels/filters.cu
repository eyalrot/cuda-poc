/*
 * CUDA Kernel Optimizer Agent - Filter Kernels
 * 
 * Optimized CUDA kernels for image filtering operations:
 * - Gaussian blur
 * - Box filter
 * - Median filter
 * - Bilateral filter
 * - Sobel edge detection
 * 
 * All kernels support CHW (Channel-Height-Width) memory layout
 * Target: CUDA 12.9, SM 89/90 architectures
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdint>

// Gaussian blur kernel with shared memory optimization
// Data type: float32 only (as specified in Python file)
__global__ void gaussian_blur_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int height, int width,
    float sigma, int kernel_size
) {
    extern __shared__ float shared_tile[];
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Pre-calculated Gaussian weights
    const int radius = kernel_size / 2;
    const float inv_sigma_sq = 1.0f / (2.0f * sigma * sigma);
    
    for (int c = 0; c < channels; c++) {
        const int idx = c * height * width + y * width + x;
        
        // Load tile with halo
        const int tile_x = tx + radius;
        const int tile_y = ty + radius;
        const int tile_width = blockDim.x + 2 * radius;
        
        // Load center pixel
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
        
        // Apply Gaussian filter
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                float weight = expf(-(kx * kx + ky * ky) * inv_sigma_sq);
                int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                sum += weight * shared_tile[src_idx];
                weight_sum += weight;
            }
        }
        
        output[idx] = sum / weight_sum;
        __syncthreads();
    }
}

// Fused Gaussian blur + Sobel edge detection kernel
// Data type: float32 only
__global__ void fused_blur_sobel_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
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
    
    const int tile_width = blockDim.x + 4; // 2 pixels halo for blur + 1 for sobel
    const float inv_sigma_sq = 1.0f / (2.0f * sigma * sigma);
    
    for (int c = 0; c < channels; c++) {
        const int idx = c * height * width + y * width + x;
        
        // Load tile with extended halo for both operations
        const int tile_x = tx + 2;
        const int tile_y = ty + 2;
        
        // Load center and halo pixels
        shared_tile[tile_y * tile_width + tile_x] = input[idx];
        
        // Load extended halo
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
        
        __syncthreads();
        
        // First apply Gaussian blur
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
        
        // Then apply Sobel edge detection
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
        output[idx] = (magnitude > threshold) ? 1.0f : 0.0f;
        
        __syncthreads();
    }
}

// Box filter kernel
// Data type: float32 only
__global__ void box_filter_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int height, int width,
    int kernel_size
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int radius = kernel_size / 2;
    const float inv_kernel_area = 1.0f / (kernel_size * kernel_size);
    
    for (int c = 0; c < channels; c++) {
        const int idx = c * height * width + y * width + x;
        
        float sum = 0.0f;
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int src_y = min(max(0, y + ky), height - 1);
                int src_x = min(max(0, x + kx), width - 1);
                int src_idx = c * height * width + src_y * width + src_x;
                sum += input[src_idx];
            }
        }
        
        output[idx] = sum * inv_kernel_area;
    }
}

// Median filter kernel (approximate using histogram)
// Data type: float32 only
__global__ void median_filter_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int height, int width,
    int kernel_size
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int radius = kernel_size / 2;
    const int kernel_area = kernel_size * kernel_size;
    const int median_pos = kernel_area / 2;
    
    for (int c = 0; c < channels; c++) {
        const int idx = c * height * width + y * width + x;
        
        // Collect neighborhood values
        float values[25]; // Max 5x5 kernel
        int count = 0;
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int src_y = min(max(0, y + ky), height - 1);
                int src_x = min(max(0, x + kx), width - 1);
                int src_idx = c * height * width + src_y * width + src_x;
                values[count++] = input[src_idx];
            }
        }
        
        // Simple bubble sort for small arrays
        for (int i = 0; i < count - 1; i++) {
            for (int j = 0; j < count - i - 1; j++) {
                if (values[j] > values[j + 1]) {
                    float temp = values[j];
                    values[j] = values[j + 1];
                    values[j + 1] = temp;
                }
            }
        }
        
        output[idx] = values[median_pos];
    }
}

// Bilateral filter kernel
// Data type: float32 only
__global__ void bilateral_filter_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int height, int width,
    float sigma_space, float sigma_color,
    int kernel_size
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
        
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int src_y = min(max(0, y + ky), height - 1);
                int src_x = min(max(0, x + kx), width - 1);
                int src_idx = c * height * width + src_y * width + src_x;
                
                float neighbor_val = input[src_idx];
                
                // Spatial weight
                float spatial_weight = expf(-(kx * kx + ky * ky) * inv_sigma_space_sq);
                
                // Color weight
                float color_diff = center_val - neighbor_val;
                float color_weight = expf(-(color_diff * color_diff) * inv_sigma_color_sq);
                
                float total_weight = spatial_weight * color_weight;
                sum += total_weight * neighbor_val;
                weight_sum += total_weight;
            }
        }
        
        output[idx] = sum / weight_sum;
    }
}

