/*
 * Optimized CUDA kernel for image_filter_opencv.py
 * Fuses 5x5 Gaussian filter with thresholding operation
 * 
 * Operations:
 * 1. Apply 5x5 Gaussian filter with zero padding
 * 2. Apply threshold of 0.5 (values > 0.5 kept, others set to 0)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 5x5 Gaussian kernel coefficients (normalized by 256)
__constant__ float c_gaussian_kernel[25] = {
    1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f,
    4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f,
    6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f, 6.0f/256.0f,
    4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f,
    1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f
};

/*
 * Fused Gaussian filter + threshold kernel
 * Applies 5x5 Gaussian filter with zero padding and thresholds at 0.5
 */
__global__ void gaussian_threshold_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    
    // Apply 5x5 Gaussian filter with zero padding (border pixels get 0 contribution)
    #pragma unroll
    for (int ky = -2; ky <= 2; ++ky) {
        #pragma unroll
        for (int kx = -2; kx <= 2; ++kx) {
            const int px = x + kx;
            const int py = y + ky;
            
            // Zero padding: skip if outside bounds
            if (px >= 0 && px < width && py >= 0 && py < height) {
                const int kernel_idx = (ky + 2) * 5 + (kx + 2);
                sum += input[py * width + px] * c_gaussian_kernel[kernel_idx];
            }
        }
    }
    
    // Apply threshold: keep values > 0.5, set others to 0
    output[y * width + x] = (sum > 0.5f) ? sum : 0.0f;
}

/*
 * Optimized version using shared memory for better memory access pattern
 */
__global__ void gaussian_threshold_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    // Shared memory tile with borders for the filter
    extern __shared__ float tile[];
    
    const int TILE_W = blockDim.x;
    const int TILE_H = blockDim.y;
    const int RADIUS = 2;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * TILE_W + tx;
    const int y = blockIdx.y * TILE_H + ty;
    
    // Load data into shared memory with borders
    const int shared_w = TILE_W + 2 * RADIUS;
    const int shared_idx = (ty + RADIUS) * shared_w + (tx + RADIUS);
    
    // Load center pixel
    if (x < width && y < height) {
        tile[shared_idx] = input[y * width + x];
    } else {
        tile[shared_idx] = 0.0f; // Zero padding
    }
    
    // Load border pixels (cooperatively)
    // Top border
    if (ty < RADIUS) {
        const int border_y = y - RADIUS;
        const int border_idx = ty * shared_w + (tx + RADIUS);
        if (x < width && border_y >= 0) {
            tile[border_idx] = input[border_y * width + x];
        } else {
            tile[border_idx] = 0.0f;
        }
    }
    
    // Bottom border
    if (ty >= TILE_H - RADIUS) {
        const int border_y = y + RADIUS;
        const int border_idx = (ty + 2 * RADIUS) * shared_w + (tx + RADIUS);
        if (x < width && border_y < height) {
            tile[border_idx] = input[border_y * width + x];
        } else {
            tile[border_idx] = 0.0f;
        }
    }
    
    // Left border
    if (tx < RADIUS) {
        const int border_x = x - RADIUS;
        const int border_idx = (ty + RADIUS) * shared_w + tx;
        if (border_x >= 0 && y < height) {
            tile[border_idx] = input[y * width + border_x];
        } else {
            tile[border_idx] = 0.0f;
        }
    }
    
    // Right border
    if (tx >= TILE_W - RADIUS) {
        const int border_x = x + RADIUS;
        const int border_idx = (ty + RADIUS) * shared_w + (tx + 2 * RADIUS);
        if (border_x < width && y < height) {
            tile[border_idx] = input[y * width + border_x];
        } else {
            tile[border_idx] = 0.0f;
        }
    }
    
    __syncthreads();
    
    if (x >= width || y >= height) return;
    
    // Apply 5x5 Gaussian filter from shared memory
    float sum = 0.0f;
    #pragma unroll
    for (int ky = -2; ky <= 2; ++ky) {
        #pragma unroll
        for (int kx = -2; kx <= 2; ++kx) {
            const int shared_y = ty + RADIUS + ky;
            const int shared_x = tx + RADIUS + kx;
            const int kernel_idx = (ky + 2) * 5 + (kx + 2);
            sum += tile[shared_y * shared_w + shared_x] * c_gaussian_kernel[kernel_idx];
        }
    }
    
    // Apply threshold and write output
    output[y * width + x] = (sum > 0.5f) ? sum : 0.0f;
}