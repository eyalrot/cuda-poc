/*
 * CUDA Kernel Optimizer Agent - Morphological Operations
 * 
 * Optimized CUDA kernels for morphological image processing:
 * - Erosion
 * - Dilation
 * - Opening (erosion followed by dilation)
 * - Closing (dilation followed by erosion)
 * - Morphological gradient
 * - Top-hat transform
 * - Black-hat transform
 * 
 * All kernels support CHW (Channel-Height-Width) memory layout
 * Target: CUDA 12.9, SM 89/90 architectures
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdint>

// Erosion kernel with shared memory optimization
// Data type: float32 only
__global__ void erosion_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const uint8_t* __restrict__ structuring_element,
    int channels, int height, int width,
    int kernel_size
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
        
        // Load halo pixels with boundary handling (mirroring)
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
        if (ty < radius) {
            int src_y = max(0, y - radius);
            shared_tile[ty * tile_width + tile_x] = 
                input[c * height * width + src_y * width + x];
        }
        if (ty >= blockDim.y - radius) {
            int src_y = min(height - 1, y + radius);
            shared_tile[(tile_y + radius) * tile_width + tile_x] = 
                input[c * height * width + src_y * width + x];
        }
        
        __syncthreads();
        
        // Apply erosion
        float min_val = shared_tile[tile_y * tile_width + tile_x];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int se_idx = (ky + radius) * kernel_size + (kx + radius);
                if (structuring_element[se_idx]) {
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    min_val = min(min_val, shared_tile[src_idx]);
                }
            }
        }
        
        output[idx] = min_val;
        __syncthreads();
    }
}

// Dilation kernel with shared memory optimization
// Data type: float32 only
__global__ void dilation_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const uint8_t* __restrict__ structuring_element,
    int channels, int height, int width,
    int kernel_size
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
        
        // Load halo pixels with boundary handling (mirroring)
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
        if (ty < radius) {
            int src_y = max(0, y - radius);
            shared_tile[ty * tile_width + tile_x] = 
                input[c * height * width + src_y * width + x];
        }
        if (ty >= blockDim.y - radius) {
            int src_y = min(height - 1, y + radius);
            shared_tile[(tile_y + radius) * tile_width + tile_x] = 
                input[c * height * width + src_y * width + x];
        }
        
        __syncthreads();
        
        // Apply dilation
        float max_val = shared_tile[tile_y * tile_width + tile_x];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int se_idx = (ky + radius) * kernel_size + (kx + radius);
                if (structuring_element[se_idx]) {
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    max_val = max(max_val, shared_tile[src_idx]);
                }
            }
        }
        
        output[idx] = max_val;
        __syncthreads();
    }
}

// Opening kernel (erosion followed by dilation)
// Data type: float32 only
__global__ void opening_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ temp_buffer,
    const uint8_t* __restrict__ structuring_element,
    int channels, int height, int width,
    int kernel_size
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
        
        // First pass: erosion
        float min_val = shared_tile[tile_y * tile_width + tile_x];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int se_idx = (ky + radius) * kernel_size + (kx + radius);
                if (structuring_element[se_idx]) {
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    min_val = min(min_val, shared_tile[src_idx]);
                }
            }
        }
        
        temp_buffer[idx] = min_val;
        
        // Update shared memory with eroded values
        shared_tile[tile_y * tile_width + tile_x] = min_val;
        
        __syncthreads();
        
        // Second pass: dilation
        float max_val = shared_tile[tile_y * tile_width + tile_x];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int se_idx = (ky + radius) * kernel_size + (kx + radius);
                if (structuring_element[se_idx]) {
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    max_val = max(max_val, shared_tile[src_idx]);
                }
            }
        }
        
        output[idx] = max_val;
        __syncthreads();
    }
}

// Closing kernel (dilation followed by erosion)
// Data type: float32 only
__global__ void closing_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ temp_buffer,
    const uint8_t* __restrict__ structuring_element,
    int channels, int height, int width,
    int kernel_size
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
        
        // First pass: dilation
        float max_val = shared_tile[tile_y * tile_width + tile_x];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int se_idx = (ky + radius) * kernel_size + (kx + radius);
                if (structuring_element[se_idx]) {
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    max_val = max(max_val, shared_tile[src_idx]);
                }
            }
        }
        
        temp_buffer[idx] = max_val;
        
        // Update shared memory with dilated values
        shared_tile[tile_y * tile_width + tile_x] = max_val;
        
        __syncthreads();
        
        // Second pass: erosion
        float min_val = shared_tile[tile_y * tile_width + tile_x];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int se_idx = (ky + radius) * kernel_size + (kx + radius);
                if (structuring_element[se_idx]) {
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    min_val = min(min_val, shared_tile[src_idx]);
                }
            }
        }
        
        output[idx] = min_val;
        __syncthreads();
    }
}

// Morphological gradient kernel (dilation - erosion)
// Data type: float32 only
__global__ void morphological_gradient_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const uint8_t* __restrict__ structuring_element,
    int channels, int height, int width,
    int kernel_size
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
        
        // Compute both dilation and erosion
        float min_val = shared_tile[tile_y * tile_width + tile_x];
        float max_val = shared_tile[tile_y * tile_width + tile_x];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int se_idx = (ky + radius) * kernel_size + (kx + radius);
                if (structuring_element[se_idx]) {
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    float val = shared_tile[src_idx];
                    min_val = min(min_val, val);
                    max_val = max(max_val, val);
                }
            }
        }
        
        // Morphological gradient = dilation - erosion
        output[idx] = max_val - min_val;
        __syncthreads();
    }
}

// Top-hat transform kernel (original - opening)
// Data type: float32 only
__global__ void top_hat_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ temp_buffer,
    const uint8_t* __restrict__ structuring_element,
    int channels, int height, int width,
    int kernel_size
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
        float original_val = input[idx];
        
        // Load tile with halo
        shared_tile[tile_y * tile_width + tile_x] = original_val;
        
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
        
        // Erosion
        float min_val = shared_tile[tile_y * tile_width + tile_x];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int se_idx = (ky + radius) * kernel_size + (kx + radius);
                if (structuring_element[se_idx]) {
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    min_val = min(min_val, shared_tile[src_idx]);
                }
            }
        }
        
        // Update shared memory with eroded values
        shared_tile[tile_y * tile_width + tile_x] = min_val;
        
        __syncthreads();
        
        // Dilation (completing opening)
        float max_val = shared_tile[tile_y * tile_width + tile_x];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int se_idx = (ky + radius) * kernel_size + (kx + radius);
                if (structuring_element[se_idx]) {
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    max_val = max(max_val, shared_tile[src_idx]);
                }
            }
        }
        
        // Top-hat = original - opening
        output[idx] = original_val - max_val;
        __syncthreads();
    }
}

// Black-hat transform kernel (closing - original)
// Data type: float32 only
__global__ void black_hat_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ temp_buffer,
    const uint8_t* __restrict__ structuring_element,
    int channels, int height, int width,
    int kernel_size
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
        float original_val = input[idx];
        
        // Load tile with halo
        shared_tile[tile_y * tile_width + tile_x] = original_val;
        
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
        
        // Dilation
        float max_val = shared_tile[tile_y * tile_width + tile_x];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int se_idx = (ky + radius) * kernel_size + (kx + radius);
                if (structuring_element[se_idx]) {
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    max_val = max(max_val, shared_tile[src_idx]);
                }
            }
        }
        
        // Update shared memory with dilated values
        shared_tile[tile_y * tile_width + tile_x] = max_val;
        
        __syncthreads();
        
        // Erosion (completing closing)
        float min_val = shared_tile[tile_y * tile_width + tile_x];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int se_idx = (ky + radius) * kernel_size + (kx + radius);
                if (structuring_element[se_idx]) {
                    int src_idx = (tile_y + ky) * tile_width + (tile_x + kx);
                    min_val = min(min_val, shared_tile[src_idx]);
                }
            }
        }
        
        // Black-hat = closing - original
        output[idx] = min_val - original_val;
        __syncthreads();
    }
}

