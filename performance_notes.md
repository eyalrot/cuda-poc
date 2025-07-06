# Performance Notes - Gaussian Threshold Kernel

## Overview
This project implements an optimized CUDA kernel that fuses the operations from `image_filter_opencv.py`:
1. 5x5 Gaussian filter with zero padding
2. Thresholding at 0.5

## Optimization Strategy

### Kernel Fusion
The main optimization is fusing the Gaussian filter and threshold operations into a single kernel pass, eliminating the need to write intermediate results to global memory.

### Memory Access Patterns
Two kernel variants are provided:
1. **Global Memory Version**: Direct computation from global memory
2. **Shared Memory Version**: Uses shared memory tiles to reduce global memory accesses

### Expected Performance
- Memory bandwidth reduction: ~50% compared to separate operations
- Reduced kernel launch overhead
- Better cache utilization through coalesced memory access

## Implementation Details

### Data Type
- Uses `float32` as specified in the Python code
- Zero padding implemented by boundary checks

### Block Configuration
- 16x16 thread blocks for optimal occupancy
- Grid sized to cover entire image

### Shared Memory Usage
- Tile size: 20x20 (16x16 + 4 pixel border for 5x5 kernel)
- Cooperative loading of border pixels
- Single synchronization point after loading