# CUDA Gaussian Threshold Kernel

This project implements an optimized CUDA kernel for the image processing operations defined in `image_processing/image_filter_opencv.py`.

## Overview

The project optimizes a specific image processing pipeline:
1. 5x5 Gaussian filter with zero padding
2. Thresholding at 0.5

The CUDA implementation fuses these operations into a single kernel for maximum performance.

## Project Structure

```
cuda-poc/
├── image_processing/
│   └── image_filter_opencv.py    # Python reference implementation
├── kernels/
│   └── gaussian_threshold_kernel.cu  # Optimized CUDA kernel
├── host/
│   ├── gaussian_threshold_launcher.cu  # Host-side launcher
│   └── gaussian_threshold_launcher.h
├── tests/
│   └── test_gaussian_threshold.cpp  # Test comparing CUDA vs CPU
├── CMakeLists.txt
├── requirements.txt              # Python dependencies
├── performance_notes.md          # Performance analysis
└── CLAUDE.md                     # Agent specification
```

## Building

```bash
mkdir build
cd build
cmake ..
make -j
```

## Running Tests

```bash
./test_gaussian_threshold
```

## Requirements

- CUDA 12.9+
- GPU with SM 89/90 (Ada Lovelace architecture)
- CMake 3.18+
- Python 3.x with numpy, scipy, opencv-python (for reference implementation)

## Performance

The optimized CUDA kernel achieves:
- ~50% memory bandwidth reduction through kernel fusion
- Eliminates intermediate memory writes
- Provides both global and shared memory implementations