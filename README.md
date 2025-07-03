# CUDA Kernel Optimizer

A project for automatically generating optimized CUDA kernels from Python image processing functions.

## Overview

This agent analyzes Python image processing functions using NumPy and generates optimized CUDA kernels with equivalent functionality. The agent focuses on maximizing performance through kernel fusion and advanced CUDA optimization techniques while maintaining numerical accuracy.

## Key Features

- **Automatic Kernel Generation**: Converts Python/NumPy code to optimized CUDA kernels
- **Kernel Fusion**: Automatically fuses operations to minimize memory bandwidth
- **Float32 Support**: Optimized for float32 data type as specified in Python files
- **Target Architecture**: CUDA 12.8+, SM 89/90 (Ada Lovelace and newer)

## Project Structure

```
cuda-poc/
├── kernels/           # CUDA kernel implementations
│   ├── filters.cu     # Image filtering operations
│   ├── morphology.cu  # Morphological operations
│   └── fused_ops.cu   # Fused kernel operations
├── host/              # Host-side launcher code
│   ├── launchers.h    # C++ API headers
│   └── launchers.cu   # Kernel launcher implementations
├── tests/             # Test files
├── examples/          # Example usage
├── image_processing/  # Python reference implementations
└── CLAUDE.md          # Agent specification document
```

## Building

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Requirements

- CUDA 12.8 or later
- CMake 3.18 or later
- C++17 compatible compiler
- GPU with compute capability 8.9 or higher (RTX 40 series)

## Usage

The project generates a static library `libcuda_image_kernels.a` that can be linked with your application.

See `examples/simple_test.cpp` for basic usage.

## Data Format

Input files follow the binary format specified in Python comments:
- Header: `[H:int32][W:int32][C:int32][dtype:int32]`
- Data: float32 array in CHW (Channel-Height-Width) format

## License

This project is for demonstration purposes.