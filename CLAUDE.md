# CUDA Kernel Optimizer Agent Specification

## CRITICAL REQUIREMENT: Follow Python File Specifications

**THE AGENT MUST GENERATE CUDA KERNELS EXACTLY ACCORDING TO THE SPECIFICATIONS IN THE PYTHON FILE COMMENTS.**

When a Python file is provided:
1. **READ THE COMMENTS FIRST** - They contain the exact specifications
2. **USE ONLY THE DATA TYPES SPECIFIED** - If the comment says "float32", generate kernels ONLY for float32
3. **FOLLOW THE BINARY FORMAT EXACTLY** - The comment specifies the input/output file format
4. **RESPECT THE ACCURACY THRESHOLD** - Use the threshold specified in the comments for validation

Example Python comment that MUST be followed:
```python
"""
Input format: binary file with header [H:int32][W:int32][C:int32][dtype:int32][data]
Data type: float32 (dtype=0)
Threshold: 1e-5 (absolute error)
"""
```

This means:
- Generate kernels ONLY for float32 type
- Expect binary files with the exact header format specified
- Validate results with 1e-5 absolute error threshold

## Project Overview

This agent analyzes Python image processing functions using NumPy and generates optimized CUDA kernels with equivalent functionality. The agent focuses on maximizing performance through kernel fusion and advanced CUDA optimization techniques while maintaining numerical accuracy AS SPECIFIED IN THE PYTHON FILE.

## Core Capabilities

### Supported Operations
- **Filters**: Gaussian blur, box filter, median filter, bilateral filter
- **Convolutions**: 2D convolutions with arbitrary kernels
- **Morphological**: Erosion, dilation, opening, closing
- **Histogram**: Computation, equalization, thresholding
- **Thresholding**: Binary, Otsu, adaptive
- **Basic Operations**: Element-wise operations, reductions

### Data Types & Formats
- **Supported Types**: ONLY THE TYPES SPECIFIED IN THE PYTHON FILE COMMENTS
- **Default Type**: float32 (if Python specifies float32, DO NOT generate uint8/uint16 support)
- **Channel Support**: As specified in the Python file
- **Memory Layout**: CHW (Channel-Height-Width) format
- **Image Dimensions**: 2D images with multiple channels
- **Binary Format**: MUST match the format in Python comments exactly

### Target Environment
- **CUDA Version**: 12.9
- **GPU Architecture**: SM 89/90 (Ada Lovelace and newer)
- **Memory Modes**: Unified memory and explicit transfers
- **Build System**: CMake
- **Library Output**: CUDA files should be generated to static lib and the test should be run from an executable

## Agent Behavior

### Input Analysis - CRITICAL STEPS

When given a Python function, the agent MUST:

1. **FIRST AND FOREMOST - Parse Python comments** to extract:
   - Input data format specification (THIS IS MANDATORY)
   - Data type (USE ONLY THIS TYPE - NO OTHERS)
   - Numerical accuracy threshold (USE THIS EXACT VALUE)
   - Expected binary file structure (FOLLOW EXACTLY)

2. **Generate kernels ONLY for the specified data type**:
   - If Python says "float32", generate ONLY float32 kernels
   - DO NOT add uint8 or uint16 support unless explicitly requested
   - DO NOT create template functions for multiple types

3. **Analyze the computational pattern** to identify:
   - Memory access patterns
   - Parallelization opportunities
   - Fusion candidates
   - Data dependencies

3. **Recognizes NumPy patterns** and maps to CUDA equivalents:
   ```python
   # Example Python input
   """
   Input format: binary file with header [H:int32][W:int32][C:int32][dtype:int32][data]
   Threshold: 1e-5 (absolute error)
   """
   def process_image(img):
       blurred = gaussian_filter(img, sigma=2.0)
       edges = sobel(blurred)
       return (edges > threshold_otsu(edges)).astype(np.uint8)
   ```

### Code Generation Strategy

#### 1. Kernel Fusion Analysis
- **Prioritize fusion** for memory-bound operations
- **Fuse operations** sharing the same data access pattern
- **Generate fused kernels** even at the cost of code complexity
- **Decision criteria**:
  - Memory bandwidth savings > 20%
  - Shared input/output buffers
  - Compatible thread mapping

#### 2. Memory Optimization
- **Shared Memory Usage**:
  - Estimate shared memory requirements
  - Use for stencil operations (filters, convolutions)
  - Implement tiling strategies
  - Static allocation preferred

- **Memory Access Patterns**:
  - Ensure coalesced global memory access
  - Optimize for CHW layout
  - Minimize bank conflicts in shared memory

#### 3. CUDA Features Utilization
- Warp-level primitives (shuffle, vote)
- Texture memory for filtered reads
- Fast math operations where accuracy permits
- Grid-stride loops for large images

### Generated Code Structure

#### Kernel Code Format
```cuda
/*
 * Generated CUDA kernel for: [original_function_name]
 * Original Python logic:
 *   blurred = gaussian_filter(img, sigma=2.0)
 *   edges = sobel(blurred)
 *   return (edges > threshold_otsu(edges)).astype(np.uint8)
 * 
 * Optimization decisions:
 * - Fused gaussian_blur + sobel + threshold into single kernel
 * - Shared memory tile size: 34x34 for 32x32 output block
 * - Expected memory bandwidth utilization: 85%
 * - Arithmetic intensity: 15.2 FLOP/byte
 */

template<typename T>
__global__ void fused_blur_sobel_threshold_kernel(
    const T* __restrict__ input,    // CHW layout
    uint8_t* __restrict__ output,   // CHW layout
    int channels, int height, int width,
    float sigma, float threshold
) {
    // Detailed implementation with extensive comments
    // explaining each optimization choice
    
    // Thread-to-pixel mapping
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Shared memory declaration
    extern __shared__ T shared_tile[];
    
    // ... kernel implementation ...
}
```

#### Host Code Structure
```cpp
// Host wrapper for kernel launch
void launch_fused_blur_sobel_threshold(
    const void* d_input,
    void* d_output,
    int channels, int height, int width,
    float sigma,
    cudaDataType_t dtype,
    bool use_unified_memory
) {
    // Block and grid configuration
    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);
    
    // Calculate shared memory size
    size_t shared_mem_size = /* calculated based on tile size */;
    
    // Launch appropriate kernel based on data type
    switch(dtype) {
        case CUDA_R_32F:
            fused_blur_sobel_threshold_kernel<float><<<grid, block, shared_mem_size>>>(
                /* parameters */
            );
            break;
        // ... other types ...
    }
}
```

### Testing Framework

#### Test Code Generation
For each kernel, generate corresponding test code:

```cpp
/*
 * Test for: fused_blur_sobel_threshold_kernel
 * Threshold: 1e-5 (from Python comments)
 */
class TestBlurSobelThreshold {
private:
    struct BinaryHeader {
        int32_t height;
        int32_t width;
        int32_t channels;
        int32_t dtype;
    };
    
    bool load_binary_file(const std::string& filename, void** data, BinaryHeader& header);
    bool compare_outputs(const void* cuda_output, const void* expected, 
                        size_t size, float threshold, cudaDataType_t dtype);
    
public:
    bool run_test(const std::string& input_file, 
                  const std::string& expected_output_file) {
        // Load input and expected output
        // Allocate GPU memory
        // Run kernel
        // Compare results
        // Report pass/fail with error statistics
    }
};
```

### Compilation Verification

After generating code, the agent:
1. Creates/updates CMakeLists.txt
2. Attempts compilation with specified CUDA flags
3. Reports compilation errors if any
4. Suggests fixes for common issues

```cmake
# Generated CMakeLists.txt snippet
add_library(image_kernels STATIC
    kernels/filters.cu
    kernels/morphology.cu
    host/launchers.cpp
)

target_compile_options(image_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        -arch=sm_89
        --ptxas-options=-v
    >
)

# Test executables
add_executable(test_blur_sobel tests/test_blur_sobel.cpp)
target_link_libraries(test_blur_sobel image_kernels)
```

## Decision Making Process

### When to Fuse Kernels
1. **Always fuse** when operations share memory access patterns
2. **Consider fusion** when:
   - Combined arithmetic intensity > individual operations
   - Intermediate results fit in registers/shared memory
   - No significant increase in register pressure

3. **Avoid fusion** when:
   - Register spilling occurs
   - Shared memory exceeds limits
   - Code complexity severely impacts maintainability

### Optimization Priority
1. **Memory bandwidth** (most critical for image processing)
2. **Occupancy** (aim for >50%)
3. **Instruction throughput**
4. **Code reusability**

### Error Handling
- **Boundary conditions**: Match Python behavior (mirror, clamp, or zero)
- **Special values**: Handle NaN/Inf for float operations
- **Out-of-bounds**: Always include bounds checking

## Agent Workflow

1. **Parse Python Function**
   - Extract algorithm logic
   - Identify data types and dimensions
   - Read threshold from comments

2. **Analyze Optimization Opportunities**
   - Memory access patterns
   - Computational intensity
   - Fusion candidates

3. **Generate CUDA Code**
   - Create kernels with detailed comments
   - Generate host wrapper functions
   - Create test code

4. **Verify Compilation**
   - Generate/update CMakeLists.txt
   - Compile with nvcc
   - Report any errors

5. **Document Decisions**
   - Explain optimization choices
   - Provide performance estimates
   - Note any limitations

## Output Format

The agent produces:
```
project/
├── kernels/
│   ├── fused_ops.cu      # Fused kernels
│   ├── filters.cu        # Individual filter kernels
│   └── morphology.cu     # Morphological operations
├── host/
│   ├── launchers.cpp     # Host-side kernel launchers
│   └── launchers.h
├── tests/
│   ├── test_common.h     # Test utilities
│   └── test_[kernel].cpp # Specific kernel tests
├── CMakeLists.txt        # Build configuration
└── performance_notes.md  # Optimization rationale
```

## Limitations & Constraints

1. **No support for**:
   - Recursive algorithms
   - Dynamic memory allocation in kernels
   - Variable-size outputs

2. **Requires manual intervention for**:
   - Complex control flow
   - Algorithms with sequential dependencies
   - Non-local memory access patterns

## Performance Expectations

The agent aims to achieve:
- **Memory bandwidth utilization**: >80% for memory-bound ops
- **Speedup vs NumPy**: 10-100x depending on operation
- **Numerical accuracy**: Within specified threshold
- **Kernel fusion benefit**: 20-50% over individual kernels

