# CUDA Kernel Optimizer Agent Specification

## CRITICAL REQUIREMENT: Optimize ONLY the Provided Python Code

**THE AGENT MUST OPTIMIZE ONLY THE SPECIFIC PYTHON CODE PROVIDED IN THE image_processing FOLDER.**

**DO NOT CREATE GENERAL-PURPOSE KERNEL LIBRARIES OR ADDITIONAL KERNELS BEYOND WHAT IS NEEDED.**

When a Python file is provided:
1. **ANALYZE ONLY THE PROVIDED PYTHON FILE** - Do not create extra kernels for operations not present in the code
2. **OPTIMIZE EXACTLY WHAT IS REQUESTED** - Focus on fusing the specific operations in the Python code
3. **NO FUTURE-PROOFING** - Do not create additional kernels for potential future use cases
4. **FOLLOW THE SPECIFICATIONS IN COMMENTS** - Use exact data types, formats, and thresholds specified

Example: If the Python code only does:
```python
def process_image(img):
    filtered = gaussian_filter(img, sigma=2.0)
    return threshold(filtered, 0.5)
```
Then create ONLY a fused kernel for gaussian_filter + threshold. Do NOT create separate kernels for other filters, morphological operations, or any operations not present in the code.

## Project Overview

This agent analyzes the specific Python functions provided in the image_processing folder and generates optimized CUDA kernels ONLY for those exact operations. The focus is on fusing the operations present in the Python code for maximum performance, NOT on building a comprehensive image processing library.

### Example of What TO DO vs What NOT TO DO

**CORRECT Approach:**
If image_processing/filter.py contains:
```python
def process(img):
    return gaussian_blur(img, sigma=2.0)
```
→ Create ONE kernel: gaussian_blur_kernel.cu

**WRONG Approach:**
Same Python code as above, but creating:
- gaussian_blur_kernel.cu
- box_filter_kernel.cu (NOT IN PYTHON CODE!)
- median_filter_kernel.cu (NOT IN PYTHON CODE!)
- bilateral_filter_kernel.cu (NOT IN PYTHON CODE!)
- morphology_kernels.cu (NOT IN PYTHON CODE!)

The background knowledge about various filters is ONLY for understanding the code, NOT for creating additional kernels.

## Core Capabilities

### Approach
The agent is capable of understanding and optimizing various image processing patterns, but will ONLY implement kernels for operations actually present in the provided Python code. Background knowledge about filters, convolutions, morphological operations, etc. is used solely to understand and optimize the specific code provided, not to create a library of kernels.

### Data Types & Formats
- **Supported Types**: ONLY THE TYPES SPECIFIED IN THE PYTHON FILE COMMENTS
- **Default Type**: float32 (if Python specifies float32, DO NOT generate uint8/uint16 support)
- **Channel Support**: As specified in the Python file
- **Memory Layout**: CHW (Channel-Height-Width) format
- **Image Dimensions**: 2D images with multiple channels
- **Binary Format**: MUST match the format in Python comments exactly

### Target Environment
- **CUDA Version**: 12.9+ (tested with 12.9 in CI)
- **GPU Architecture**: SM 89/90 (Ada Lovelace and newer)
- **Memory Modes**: Unified memory and explicit transfers
- **Build System**: CMake
- **Library Output**: CUDA files should be generated to static lib and the test should be run from an executable

## Agent Behavior

### Input Analysis - CRITICAL STEPS

When given a Python function, the agent MUST:

1. **IDENTIFY THE EXACT OPERATIONS IN THE PYTHON CODE**:
   - List every operation performed in the Python function
   - DO NOT add operations that aren't in the code
   - Focus ONLY on what the Python code actually does

2. **PARSE PYTHON COMMENTS** to extract specifications:
   - Input data format specification
   - Data type (USE ONLY THIS TYPE)
   - Numerical accuracy threshold
   - Expected binary file structure

3. **GENERATE MINIMAL KERNEL SET**:
   - Create the minimum number of kernels needed
   - Prioritize fusion of operations in the Python code
   - DO NOT create separate kernels unless fusion is impossible
   - DO NOT create kernels for operations not in the Python code

4. **Analyze the computational pattern** to identify:
   - Memory access patterns for the specific operations
   - Opportunities to fuse the operations present
   - Data dependencies between the actual operations

5. **Map ONLY the NumPy patterns present in the code to CUDA**:
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

1. **Parse Python Function in image_processing folder**
   - Identify EXACT operations present in the code
   - Extract data types and specifications from comments
   - List operations to be optimized (ONLY those in the Python code)

2. **Plan Minimal Kernel Implementation**
   - Determine if operations can be fused
   - Design the minimum number of kernels needed
   - NO extra kernels for operations not in the Python code

3. **Generate ONLY Required CUDA Code**
   - Create fused kernel(s) for the specific operations
   - Generate host wrapper functions
   - Create test code for the specific implementation

4. **Verify Implementation**
   - Ensure it matches the Python code behavior exactly
   - Compile and test
   - Report any errors

5. **Document Implementation**
   - Explain fusion decisions for the specific operations
   - Note performance characteristics
   - DO NOT mention other possible kernels or future extensions

## Output Format

The agent produces ONLY the files needed for the specific Python code being optimized:
```
project/
├── kernels/
│   └── optimized_kernel.cu  # ONLY the kernel(s) for the Python code
├── host/
│   ├── launcher.cpp         # Host wrapper for the specific kernel
│   └── launcher.h
├── tests/
│   └── test_optimized.cpp   # Test for the specific implementation
└── CMakeLists.txt           # Build configuration
```

DO NOT create multiple kernel files for different operation types unless they are ALL present in the Python code.

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

## Important Guidelines and Best Practices

### Build Configuration
1. **CUDA Architecture Settings**:
   - Use ONLY SM 89 and 90 architectures (Ada Lovelace)
   - Set in CMakeLists.txt: `set(CMAKE_CUDA_ARCHITECTURES "89;90")`
   - Do NOT use `-arch=sm_XX` flags directly in compile options
   - Let CMake handle architecture flags automatically

2. **CUDA Version Requirements**:
   - Minimum CUDA 12.9 required
   - CI uses Docker image: `nvidia/cuda:12.9.0-devel-ubuntu22.04`
   - Ensure local development matches CI environment

### CI/CD Best Practices
1. **Use Docker-based CI**:
   - Eliminates CUDA installation overhead (~2-3 minutes saved)
   - Provides consistent build environment
   - Pre-installed CUDA toolkit in container
   - Total CI time: ~2 minutes vs ~4 minutes traditional

2. **CI Optimization**:
   - Build only Release mode in CI (Debug builds are optional)
   - Use container caching for faster subsequent runs
   - Run code quality checks in parallel with builds

3. **Docker CI Structure**:
   ```yaml
   container:
     image: nvidia/cuda:12.9.0-devel-ubuntu22.04
   ```

### Code Quality Standards
1. **Formatting**:
   - Use clang-format for consistent code style
   - CI enforces formatting checks
   - Run locally before committing

2. **Static Analysis**:
   - cppcheck runs on all C++ code
   - Address warnings before merging

### Development Workflow
1. **Local Testing**:
   - Build with same flags as CI: `-O3 -use_fast_math --ptxas-options=-v`
   - Test with Release build for performance validation
   - Use Debug build only for troubleshooting

2. **Performance Verification**:
   - Always check ptxas output for register usage
   - Monitor shared memory usage
   - Verify memory access patterns are coalesced

3. **Binary Compatibility**:
   - Generated kernels must match Python specification exactly
   - Test with provided binary test files
   - Validate numerical accuracy against Python reference

### Common Pitfalls to Avoid
1. **DO NOT add support for multiple architectures** unless specified
2. **DO NOT use older CUDA versions** - stick to 12.9+
3. **DO NOT generate template code** for multiple data types unless requested
4. **DO NOT skip binary format validation** - it must match Python specs exactly
5. **DO NOT use manual `-arch` flags** - use CMAKE_CUDA_ARCHITECTURES instead

