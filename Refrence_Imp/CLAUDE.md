# CUDA Kernel Implementation Guidelines

## Expected File Structure for New Kernel Implementations

When implementing new CUDA kernels and their wrappers, follow this standardized structure:

```
kernel_name/
├── kernel_name.cu              # Main CUDA kernel implementation
├── kernel_name_test.cpp        # Standalone test file
├── kernel_name_torch.cpp       # PyTorch C++ extension wrapper
└── kernel_name_torch.hpp       # PyTorch interface header
```

## File Responsibilities and Patterns

### 1. Main CUDA Kernel File (`kernel_name.cu`)

**Required Components**:
- **Error Handling Macro**:
  ```cuda
  #define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
  ```

- **Kernel Function**: Core computation logic with clear parameter documentation
- **Launch Function**: Wrapper that handles grid/block configuration and stream management
- **Benchmark Function**: Performance testing with timing and bandwidth measurements
- **Test Entry Point**: Simple test function for standalone execution

**Design Patterns**:
- Use streams for asynchronous execution
- Implement warm-up runs before timing
- Calculate and report memory bandwidth
- Use appropriate thread block dimensions for the problem
- Include CUDA event-based timing

### 2. Test File (`kernel_name_test.cpp`)

**Purpose**: Standalone testing without framework dependencies

**Expected Contents**:
- Unit tests for correctness validation
- Edge case testing
- Performance regression tests
- Memory leak detection
- Error condition handling

### 3. PyTorch Extension Files

#### Header File (`kernel_name_torch.hpp`)
- Function declarations for Python bindings
- PyTorch tensor interface definitions
- Parameter validation functions

#### Implementation File (`kernel_name_torch.cpp`)
- PyTorch C++ extension implementation
- Tensor shape validation
- Data type conversions
- Python module registration

## General Implementation Guidelines

### Memory Management Pattern
1. Explicit device memory allocation with error checking
2. Proper cleanup in all code paths
3. Stream-aware memory operations
4. Consider unified memory for simplicity where appropriate

### Performance Optimization Patterns
1. **Memory Access**:
   - Ensure coalesced global memory access
   - Use appropriate data types (fp16 for bandwidth-limited operations)
   - Minimize memory transactions

2. **Benchmarking**:
   - Always include bandwidth calculations
   - Report kernel execution time
   - Test with representative batch sizes
   - Include warm-up iterations

### Error Handling Standards
1. Check all CUDA API calls
2. Provide meaningful error messages with context
3. Include file and line information in error reports
4. Allow graceful degradation where possible

### Code Organization Principles
1. **Separation of Concerns**:
   - Keep CUDA kernels independent of frameworks
   - Isolate platform-specific code
   - Maintain clear interfaces between components

2. **Modularity**:
   - One kernel per file
   - Self-contained benchmark functions
   - Reusable launch wrappers

3. **Documentation**:
   - Document kernel assumptions and constraints
   - Explain grid/block configuration choices
   - Include performance characteristics

## Integration Workflow

1. **Development Order**:
   - Implement and test CUDA kernel first
   - Add standalone benchmarks
   - Create framework wrappers as needed

2. **Testing Strategy**:
   - Unit test with standalone executable
   - Benchmark for performance validation
   - Integration test with framework wrappers

3. **Performance Validation**:
   - Measure memory bandwidth utilization
   - Compare against theoretical limits
   - Profile for bottlenecks

## Naming Conventions

- Kernel functions: `kernel_name_kernel`
- Launch functions: `launch_kernel_name`
- Benchmark functions: `benchmark` or `benchmark_kernel_name`
- Test functions: `kernel_name_test`
- PyTorch functions: Match PyTorch naming conventions

## Required Includes and Dependencies

### CUDA Kernel Files
- `<cuda_runtime.h>` - CUDA runtime API
- `<cuda_fp16.h>` - For half precision operations
- `<iostream>` - For output and debugging
- `<cstdlib>` - For exit() in error handling

### Test Files
- Testing framework headers as needed
- CUDA kernel headers

### PyTorch Extension Files
- `<torch/extension.h>` - PyTorch C++ API
- Custom kernel headers

This structure ensures consistent, maintainable, and performant CUDA kernel implementations across the project.

## CUDA Code Generation Guidelines

### Target Environment
- **CUDA Version**: 12.8 or higher
- **GPU Architecture**: SM 89 and above (Ada Lovelace, Hopper)
- **Optimization Level**: Highly optimized kernels targeting maximum performance

### Code Generation Principles

1. **Architecture-Specific Optimizations**:
   - Leverage features available in SM 89+ (Ada Lovelace)
   - Use tensor cores when applicable for matrix operations
   - Exploit increased shared memory capacity (up to 228KB per SM)
   - Utilize asynchronous memory copy operations

2. **Memory Optimization**:
   - Maximize memory bandwidth utilization (aim for >80% of theoretical)
   - Use vectorized loads/stores (float4, int4, etc.)
   - Implement memory access patterns that minimize cache conflicts
   - Consider using texture memory for spatially-local access patterns

3. **Kernel Fusion**:
   - Aggressively fuse operations to reduce memory bandwidth requirements
   - Eliminate intermediate memory writes where possible
   - Balance register pressure with operation fusion

4. **Warp-Level Programming**:
   - Use warp-level primitives (shuffle, vote, match)
   - Minimize warp divergence
   - Leverage cooperative groups for flexible synchronization

5. **Instruction-Level Optimizations**:
   - Use fast math operations where accuracy permits
   - Minimize integer division and modulo operations
   - Leverage hardware intrinsics for special functions

6. **Launch Configuration**:
   - Dynamic block size selection based on kernel requirements
   - Maximize occupancy while balancing resource usage
   - Use persistent kernels for small, frequent operations

### Advanced Features to Consider
- Thread Block Clusters (SM 90+)
- Distributed Shared Memory
- Asynchronous Transaction Barriers
- Hardware-accelerated reduction operations

### Performance Targets
- Memory-bound kernels: >85% bandwidth efficiency
- Compute-bound kernels: >70% SM utilization
- Latency-critical kernels: <10μs launch overhead

*Note: Additional guidelines will be added as the project evolves and new optimization patterns are identified.*