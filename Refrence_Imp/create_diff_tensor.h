#ifndef CREATE_DIFF_TENSOR_H
#define CREATE_DIFF_TENSOR_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Function declarations for CUDA kernels
void launch_diff_tensors(const __half* d_input,
                        __half* d_output0,
                        __half* d_output1,
                        __half* d_output2,
                        int width, int height, int batch, cudaStream_t stream);

void benchmark(int batch, int width, int height);

int diff_tensor_test();

#endif // CREATE_DIFF_TENSOR_H