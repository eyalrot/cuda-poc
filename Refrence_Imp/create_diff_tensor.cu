#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "create_diff_tensor.h"

// Macro for checking CUDA errors
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code)
                  << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

// Kernel: each thread processes one pixel (for a given batch element)
// and computes the three output differences into separate output arrays.
__global__ void create_diff_tensors(const __half* input,
                                    __half* output0,
                                    __half* output1,
                                    __half* output2,
                                    int width, int height, int batch) {
    // Compute pixel coordinates (j: column, i: row) and batch index (b)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;  // each grid z-index corresponds to a batch element

    if (i < height && j < width && b < batch) {
        int pixel = i * width + j;           // pixel index within one channel
        int image_size = width * height;       // number of pixels per channel
        int base = b * 3 * image_size;         // starting index for batch b in input

        // Read the three input channels for this pixel.
        __half in0 = input[base + 0 * image_size + pixel];
        __half in1 = input[base + 1 * image_size + pixel];
        __half in2 = input[base + 2 * image_size + pixel];

        // Compute differences using half precision subtraction (__hsub)
        __half diff0 = __hsub(in0, in1);  // diff0 = input0 - input1
        __half diff1 = __hsub(in0, in2);  // diff1 = input0 - input2
        __half diff2 = __hsub(in1, in2);  // diff2 = input1 - input2

        // Write the results to the separate output arrays.
        output0[b * image_size + pixel] = diff0;
        output1[b * image_size + pixel] = diff1;
        output2[b * image_size + pixel] = diff2;
    }
}

// Launch function to call the kernel using a CUDA stream.
void launch_diff_tensors(const __half* d_input,
                         __half* d_output0,
                         __half* d_output1,
                         __half* d_output2,
                         int width, int height, int batch, cudaStream_t stream) {
    // Define kernel launch configuration:
    // Use a 2D block for pixel dimensions and grid's z-dimension for batch.
    dim3 block(192, 2, 1);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              batch);

    // Launch the kernel.
    create_diff_tensors<<<grid, block, 0, stream>>>(d_input, d_output0, d_output1, d_output2,
                                                    width, height, batch);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Benchmark function: allocates memory, runs the kernel, and prints timing & bandwidth.
// Note: Input tensor is of shape [B,3,W,H] and each output tensor is [B,W,H].
void benchmark(int batch, int width, int height) {
    size_t image_size = width * height;
    size_t num_input_elements = batch * 3 * image_size;
    size_t num_output_elements = batch * image_size;
    size_t input_bytes = num_input_elements * sizeof(__half);
    size_t output_bytes = num_output_elements * sizeof(__half);
    // Total bytes transferred: input read + three outputs written.
    size_t total_bytes = input_bytes + 3 * output_bytes;

    // Allocate device memory for input tensor.
    __half* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));

    // Allocate device memory for each output tensor.
    __half* d_output0;
    __half* d_output1;
    __half* d_output2;
    CUDA_CHECK(cudaMalloc(&d_output0, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_output1, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_output2, output_bytes));

    // Optionally, initialize input (here we just set it to zeros).
    CUDA_CHECK(cudaMemset(d_input, 0, input_bytes));
    CUDA_CHECK(cudaMemset(d_output0, 0, output_bytes));
    CUDA_CHECK(cudaMemset(d_output1, 0, output_bytes));
    CUDA_CHECK(cudaMemset(d_output2, 0, output_bytes));

    // Create a CUDA stream.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up kernel launch.
    launch_diff_tensors(d_input, d_output0, d_output1, d_output2, width, height, batch, stream);

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start time.
    CUDA_CHECK(cudaEventRecord(start, stream));
    // Launch the kernel.
    launch_diff_tensors(d_input, d_output0, d_output1, d_output2, width, height, batch, stream);
    // Record end time.
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Compute effective bandwidth (GB/s).
    double seconds = milliseconds / 1000.0;
    double bandwidth = (double)total_bytes / seconds / 1e9; // GB/s

    std::cout << "Batch size: " << batch << std::endl;
    std::cout << "Kernel time: " << milliseconds << " ms" << std::endl;
    std::cout << "Memory bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Cleanup.
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output0));
    CUDA_CHECK(cudaFree(d_output1));
    CUDA_CHECK(cudaFree(d_output2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int diff_tensor_test() {
    // Define image dimensions.
    int width = 176;   // adjust as needed
    int height = 1008; // adjust as needed

    std::cout << "Benchmark for batch size 100:" << std::endl;
    benchmark(100, width, height);

    std::cout << "\nBenchmark for batch size 1000:" << std::endl;
    benchmark(1000, width, height);

    return 0;
}