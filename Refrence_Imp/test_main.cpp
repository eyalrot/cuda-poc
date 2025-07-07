#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>

// Custom test environment for CUDA initialization
class CUDAEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        
        if (error != cudaSuccess || deviceCount == 0) {
            std::cerr << "No CUDA-capable devices found!" << std::endl;
            exit(1);
        }
        
        // Set device and print info
        cudaSetDevice(0);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        std::cout << "==================================" << std::endl;
        std::cout << "CUDA Device Information:" << std::endl;
        std::cout << "Device Name: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "==================================" << std::endl;
    }
    
    void TearDown() override {
        // Reset device after all tests
        cudaDeviceReset();
    }
};

int main(int argc, char **argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Add custom CUDA environment
    ::testing::AddGlobalTestEnvironment(new CUDAEnvironment);
    
    // Run all tests
    return RUN_ALL_TESTS();
}