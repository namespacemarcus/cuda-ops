#include <cuda_runtime.h>
#include <iostream>

int main() {
    int dev = 0;
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        std::cerr << "No CUDA devices available\n";
        return 1;
    }
    if (dev >= devCount) dev = 0;
    cudaDeviceProp devProp;
    if (cudaGetDeviceProperties(&devProp, dev) != cudaSuccess) {
        std::cerr << "Failed to get device properties\n";
        return 1;
    }
    std::cout << "Using GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "Number of SMs: " << devProp.multiProcessorCount << std::endl;
    std::cout << "Shared memory per block: "
              << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Max threads per block: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per multiprocessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max warps per SM: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    return 0;
}
