cmake_minimum_required(VERSION 3.20.0)
project(cuda_kernels LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit)
add_subdirectory(src)