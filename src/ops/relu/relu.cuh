#pragma once

#include <cuda_fp16.h>


__global__ void relu_f32_kernel(float *x, float *y, int N);
__global__ void relu_f32x4_kernel(float *x, float *y, int N);

__global__ void relu_f16_kernel(half *x, half *y, int N);
__global__ void relu_f16x2_kernel(half *x, half *y, int N);
__global__ void relu_f16x8_kernel(half *x, half *y, int N);
__global__ void relu_f16x8_pack_kernel(half *x, half *y, int N);