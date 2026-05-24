#pragma once

#include <cuda_fp16.h>

#define WARP_SIZE          32
#define INT4(value)        (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value)      (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value)       (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value)     (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c, int N);
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c,
                                             int N);

__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N);
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N);
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N);
__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c,
                                                  int N);
