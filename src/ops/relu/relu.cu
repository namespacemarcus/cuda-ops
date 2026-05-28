#include "../../common/cuda_utils.h"
#include "relu.cuh"

// FP32
__global__ void relu_f32_kernel(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = fmaxf(x[idx], 0.0f);
    }
}

__global__ void relu_f32x4_kernel(float *x, float *y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if ((idx + 3) < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = fmaxf(reg_x.x, 0.0f);
        reg_y.y = fmaxf(reg_x.y, 0.0f);
        reg_y.z = fmaxf(reg_x.z, 0.0f);
        reg_y.w = fmaxf(reg_x.w, 0.0f);
        FLOAT4(y[idx]) = reg_y;
    } else {
        for (int i = idx; i < N; ++i) {
            y[i] = fmaxf(x[i], 0.0f);
        }
    }
}

// FP16
__global__ void relu_f16_kernel(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = __hmax(__float2half(0.0f), x[idx]);
    }
}

__global__ void relu_f16x2_kernel(half *x, half *y, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if ((idx + 1) < N) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
        reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
        HALF2(y[idx]) = reg_y;
    } else if (idx < N) {
        y[idx] = __hmax(__float2half(0.0f), x[idx]);
    }
}

__global__ void relu_f16x8_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if ((idx + 7) < N) {
        half2 reg_x_0 = HALF2(x[idx]);
        half2 reg_x_1 = HALF2(x[idx + 2]);
        half2 reg_x_2 = HALF2(x[idx + 4]);
        half2 reg_x_3 = HALF2(x[idx + 6]);
        half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;

        reg_y_0.x = __hmax(__float2half(0.0f), reg_x_0.x);
        reg_y_0.y = __hmax(__float2half(0.0f), reg_x_0.y);
        reg_y_1.x = __hmax(__float2half(0.0f), reg_x_1.x);
        reg_y_1.y = __hmax(__float2half(0.0f), reg_x_1.y);
        reg_y_2.x = __hmax(__float2half(0.0f), reg_x_2.x);
        reg_y_2.y = __hmax(__float2half(0.0f), reg_x_2.y);
        reg_y_3.x = __hmax(__float2half(0.0f), reg_x_3.x);
        reg_y_3.y = __hmax(__float2half(0.0f), reg_x_3.y);

        HALF2(y[idx]) = reg_y_0;
        HALF2(y[idx + 2]) = reg_y_1;
        HALF2(y[idx + 4]) = reg_y_2;
        HALF2(y[idx + 6]) = reg_y_3;
    } else {
        for (int i = idx; i < N; ++i) {
            y[i] = __hmax(__float2half(0.0f), x[i]);
        }
    }
}

__global__ void relu_f16x8_pack_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const half2 zero2 = {__float2half(0.0f), __float2half(0.0f)};
    half pack_x[8], pack_y[8]; // 8*16=128bits

    if ((idx + 7) < N) {
        LDST128BITS(pack_x[idx]) = LDST128BITS(x[idx]);
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), zero2);
        }
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[idx]);
    } else {
        for (int i = idx; i < N; ++i) {
            y[i] = __hmax(__float2half(0.0f), x[i]);
        }
    }
}
