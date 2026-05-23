
#include <cuda_runtime.h>
#include <random>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE         32

template <unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce(float *d_input, float *d_output) {
    float sum = 0.f;
    unsigned int index = blockIdx.x * NUM_PER_BLOCK + threadIdx.x;
    unsigned int tid = threadIdx.x;

    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        sum += d_input[index + i * THREADS_PER_BLOCK];
    }

    // for (int offset = 16; offset > 0; offset >>= 1) {
    //     sum += __shfl_down_sync(0xffffffff, sum, offset);
    // }
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    __shared__ float warpLevelSums[32];
    const unsigned int laneId = tid % WARP_SIZE;
    const unsigned int warpId = tid / WARP_SIZE;
    if (laneId == 0) {
        warpLevelSums[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        sum = (laneId < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0.f;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    if (tid == 0) {
        d_output[blockIdx.x] = sum;
    }
}

bool check(float *out, float *res, int n) {
    for (int i = 0; i < n; ++i) {
        if (abs(out[i] - res[i]) > 5e-3) {
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024;
    float *input = (float *)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    // int block_num = ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) / 2;
    constexpr int block_num = 1024;
    constexpr int num_per_block = N / block_num;
    constexpr int num_per_thread = num_per_block / THREADS_PER_BLOCK;

    float *output = (float *)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        input[i] = dist(gen);
    }
    // cpu calc
    float *result = (float *)malloc(block_num * sizeof(float));
    for (int i = 0; i < block_num; ++i) {
        float cur = 0;
        for (int j = 0; j < num_per_block; ++j) {
            cur += input[i * num_per_block + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num);
    dim3 Block(THREADS_PER_BLOCK);

    reduce<num_per_block, num_per_thread><<<Grid, Block>>>(d_input, d_output);
    cudaMemcpy(output, d_output, block_num * sizeof(float),
               cudaMemcpyDeviceToHost);
    if (check(output, result, block_num)) {
        printf("check pass\n");
    } else {
        printf("check fail\n");
        for (int i = 0; i < block_num; ++i) {
            printf("%f ", output[i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
