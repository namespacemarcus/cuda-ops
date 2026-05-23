#include <cuda_runtime.h>
#include <random>

#define THREADS_PER_BLOCK 256

__global__ void reduce(float *d_input, float *d_output) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (s * 2) == 0) {
            d_input[index] += d_input[index + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = d_input[index];
    }
}

bool check(float *out, float *res, int n) {
    for (int i = 0; i < n; ++i) {
        if (abs(out[i] - res[i]) > 1e-4) {
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

    int block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // Allocate one float per block, each block produces one result
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
        for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
            cur += input[i * THREADS_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num);
    dim3 Block(THREADS_PER_BLOCK);

    reduce<<<Grid, Block>>>(d_input, d_output);
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
