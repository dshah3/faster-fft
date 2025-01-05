#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "experiments/headers/sgemm_naive.h"

// kernel

// this is without alpha and beta
__global__ void sgemm_naive(int M, int N, int K, const float *A, const float *B, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp + C[x * N + y];
    }
}

// launcher
void launch_sgemm_naive(int M, int N, int K, const float *d_A, const float *d_B, float *d_C, dim3 gridDim, dim3 blockDim) {

    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
} 