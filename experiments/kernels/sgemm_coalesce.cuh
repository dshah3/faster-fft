#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// kernel

// this is without alpha and beta

/*

Matrix sizes;
MxK * KxN = MxN

Global memory coalescing is simply just dividing threadIdx.x and modulo threadIdx.y

*/

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const uint BLOCKSIZE>
__global__ void sgemm_coalesce(int M, int N, int K, const float *A, const float *B, float *C) {
    const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp + C[x * N + y];
    }
}