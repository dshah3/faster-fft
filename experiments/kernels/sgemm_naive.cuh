#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// kernel

// this is without alpha and beta

/*

Matrix sizes:
MxK * KxN = MxN

*/

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void sgemm_naive(int M, int N, int K, const float *A,
                            const float *B, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = tmp * C[x * N + y];
  }
}