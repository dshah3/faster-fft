#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cublas_v2.h>


// kernel without alpha and beta

/*

Matrix sizes;
MxK * KxN = MxN

Shared memory access

*/

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const uint BLOCKSIZE>
__global__ void sgemm_smem(int M, int N, int K, const float *A, const float *B, float *C) {

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0;
    for (int blkIdx = 0; blkIdx < K; blkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int i = 0; i < BLOCKSIZE; ++i) {
            tmp += As[threadRow * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] += tmp;

}