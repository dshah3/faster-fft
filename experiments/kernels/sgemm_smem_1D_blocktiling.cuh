#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cassert>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_smem_1D_blocktiling(int M, int N, int K, const float *A, const float *B, float *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const uint threadRow = threadIdx.x / BN;
    const uint threadCol = threadIdx.x % BN;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;

    float threadResults[TM] = {0.0};
    for (uint blkIdx = 0; blkIdx < K; blkIdx += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] += threadResults[resIdx];
    }

}