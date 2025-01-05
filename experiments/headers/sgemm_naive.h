#ifndef SGEMM_NAIVE_H
#define SGEMM_NAIVE_H

__global__ void sgemm_naive(int M, int N, int K, const float *A, const float *B, float *C);

void launch_sgemm_naive(int M, int N, int K, const float *d_A, const float *d_B, float *d_C, dim3 gridDim, dim3 blockDim);

#endif