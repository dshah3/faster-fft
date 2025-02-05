#include "kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void randomize_matrix(float *matrix, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    const int mean = 1;
    const int std = 0;
    std::normal_distribution<float> dist(mean, std);
    
    for (int i = 0; i < cols * cols; i++) {
        matrix[i] = dist(gen);
    }
}

bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // cuBLAS uses column-major order. So we change the order of our row-major A &
  // B, since (B^T*A^T)^T = (A*B)
  // This runs cuBLAS in full fp32 mode
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasBF16(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // This runs cuBLAS with mixed precision (performing the mul with operands
  // downcast to bf16), which is ~4x faster
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasTF32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // This runs cuBLAS with mixed precision (performing the mul with operands
  // downcast to bf16), which is ~4x faster
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


void run_sgemm_naive(int M, int N, int K, float *A, float *B, float *C) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}

void run_sgemm_coalesce(int M, int N, int K, float *A, float *B, float *C) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);
    sgemm_coalesce<32><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}

void run_kernel(int kernel_num, int M, int N, int K, float *A, float *B, float *C, cublasHandle_t handle) {
    switch(kernel_num) {
        case 0:
            runCublasFP32(handle, M, N, K, 1.0, A, B, 1.0, C);
            break;
        case 1:
            run_sgemm_naive(M, N, K, A, B, C);
            break;
        case 2:
            run_sgemm_coalesce(M, N, K, A, B, C);
            break;
        default:
            throw std::invalid_argument("Unknown kernel number");
    }
}