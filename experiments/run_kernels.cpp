#include <cuda_runtime.h>
#include <iostream>
#include "experiments/headers/sgemm_naive.h"
#include <cmath>
#include <random>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

int main() {
    
    int M = 1024, N = 1024, K = 1024;

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(dis(gen));
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(dis(gen));

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start, 0);

    // Launch kernel
    launch_sgemm_naive(M, N, K, d_A, d_B, d_C, gridDim, blockDim);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute reference result on CPU
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(h_C[i] - h_C_ref[i]);
        max_error = fmax(max_error, error);
    }
    std::cout << "Max error: " << max_error << std::endl;
    
    // Clean up host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return 0;
}