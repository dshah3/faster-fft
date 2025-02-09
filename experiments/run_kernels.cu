#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errorLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  CudaDeviceInfo();

  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  }
  
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;
  
  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr;
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr,
        *d_C_ref = nullptr;

    A = (float *)malloc(sizeof(float) * max_size * max_size);
    B = (float *)malloc(sizeof(float) * max_size * max_size);
    C = (float *)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size);
    randomize_matrix(B, max_size);
    randomize_matrix(C, max_size);

    cudaCheck(cudaMalloc((void **)&d_A, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&d_B, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&d_C, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&d_C_ref, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(d_A, A, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, B, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_C, C, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_C_ref, C, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));

    int repeat_times = 50;
    for (int size : SIZE) {
      m = n = k = size;

      std::cout << "dimensions(m=n=k) " << m << std::endl;
      if (kernel_num != 0) {
        run_kernel(0, m, n, k, d_A, d_B, d_C_ref, handle);
        run_kernel(kernel_num, m, n, k, d_A, d_B, d_C, handle);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaGetLastError());
        cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, d_C_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

        if (!verify_matrix(C_ref, C, m * n)) {
          std::cout
              << "Failed to pass the correctness verification against NVIDIA "
                "cuBLAS."
              << std::endl;
          if (m <= 128) {
            std::cout << " Logging faulty output into " << errorLogFile << "\n";
          }
          exit(EXIT_FAILURE);
        }
      }

      cudaEventRecord(beg);
      for (int j = 0; j < repeat_times; j++) {
        // We don't reset dC between runs to save time
        run_kernel(kernel_num, m, n, k, d_A, d_B, d_C, handle);
      }

      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, beg, end);
      elapsed_time /= 1000.;

      long flops = 2 * m * n * k;
      printf(
          "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
          "(%ld).\n",
          elapsed_time / repeat_times,
          (repeat_times * flops * 1e-9) / elapsed_time, m);
      fflush(stdout);
      // make dC and dC_ref equal again (we modified dC while calling our kernel
      // for benchmarking)
      cudaCheck(cudaMemcpy(d_C, d_C_ref, sizeof(float) * m * n,
                          cudaMemcpyDeviceToDevice));
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    cublasDestroy(handle);

    return 0;

}