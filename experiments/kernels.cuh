#pragma once

#include "kernels/sgemm_naive.cuh"
#include "kernels/sgemm_coalesce.cuh"
#include "kernels/sgemm_smem.cuh"
#include "kernels/sgemm_smem_1D_blocktiling.cuh"
// #include "kernels/5_kernel_2D_blocktiling.cuh"
// #include "kernels/6_kernel_vectorize.cuh"
// #include "kernels/7_kernel_resolve_bank_conflicts.cuh"
// #include "kernels/8_kernel_bank_extra_col.cuh"
// #include "kernels/9_kernel_autotuned.cuh"
// #include "kernels/10_kernel_warptiling.cuh"
// #include "kernels/11_kernel_double_buffering.cuh"
// #include "kernels/12_kernel_double_buffering.cuh"