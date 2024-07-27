#ifndef _h_gemm_cuh
#define _h_gemm_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

// alpha * ( a[m,k] @ b[k,n] ) + beta * c[m,n] -> c[m,n]

void h_gemm_cuda
(
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    const int size_m,
    const int size_n,
    const int size_k,
    const half* a,
    const half* b,
    half* c,
    const float alpha,
    const float beta
);

void h_gemm_cublas
(
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    const int size_m,
    const int size_n,
    const int size_k,
    const half* a,
    const half* b,
    half* c,
    const float alpha,
    const float beta
);

#endif

