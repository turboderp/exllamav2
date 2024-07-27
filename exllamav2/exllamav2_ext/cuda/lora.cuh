#ifndef _lora_cuh
#define _lora_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "q_matrix.cuh"

void apply_loras_cuda
(
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    const std::unordered_map<uintptr_t, std::tuple<half*, half*, int>>& adapters,
    const std::vector<uintptr_t>& ids,
    QMatrix* base,
    const half* input,
    half* output,
    half* temp,
    int rows
);

#endif
