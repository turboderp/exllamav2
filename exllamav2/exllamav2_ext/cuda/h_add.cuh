#ifndef _h_add_cuh
#define _h_add_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

void cuda_vector_add_
(
    cudaStream_t stream,
    half* dest,
    const half* source,
    int width,
    int height
);

void cuda_vector_set_
(
    cudaStream_t stream,
    half* dest,
    const half* source,
    int width,
    int height
);

#endif