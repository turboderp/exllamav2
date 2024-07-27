#ifndef _softcap_cuh
#define _softcap_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

void softcap_cuda_
(
    cudaStream_t stream,
    float* x,
    const uint64_t numel,
    const float scale
);

void h_softcap_cuda_
(
    cudaStream_t stream,
    half* x,
    const uint64_t numel,
    const float scale
);

#endif