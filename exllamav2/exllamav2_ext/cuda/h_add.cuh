#ifndef _h_add_cuh
#define _h_add_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

void cuda_vector_add_
(
    half* dest,
    const half* source,
    int width,
    int height
);

#endif