#ifndef _head_norm_cuh
#define _head_norm_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

void head_norm_cuda
(
    const half* x,
    const half* w,
    const half* b,
    half* y,
    const float epsilon,
    const int rows,
    const int num_heads,
    const int head_dim
);

#endif