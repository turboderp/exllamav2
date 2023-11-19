#ifndef _rms_norm_cuh
#define _rms_norm_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

void rms_norm_cuda
(
    const half* x,
    const half* w,
    half* y,
    const float epsilon,
    const int rows,
    const int dim
);

#endif