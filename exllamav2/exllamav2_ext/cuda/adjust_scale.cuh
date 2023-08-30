#ifndef _adjust_scale_cuh
#define _adjust_scale_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

float adjust_scale_cuda
(
    const float* qscale,
    const float* x,
    float* grid,
    const float max_adjust,
    const float min_adjust,
    const int adjust_steps,
    const int rows,
    const int columns,
    const float norm,
    const int qzero,
    const int maxq
);

#endif