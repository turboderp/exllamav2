#ifndef _rms_norm_cuh
#define _rms_norm_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include "graph.cuh"

void rms_norm_cuda
(
    cudaStream_t stream,
    const void* x,
    const half* w,
    void* y,
    const float epsilon,
    const int rows,
    const int dim,
    const bool add_residual = false,
    const bool input_fp32 = false,
    const bool output_fp32 = false,
    Graph* graph = NULL,
    int label = 0
);

void rms_norm_cuda_update_x
(
    Graph* graph,
    int label,
    void* x
);

void rms_norm_cuda_update_y
(
    Graph* graph,
    int label,
    void* y
);

#endif