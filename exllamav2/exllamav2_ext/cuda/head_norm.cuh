#ifndef _head_norm_cuh
#define _head_norm_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include "graph.cuh"

void head_norm_cuda
(
    cudaStream_t stream,
    const half* x,
    const half* w,
    const half* b,
    half* y,
    const float epsilon,
    const int rows,
    const int num_heads,
    const int head_dim,
    Graph* graph = NULL,
    int label = 0
);

void head_norm_cuda_update_x
(
    Graph* graph,
    int label,
    void* x
);

void head_norm_cuda_update_y
(
    Graph* graph,
    int label,
    void* y
);

#endif