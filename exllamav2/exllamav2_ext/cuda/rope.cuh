#ifndef _rope_cuh
#define _rope_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include "graph.cuh"

void rope_cuda
(
    cudaStream_t stream,
    half* x,
    const half* sin,
    const half* cos,
    const int batch_size,
    const int rows_per_batch,
    const int head_dim,
    const int num_heads,
    const int past_len,
    const int32_t* past_lens,
    const bool neox_style
);

void rope_cuda_qk
(
    cudaStream_t stream,
    half* x_q,
    half* x_k,
    const half* sin,
    const half* cos,
    const int batch_size,
    const int rows_per_batch_q,
    const int rows_per_batch_k,
    const int head_dim,
    const int num_heads_q,
    const int num_heads_k,
    const int past_len,
    const int32_t* past_lens,
    const bool neox_style,
    Graph* graph = NULL,
    int label = 0
);

void rope_cuda_qk_update_q
(
    Graph* graph,
    int label,
    void* q
);

void rope_cuda_qk_update_k
(
    Graph* graph,
    int label,
    void* k
);

void rope_cuda_qk_update_past_len
(
    Graph* graph,
    int label,
    int past_len
);

void rope_cuda_qk_update_past_lens
(
    Graph* graph,
    int label,
    void* past_lens
);

#endif
