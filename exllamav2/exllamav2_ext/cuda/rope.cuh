#ifndef _rope_cuh
#define _rope_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

void rope_cuda
(
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
    const bool neox_style
);

#endif
