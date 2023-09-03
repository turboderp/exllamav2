#ifndef _rope_cuh
#define _rope_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

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
    const uint32_t* past_lens
);

#endif
