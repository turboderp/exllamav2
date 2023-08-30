#ifndef _torch_attn_h
#define _torch_attn_h

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cstdio>

torch::Tensor torch_attn
(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor temp_q,
    int past_len,
    int q_len,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    torch::Tensor attn_mask
);

#endif