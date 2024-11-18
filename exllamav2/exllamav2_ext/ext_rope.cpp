#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"

#include "cuda/rope.cuh"

#include "cpp/util.h"


// RoPE rotary positional embeddings, in-place

void rope_
(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos,
    int past_len,
    int num_heads,
    int head_dim,
    torch::Tensor offsets,
    bool neox_style
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(sin, kHalf);
    TORCH_CHECK_DTYPE(cos, kHalf);
    TORCH_CHECK(head_dim == cos.size(-1), "cos table does not match head_dim");
    TORCH_CHECK(head_dim == sin.size(-1), "sin table does not match head_dim");
    TORCH_CHECK_DTYPE_OPT(offsets, kInt);

    int batch_size = x.size(0);
    int rows_per_batch = x.numel() / head_dim / batch_size;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    rope_cuda
    (
        stream,
        (half*) x.data_ptr(),
        (const half*) sin.data_ptr(),
        (const half*) cos.data_ptr(),
        batch_size,
        rows_per_batch,
        head_dim,
        num_heads,
        past_len,
        offsets.device().is_meta() ? NULL : (int32_t*) offsets.data_ptr(),
        neox_style
    );
}

long gen_mrope_pos_ids
(
    torch::Tensor mrope_pos_ids,
    torch::Tensor ids,
    int merge_size,
    const std::vector<std::tuple<long, long>> &spans,
    const std::vector<std::tuple<long, long, long>> &grids
)
{
    int max_length = mrope_pos_ids.size(1);
    int in_length = ids.size(0);

    long* in_ids = (long*) ids.data_ptr();
    long* pos_ids = (long*) mrope_pos_ids.data_ptr();

    long* out_t = pos_ids;
    long* out_h = pos_ids + max_length;
    long* out_w = pos_ids + 2 * max_length;

    long base_t = 0;
    long next_base_t = 0;

    for (int i = 0; i < max_length; ++i)
    {
        bool is_emb = false;
        if (i < in_length)
        {
            long id = in_ids[i];

            for (int j = 0; j < spans.size(); ++j)
            {
                long span_start = std::get<0>(spans[j]);
                long span_end = std::get<1>(spans[j]);
                long span = span_end - span_start;
                if (id >= span_start && id < span_end)
                {
                    is_emb = true;
                    long k = id - span_start;
                    long grid_t = std::get<0>(grids[j]);
                    long grid_h = std::get<1>(grids[j]) / (long)merge_size;
                    long grid_w = std::get<2>(grids[j]) / (long)merge_size;
                    long k_t = base_t + (k / grid_w / grid_h) % grid_t;
                    long k_h = base_t + (k / grid_w) % grid_h;
                    long k_w = base_t + k % grid_w;
                    *out_t++ = k_t;
                    *out_h++ = k_h;
                    *out_w++ = k_w;
                    // DBGI3(k_t, k_h, k_w);
                    next_base_t = std::max(next_base_t, k_t + 1);
                    next_base_t = std::max(next_base_t, k_h + 1);
                    next_base_t = std::max(next_base_t, k_w + 1);
                    break;
                }
            }
        }
        if (!is_emb)
        {
            base_t = next_base_t;
            *out_t++ = base_t;
            *out_h++ = base_t;
            *out_w++ = base_t;
            // DBGI3(base_t, base_t, base_t);
            base_t++;
            next_base_t = base_t;
        }
    }

    return next_base_t;
}

