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
