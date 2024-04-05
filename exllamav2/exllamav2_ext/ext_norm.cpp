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
#include "ext_norm.h"

#include "cuda/rms_norm.cuh"
#include "cuda/layer_norm.cuh"
#include "cuda/head_norm.cuh"

#include "cpp/util.h"


// RMS layernorm

void rms_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    float epsilon
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);
    TORCH_CHECK_SHAPES(x, 0, y, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, y, 1, 1);

    int rows = x.size(0);
    int dim = x.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    rms_norm_cuda
    (
        (half*) x.data_ptr(),
        (half*) w.data_ptr(),
        (half*) y.data_ptr(),
        epsilon,
        rows,
        dim
    );
}

void rms_norm_
(
    torch::Tensor x,
    torch::Tensor w,
    float epsilon
)
{
    rms_norm(x, w, x, epsilon);
}


// Layernorm

void layer_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor y,
    float epsilon
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE_OPT(b, kHalf);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);
    TORCH_CHECK_SHAPES(x, 0, y, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, y, 1, 1);
    TORCH_CHECK_SHAPES_OPT(b, 0, w, 0, 1);

    int rows = x.size(0);
    int dim = x.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    layer_norm_cuda
    (
        (half*) x.data_ptr(),
        (half*) w.data_ptr(),
        b.device().is_meta() ? NULL : (half*) b.data_ptr(),
        (half*) y.data_ptr(),
        epsilon,
        rows,
        dim
    );
}

void layer_norm_
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    float epsilon
)
{
    layer_norm(x, w, b, x, epsilon);
}


// Head norm

void head_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor y,
    float epsilon
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE_OPT(b, kHalf);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_SHAPES(x, -1, w, -1, 1);
    TORCH_CHECK_SHAPES(x, -2, w, -2, 1);
    TORCH_CHECK_SHAPES(x, 0, y, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, y, 1, 1);
    TORCH_CHECK_SHAPES_OPT(b, 0, w, 0, 1);
    TORCH_CHECK_SHAPES_OPT(b, 1, w, 1, 1);

    int head_dim = x.size(-1);
    int num_heads = x.size(-2);
    int rows = x.numel() / head_dim / num_heads;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    head_norm_cuda
    (
        (half*) x.data_ptr(),
        (half*) w.data_ptr(),
        b.device().is_meta() ? NULL : (half*) b.data_ptr(),
        (half*) y.data_ptr(),
        epsilon,
        rows,
        num_heads,
        head_dim
    );
}

void head_norm_
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    float epsilon
)
{
    head_norm(x, w, b, x, epsilon);
}
