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
#include "ext_tp.h"

// RMS layernorm

void rms_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    float epsilon
)
{
    bool input_fp32 = x.dtype() == torch::kFloat;
    bool output_fp32 = y.dtype() == torch::kFloat;
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);
    TORCH_CHECK_SHAPES(x, 0, y, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, y, 1, 1);

    int rows = x.size(0);
    int dim = x.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    rms_norm_cuda
    (
        stream,
        (void*) x.data_ptr(),
        (half*) w.data_ptr(),
        (void*) y.data_ptr(),
        epsilon,
        rows,
        dim,
        false,
        input_fp32,
        output_fp32
    );
}

void rms_norm_tp
(
    std::vector<torch::Tensor> x,
    std::vector<torch::Tensor> w,
    std::vector<torch::Tensor> y,
    float epsilon,
    uintptr_t tp_context
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    int rows = x[0].size(0);
    int dim = x[0].size(1);

    for (int i = 0; i < x.size(); ++i)
    {
        int dev = x[i].device().index();
//        DBGI(dev);
//        DBGI(ctx->streams[dev]);
        cudaSetDevice(dev);
        rms_norm_cuda
        (
            ctx->streams[dev],
            (void*) x[i].data_ptr(),
            (half*) w[i].data_ptr(),
            (void*) y[i].data_ptr(),
            epsilon,
            rows,
            dim,
            false,
            false,  // TODO: FP32 residual
            false
        );
    }
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    layer_norm_cuda
    (
        stream,
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    head_norm_cuda
    (
        stream,
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
