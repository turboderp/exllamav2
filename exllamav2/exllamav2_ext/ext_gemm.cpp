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
#include "ext_gemm.h"

#include "cuda/h_gemm.cuh"

#include "cpp/util.h"

void gemm_half_half_half
(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    const float alpha,
    const float beta,
    bool force_cublas
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(c, kHalf);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK_SHAPES(a, 1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, 1, 1);

    if (force_cublas)
    {
        h_gemm_cublas
        (
            stream,
            at::cuda::getCurrentCUDABlasHandle(),
            c.size(0), // m
            c.size(1), // n
            a.size(1), // k
            (const half*) a.data_ptr(),
            (const half*) b.data_ptr(),
            (half*) c.data_ptr(),
            alpha,
            beta
        );
    }
    else
    {
        h_gemm_cuda
        (
            stream,
            at::cuda::getCurrentCUDABlasHandle(),
            c.size(0), // m
            c.size(1), // n
            a.size(1), // k
            (const half*) a.data_ptr(),
            (const half*) b.data_ptr(),
            (half*) c.data_ptr(),
            alpha,
            beta
        );
    }
}
