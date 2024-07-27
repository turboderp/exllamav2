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
#include "cuda/softcap.cuh"
#include "cpp/util.h"

// Apply softcapping inplace: x = scale * tanh(x/scale)

void softcap_
(
    torch::Tensor x,
    float scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    uint64_t numel = x.numel();

    if (x.dtype() == torch::kFloat)
    {
        softcap_cuda_
        (
            stream,
            (float*) x.data_ptr(),
            numel,
            scale
        );
    }
    else if (x.dtype() == torch::kHalf)
    {
        h_softcap_cuda_
        (
            stream,
            (half*) x.data_ptr(),
            numel,
            scale
        );
    }
    else
    {
        TORCH_CHECK(false, "softcap_ wrong dtype");
    }
}
