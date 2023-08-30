#ifndef _quantize_func_h
#define _quantize_func_h

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cstdio>

void quantize_range
(
    torch::Tensor quant,
    torch::Tensor scale,
    torch::Tensor out_q,
    float qzero,
    float maxq,
    torch::Tensor hessian_inv,
    torch::Tensor weights,
    torch::Tensor error,
    int a,
    int b
);

#endif