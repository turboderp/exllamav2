
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#define STLOADER_BLOCK_SIZE (1*1024*1024)
#define STLOADER_THREADS 8

void stloader_read
(
    const char* filename,
    size_t offset,
    size_t size,
    torch::Tensor target
);

void tensor_remap
(
    torch::Tensor tensor,
    torch::Tensor index
);

void tensor_remap_4bit
(
    torch::Tensor tensor,
    torch::Tensor index
);