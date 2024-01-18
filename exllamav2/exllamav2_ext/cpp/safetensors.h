#ifndef _safetensors_h
#define _safetensors_h

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cstdio>

class STFile
{
public:
    STFile(const char* filename, uintptr_t pinned_buffer);
    ~STFile();

    void fastload
    (
        std::vector<torch::Tensor>& targets,
        std::vector<size_t> offsets,
        std::vector<size_t> lengths,
        size_t h_offset
    );

    int file_descriptor;
    size_t filesize;
    size_t padded_size;
    size_t block_size;
    void* aligned_buffer;
};

uintptr_t safetensors_open(const char* filename, uintptr_t pinned_buffer);
void safetensors_close(uintptr_t handle);
uintptr_t safetensors_pinned_buffer();
void safetensors_free_pinned_buffer(uintptr_t b);

void safetensors_fastload
(
    uintptr_t handle,
    std::vector<torch::Tensor>& targets,
    std::vector<size_t> offsets,
    std::vector<size_t> lengths,
    size_t h_offset
);

#endif