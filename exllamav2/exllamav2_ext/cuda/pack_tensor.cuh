#ifndef _pack_tensor_cuh
#define _pack_tensor_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

void pack_rows_4_cuda
(
    const uint16_t* input,
    uint32_t* output,
    const int rows,
    const int columns
);

void pack_rows_6_cuda
(
    const uint16_t* input,
    uint32_t* output,
    const int rows,
    const int columns
);

void pack_columns_cuda
(
    const uint16_t* input,
    uint32_t* output,
    const int in_rows,
    const int out_rows,
    const int columns,
    const int bits
);

#endif