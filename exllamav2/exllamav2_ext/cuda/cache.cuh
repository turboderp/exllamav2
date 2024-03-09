#ifndef _cache_cuh
#define _cache_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

void array_fp16_to_fp8_cuda(const half* pIn, unsigned char* pOut, int stride, int height, int offset, int width);
void array_fp8_to_fp16_cuda(const unsigned char* pIn, half* pOut, int stride, int height, int offset, int width);

void array_fp16_to_q4_kv_cuda
(
    const half* k_in,
    unsigned char* k_out,
    half* k_scales,
    const half* v_in,
    unsigned char* v_out,
    half* v_scales,
    int stride,
    int height,
    int offset,
    int width
);

void array_q4_to_fp16_kv_cuda
(
    const unsigned char* k_in,
    const half* k_scales,
    half* k_out,
    const unsigned char* v_in,
    const half* v_scales,
    half* v_out,
    int stride,
    int height,
    int offset,
    int width
);

// void array_fp16_to_fp8_ref_cuda(const half* pIn, unsigned char *pOut, int size);
// void array_fp8_to_fp16_ref_cuda(const unsigned char* pIn, half* pOut, int size);

#endif
