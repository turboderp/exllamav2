#ifndef _cache_cuh
#define _cache_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

void array_fp16_to_fp8_cuda
(
    cudaStream_t stream,
    const half* pIn,
    unsigned char* pOut,
    int stride, int height,
    int offset, int width
);

void array_fp8_to_fp16_cuda
(
    cudaStream_t stream,
    const unsigned char* pIn,
    half* pOut,
    int stride,
    int height,
    int offset,
    int width
);

void array_fp16_to_q_kv_cuda
(
    cudaStream_t stream,
    const half* k_in,
    unsigned char* k_out,
    half* k_scales,
    const half* v_in,
    unsigned char* v_out,
    half* v_scales,
    int dim,
    int stride,
    int height,
    int offset,
    int width,
    const half* cal_k,
    const half* cal_v,
    int wbits
);

void array_q_to_fp16_kv_cuda
(
    cudaStream_t stream,
    const unsigned char* k_in,
    const half* k_scales,
    half* k_out,
    const unsigned char* v_in,
    const half* v_scales,
    half* v_out,
    int dim,
    int stride,
    int height,
    int offset,
    int width,
    const half* cal_k,
    const half* cal_v,
    int wbits
);

void array_fp16_to_q_kv_paged_cuda
(
    cudaStream_t stream,
    const half* k_in,
    unsigned char* k_out,
    half* k_scales,
    const half* v_in,
    unsigned char* v_out,
    half* v_scales,
    int batch_size,
    int dim,
    int pages_per_seq,
    const int* cache_seqlens,
    const int* block_table,
    int page_size,
    int q_len,
    const half* cal_k,
    const half* cal_v,
    int wbits
);

void array_q_to_fp16_kv_paged_cuda
(
    cudaStream_t stream,
    const unsigned char* k_in,
    const half* k_scales,
    half* k_out,
    const unsigned char* v_in,
    const half* v_scales,
    half* v_out,
    int batch_size,
    int dim,
    int pages_per_seq,
    const int* cache_seqlens,
    const int* block_table,
    int page_size,
    const half* cal_k,
    const half* cal_v,
    int wbits
);

// void array_fp16_to_fp8_ref_cuda(const half* pIn, unsigned char *pOut, int size);
// void array_fp8_to_fp16_ref_cuda(const unsigned char* pIn, half* pOut, int size);

#endif
