#ifndef _cache_cuh
#define _cache_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

void array_fp16_to_fp8_cuda(const half* pIn, unsigned char *pOut, int stride, int height, int offset, int width);
void array_fp8_to_fp16_cuda(const unsigned char* pIn, half* pOut, int stride, int height, int offset, int width);
// void array_fp16_to_fp8_ref_cuda(const half* pIn, unsigned char *pOut, int size);
// void array_fp8_to_fp16_ref_cuda(const unsigned char* pIn, half* pOut, int size);

#endif
