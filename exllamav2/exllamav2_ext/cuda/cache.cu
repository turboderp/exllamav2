#include "cache.cuh"

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080

#include <cuda_fp8.h>

// TODO: Kernel profiling

__global__ void nv_fp16_to_fp8(const half* pIn, unsigned char *pOut, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        pOut[i] = __nv_cvt_halfraw_to_fp8(pIn[i], __NV_SATFINITE, __NV_E4M3);
    }
}

__global__ void nv_fp8_to_fp16(const unsigned char* pIn, half* pOut, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        pOut[i] = __nv_cvt_fp8_to_halfraw(pIn[i], __NV_E4M3);
    }
}

__global__ void nv_fp32_to_fp16(const float* pIn, half* pOut, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        pOut[i] = __float2half(pIn[i]);
    }
}

void array_fp16_to_fp8_cuda(const half* pIn, unsigned char *pOut, int size) {
    const int threads = 512;
    int blocks = (size + threads - 1) / threads;
    nv_fp16_to_fp8<<<blocks, threads>>>(pIn, pOut, size);
}

void array_fp8_to_fp16_cuda(const unsigned char* pIn, half* pOut, int size) {
    const int threads = 512;
    int blocks = (size + threads - 1) / threads;
    nv_fp8_to_fp16<<<blocks, threads>>>(pIn, pOut, size);
}

#else

void array_fp16_to_fp8_cuda(const half* pIn, unsigned char *pOut, int size) { }

void array_fp8_to_fp16_cuda(const unsigned char* pIn, half* pOut, int size) { }

#endif