#include "cache.cuh"

// #if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
//
// #include <cuda_fp8.h>

#include "quant/qdq_util.cuh"
#include "util.cuh"

#define THREADS 32

// The upper 8 bits of FP16 are equivalent to FP8 E5M2.
//
// The range of values typically cached seem to be in the range of +/- 16, with an exponent component (with bias) up to
// about 20. Empirically, the MSE over the whole range of observed values in the K/V cache works out the same for E4M3
// and E5M2. However, over 80% of values in the cache tensors fall within the range of -1..1, where E5M2 produces about
// a 25% lower MSE.

__device__ inline uint32_t compress(uint32_t v)
{
    uint32_t vh = (v & 0xff000000) >> 16;
    uint32_t vl = (v & 0x0000ff00) >> 8;
    return vh | vl;
}

__device__ inline uint32_t decompress(uint32_t v)
{
    uint32_t vh = (v & 0xff00) << 16;
    uint32_t vl = (v & 0x00ff) << 8;
    return vh | vl;
}

__global__ void nv_fp16_to_fp8
(
    const half* __restrict__ pIn,
    unsigned char* __restrict__ pOut,
    int stride,
    int height,
    int min,
    int max
)
{
    int x = min + (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int y = blockIdx.y;
    if (x >= max) return;

    int4* in_ptr = (int4*) (pIn + y * stride + x);
    int2* out_ptr = (int2*) (pOut + y * stride + x);

    int4 in = *in_ptr;
    uint32_t c0 = compress(in.x);
    uint32_t c1 = compress(in.y);
    uint32_t c2 = compress(in.z);
    uint32_t c3 = compress(in.w);
    int2 out = make_int2(c0 | (c1 << 16), c2 | (c3 << 16));
    *out_ptr = out;
}

__global__ void nv_fp8_to_fp16
(
    const unsigned char* __restrict__ pIn,
    half* __restrict__ pOut,
    int stride,
    int height,
    int min,
    int max
)
{
    int x = min + (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int y = blockIdx.y;
    if (x >= max) return;

    int2* in_ptr = (int2*) (pIn + y * stride + x);
    int4* out_ptr = (int4*) (pOut + y * stride + x);

    int2 in = *in_ptr;
    uint32_t c0 = decompress(in.x);
    uint32_t c1 = decompress(in.x >> 16);
    uint32_t c2 = decompress(in.y);
    uint32_t c3 = decompress(in.y >> 16);
    int4 out = make_int4(c0, c1, c2, c3);
    *out_ptr = out;
}

// __global__ void nv_fp32_to_fp16(const float* pIn, half* pOut, int size)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size) {
//         pOut[i] = __float2half(pIn[i]);
//     }
// }

// __global__ void nv_fp16_to_fp8_ref(const half* pIn, unsigned char *pOut, int size)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size) {
//         pOut[i] = __nv_cvt_halfraw_to_fp8(pIn[i], __NV_SATFINITE, __NV_E4M3);
//     }
// }
//
// __global__ void nv_fp8_to_fp16_ref(const unsigned char* pIn, half* pOut, int size)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size) {
//         pOut[i] = __nv_cvt_fp8_to_halfraw(pIn[i], __NV_E4M3);
//     }
// }

void array_fp16_to_fp8_cuda(const half* pIn, unsigned char *pOut, int stride, int height, int offset, int width)
{
    int min = offset;
    int max = offset + width;
    min = min / 8 * 8;
    max = min + (max - min + 7) / 8 * 8;

    dim3 blockDim, gridDim;
    blockDim.x = THREADS;
    gridDim.x = DIVIDE((max - min) / 8, THREADS);
    gridDim.y = height;

    nv_fp16_to_fp8<<<gridDim, blockDim>>>(pIn, pOut, stride, height, min, max);
    // cuda_check( cudaPeekAtLastError() );
}

void array_fp8_to_fp16_cuda(const unsigned char* pIn, half* pOut, int stride, int height, int offset, int width)
{
    int min = offset;
    int max = offset + width;
    min = min / 8 * 8;
    max = min + (max - min + 7) / 8 * 8;

    dim3 blockDim, gridDim;
    blockDim.x = THREADS;
    gridDim.x = DIVIDE((max - min) / 8, THREADS);
    gridDim.y = height;

    nv_fp8_to_fp16<<<gridDim, blockDim>>>(pIn, pOut, stride, height, min, max);
    // cuda_check( cudaPeekAtLastError() );
}

// void array_fp16_to_fp8_ref_cuda(const half* pIn, unsigned char *pOut, int size)
// {
//     const int threads = 512;
//     int blocks = DIVIDE(size / 1, threads);
//     nv_fp16_to_fp8_ref<<<blocks, threads>>>(pIn, pOut, size);
// }
//
// void array_fp8_to_fp16_ref_cuda(const unsigned char* pIn, half* pOut, int size)
// {
//     const int threads = 512;
//     int blocks = DIVIDE(size / 1, threads);
//     nv_fp8_to_fp16_ref<<<blocks, threads>>>(pIn, pOut, size);
// }

// #else
//
// void array_fp16_to_fp8_cuda(const half* pIn, unsigned char *pOut, int size) { }
//
// void array_fp8_to_fp16_cuda(const unsigned char* pIn, half* pOut, int size) { }
//
// #endif