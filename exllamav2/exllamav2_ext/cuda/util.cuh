#ifndef _util_cuh
#define _util_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#define DBGS(__x) printf("%s\n", __x)
#define DBGI(__x) printf("%s: %i\n", #__x, __x)
#define DBGI2(__x, __y) printf("%s, %s: %i, %i\n", #__x, #__y, __x, __y)
#define DBGI3(__x, __y, __z) printf("%s, %s, %s: %i, %i, %i\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGI4(__x, __y, __z, __w) printf("%s, %s, %s, %s: %i, %i, %i, %i\n", #__x, #__y, #__z, #__w, __x, __y, __z, __w)
#define DBGX(__x) printf("%s: %x\n", #__x, __x)
#define DBGX2(__x, __y) printf("%s, %s: %x, %x\n", #__x, #__y, __x, __y)
#define DBGX3(__x, __y, __z) printf("%s, %s, %s: %x, %x, %x\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGIX(__x, __y) printf("%s, %s: %i, %x\n", #__x, #__y, __x, __y)
#define DBGIX2(__x, __y, __z) printf("%s, %s, %s: %i, %x, %x\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGIF(__x, __y) printf("%s, %s: %i, %f\n", #__x, #__y, __x, __y)
#define DBGF(__x) printf("%s: %f\n", #__x, __x)
#define DBGF2(__x, __y) printf("%s, %s: %f, %f\n", #__x, #__y, __x, __y)
#define DBGF3(__x, __y, __z) printf("%s, %s, %s: %f, %f, %f\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGH(__x) printf("%s: %f\n", #__x, __half2float(__x))
#define DBGH2(__x, __y) printf("%s, %s: %f, %f\n", #__x, #__y, __half2float(__x), __half2float(__y))
#define DBGH3(__x, __y, __z) printf("%s, %s, %s: %f, %f, %f\n", #__x, #__y, #__z, __half2float(__x), __half2float(__y), __half2float(__z))

#define DBGIH(__x, __y) printf("%s, %s: %i, %f\n", #__x, #__y, __x, __half2float(__y))
#define DBGIH2(__x, __y, __z) printf("%s, %s, %s: %i, %f, %f\n", #__x, #__y, #__z, __x, __half2float(__y), __half2float(__z))
#define DBGI2H2(__x, __y, __z, __w) printf("%s, %s, %s, %s: %i, %i, %f, %f\n", #__x, #__y, #__z, #__w, __x, __y, __half2float(__z), __half2float(__w))
#define DBGIH3(__x, __y, __z, __w) printf("%s, %s, %s, %s: %i, %f, %f, %f\n", #__x, #__y, #__z, #__w, __x, __half2float(__y), __half2float(__z), __half2float(__w))

__forceinline__ __device__ half dq_scale_(const int qs, const half max_scale)
{
    half qs_h = __hmul(__int2half_rn(qs + 1), __float2half_rn(1.0f / 16.0f));
    qs_h = __hmul(qs_h, qs_h);
    qs_h = __hmul(qs_h, max_scale);
    return qs_h;
}

__forceinline__ __device__ float clamp(float x, float a, float b)
{
    return fmaxf(a, fminf(b, x));
}

#define cuda_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void print_global_mem(const half* ptr, int rows, int columns, int stride);

__forceinline__ __device__ void print_smem(const uint8_t* ptr, int count)
{
    for (int i = 0; i < count; ++i)
    {
        printf("%4d ", *ptr++);
    }
    printf("\n");
}

__forceinline__ __device__ void print_smem(const half* ptr, int count)
{
    for (int i = 0; i < count; ++i)
    {
        printf("%6.3f", __half2float(*ptr++));
    }
    printf("\n");
}

//inline __device__ half2 ___hmax2(half2 a, half2 b)
//{
//    half a_low = __low2half(a);
//    half a_high = __high2half(a);
//    half b_low = __low2half(b);
//    half b_high = __high2half(b);
//
//    half max_low = __hgt(a_low, b_low) ? a_low : b_low;
//    half max_high = __hgt(a_high, b_high) ? a_high : b_high;
//
//    return __halves2half2(max_low, max_high);
//}

#endif