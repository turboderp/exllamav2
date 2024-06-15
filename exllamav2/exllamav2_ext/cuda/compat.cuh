#ifndef _compat_cuh
#define _compat_cuh

// atomicAdd for half types, to support CC < 7.x

__device__ __forceinline__ void atomicAdd_half(half* address, half val)
{
    unsigned int * address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do
    {
        assumed = old;
        __half_raw hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        half tmpres = __hadd(hsum, val);
        hsum = __half_raw(tmpres);
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
    }
    while (assumed != old);
}

// atomicAdd for half2 types

__device__ __forceinline__ void atomicAdd_half2(half2* address, half2 val)
{
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do
    {
        assumed = old;
        half2 old_val = *((half2*)&old);
        half2 new_val = __hadd2(old_val, val);
        old = atomicCAS(address_as_ui, assumed, *((unsigned int*)&new_val));
    }
    while (assumed != old);
}

//

#if defined(__CUDA_ARCH__) || defined(USE_ROCM)
#if __CUDA_ARCH__ < 700 || defined(USE_ROCM)

__device__ __forceinline__ void atomicAdd(half* address, half val) { atomicAdd_half(address, val); }

#if __CUDA_ARCH__ < 600 || defined(USE_ROCM)
__device__ __forceinline__ void atomicAdd(half2* address, half2 val) { atomicAdd_half2(address, val); }
#endif

#endif
#endif

// Approximate tanh

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750 || CUDART_VERSION < 11000))

__inline__ __device__ float tanh_opt(float x)
{
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
}

#else

__inline__ __device__ float tanh_opt(float x)
{
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
}

#endif

// ROCm redefinitions

#if defined(USE_ROCM)

#define __shfl_xor_sync(mask, var, laneMask) __shfl_xor(var, laneMask)
#define __shfl_down_sync(mask, var, laneMask) __shfl_down(var, laneMask)

__device__ __forceinline__ __half2 __compat_h2rcp(__half2 x)
{
    return __halves2half2
    (
         hrcp(__low2half(x)),
         hrcp(__high2half(x))
    );
}
#define h2rcp __compat_h2rcp

__device__ __forceinline__ __half2 __compat_hmax2(__half2 x, __half2 y)
{
    return __halves2half2
    (
        __hmax(__low2half(x), __low2half(y)),
        __hmax(__high2half(x), __high2half(y))
    );
}
#define __hmax2 __compat_hmax2

#define __stwb(dst, src) *dst = src
#define __stcg(dst, src) *dst = src

#endif

#endif
