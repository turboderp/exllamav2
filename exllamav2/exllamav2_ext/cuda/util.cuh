
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#define DBGS(__x) printf("%s\n", __x)
#define DBGI(__x) printf("%s: %i\n", #__x, __x)
#define DBGI2(__x, __y) printf("%s, %s: %i, %i\n", #__x, #__y, __x, __y)
#define DBGI3(__x, __y, __z) printf("%s, %s, %s: %i, %i, %i\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGF(__x) printf("%s: %f\n", #__x, __x)
#define DBGF2(__x, __y) printf("%s, %s: %f, %f\n", #__x, #__y, __x, __y)
#define DBGF3(__x, __y, __z) printf("%s, %s, %s: %f, %f, %f\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGH(__x) printf("%s: %f\n", #__x, __half2float(__x))
#define DBGH2(__x, __y) printf("%s, %s: %f, %f\n", #__x, #__y, __half2float(__x), __half2float(__y))
#define DBGH3(__x, __y, __z) printf("%s, %s, %s: %f, %f, %f\n", #__x, #__y, #__z, __half2float(__x), __half2float(__y), __half2float(__z))

#define DBGIH(__x, __y) printf("%s, %s: %i, %f\n", #__x, #__y, __x, __half2float(__y))

__forceinline__ __device__ half dq_scale_(const int qs, const half max_scale)
{
    half qs_h = __hmul(__int2half_rn(qs + 1), __float2half_rn(1.0f / 16.0f));
    qs_h = __hmul(qs_h, qs_h);
    qs_h = __hmul(qs_h, max_scale);
    return qs_h;
}

// Max_scale premultiplied by 1/256

__forceinline__ __device__ half dq_scale(const int qs, const half max_scale)
{
    int qs_i = qs + 1;
    half qs_h = __int2half_rn(qs_i * qs_i);
    qs_h = __hmul(qs_h, max_scale);
    return qs_h;
}

__forceinline__ __device__ float clamp(float x, float a, float b)
{
    return fmaxf(a, fminf(b, x));
}

__forceinline__ __device__ half dq(const int q, const int qzero, const half scale)
{
    return __hmul(__int2half_rn(q - qzero), scale);
}

__forceinline__ __device__ half dq_ns(const int q, const int qzero)
{
    //return __hsub(__int2half_rn(q), __int2half_rn(qzero));
    return __int2half_rn(q - qzero);
}

__forceinline__ __device__ int exb(const uint32_t q, const int shift, const int mask)
{
    return (int)((q >> shift) & mask);
}

__forceinline__ __device__ int exb(const uint32_t q1, const uint32_t q0, const int shift, const int mask)
{
    return (int)(__funnelshift_r(q0, q1, shift) & mask);
}

