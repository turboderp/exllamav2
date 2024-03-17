#include "cache.cuh"

#include "quant/qdq_util.cuh"
#include "util.cuh"
#include "compat.cuh"

#define THREADS 32
#define BLOCKSIZE_Q 256
#define THREADS_Q (BLOCKSIZE_Q / 2)

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

__global__ void fp16_to_fp8_kernel
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

__global__ void fp8_to_fp16_kernel
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

    fp16_to_fp8_kernel<<<gridDim, blockDim>>>(pIn, pOut, stride, height, min, max);
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

    fp8_to_fp16_kernel<<<gridDim, blockDim>>>(pIn, pOut, stride, height, min, max);
    // cuda_check( cudaPeekAtLastError() );
}

// Q4

__global__ void fp16_to_q4_kv_kernel
(
    const half* __restrict__ k_in,
    unsigned char* __restrict__ k_out,
    half* __restrict__ k_scales,
    const half* __restrict__ v_in,
    unsigned char* __restrict__ v_out,
    half* __restrict__ v_scales,
    int offset,
    int stride
)
{
    int t = threadIdx.x;
    const half* in = blockIdx.z ? v_in : k_in;
    unsigned char* out = blockIdx.z ? v_out : k_out;
    half* scales = blockIdx.z ? v_scales : k_scales;

    int block_offset = (offset + blockIdx.y * stride + blockIdx.x * BLOCKSIZE_Q);
    const half2* in2 = (const half2*) (in + block_offset);
    __shared__ uint32_t q_buffer[BLOCKSIZE_Q / 8];
    __shared__ half s_buffer[BLOCKSIZE_Q / 32];

    half2 w2 = in2[t];
    half2 o = w2;

    // Max abs value for lane_id 0..15, 16..31

    half2 absmax2 = __habs2(w2);
    half absmax = __hmax(__low2half(absmax2), __high2half(absmax2));
    absmax = __hmax(absmax, __shfl_xor_sync(0xffffffff, absmax, 8));
    absmax = __hmax(absmax, __shfl_xor_sync(0xffffffff, absmax, 4));
    absmax = __hmax(absmax, __shfl_xor_sync(0xffffffff, absmax, 2));
    absmax = __hmax(absmax, __shfl_xor_sync(0xffffffff, absmax, 1));
    absmax2 = __half2half2(absmax);

    // Normalize

    half2 c_8 = __half2half2(__int2half_rn(8));
    half c_i = __float2half_rn(1.0f / 8.0f);

    w2 = __h2div(w2, absmax2);
    w2 = __hfma2(w2, c_8, c_8);

    // Quantize & pack

    int q0 = clamp(__half2int_rn(__low2half(w2)), 0, 15);
    int q1 = clamp(__half2int_rn(__high2half(w2)), 0, 15);
    uint32_t q = q0 | (q1 << 4);

    q |= (__shfl_down_sync(0x55555555, q, 1) << 8);
    q |= (__shfl_down_sync(0x11111111, q, 2) << 16);
    if (t % 4 == 0) q_buffer[t / 4] = q;
    if (t % 16 == 0) s_buffer[t / 16] = __hmul(absmax, c_i);
    __syncthreads();

    // Store

    int4* pq = (int4*) q_buffer;
    int4* ps = (int4*) s_buffer;
    int4* out_q = (int4*) (out + block_offset / 2);
    int4* out_s = (int4*) (scales + block_offset / 32);

    if (t < BLOCKSIZE_Q / 32) out_q[t] = pq[t];
    if (t < BLOCKSIZE_Q / 256) out_s[t] = ps[t];
}

__global__ void q4_to_fp16_kv_kernel
(
    const unsigned char* __restrict__ k_in,
    const half* __restrict__ k_scales,
    half* __restrict__ k_out,
    const unsigned char* __restrict__ v_in,
    const half* __restrict__ v_scales,
    half* __restrict__ v_out,
    int offset,
    int stride
)
{
    int t = threadIdx.x;
    const unsigned char* in = blockIdx.z ? v_in : k_in;
    const half* scales = blockIdx.z ? v_scales : k_scales;
    half* out = blockIdx.z ? v_out : k_out;

    int block_offset = (offset + blockIdx.y * stride + blockIdx.x * BLOCKSIZE_Q);
    __shared__ uint32_t q_buffer[BLOCKSIZE_Q / 8];
    __shared__ half s_buffer[BLOCKSIZE_Q / 32];

    // Fetch

    int4* in_q = (int4*) (in + block_offset / 2);
    int4* in_s = (int4*) (scales + block_offset / 32);
    int4* pq = (int4*) q_buffer;
    int4* ps = (int4*) s_buffer;

    if (t < BLOCKSIZE_Q / 32) pq[t] = in_q[t];
    if (t < BLOCKSIZE_Q / 256) ps[t] = in_s[t];
    __syncthreads();

    // Get scale

    half scale = s_buffer[t / 16];
    half2 scale2 = __half2half2(scale);

    // Dequantize

    int shift0 = (t % 4) * 8;
    int shift1 = shift0 + 4;
    uint32_t q = q_buffer[t / 4];
    int q0 = ((int) ((q >> shift0) & 0x0f)) - 8;
    int q1 = ((int) ((q >> shift1) & 0x0f)) - 8;

    half w0 = __int2half_rn(q0);
    half w1 = __int2half_rn(q1);
    half2 w2 = __halves2half2(w0, w1);
    w2 = __hmul2(w2, scale2);

    // Store

    half2* out2 = (half2*) (out + block_offset);
    out2[t] = w2;
}

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
)
{
    dim3 blockDim, gridDim;
    blockDim.x = THREADS_Q;
    gridDim.x = width / BLOCKSIZE_Q;
    gridDim.y = height;
    gridDim.z = v_in ? 2 : 1;

    fp16_to_q4_kv_kernel<<<gridDim, blockDim>>>(k_in, k_out, k_scales, v_in, v_out, v_scales, offset, stride);
}

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
)
{
    dim3 blockDim, gridDim;
    blockDim.x = THREADS_Q;
    gridDim.x = width / BLOCKSIZE_Q;
    gridDim.y = height;
    gridDim.z = v_in ? 2 : 1;

    q4_to_fp16_kv_kernel<<<gridDim, blockDim>>>(k_in, k_scales, k_out, v_in, v_scales, v_out, offset, stride);
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