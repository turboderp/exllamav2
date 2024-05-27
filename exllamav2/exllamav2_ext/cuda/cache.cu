#include "cache.cuh"

#include "quant/qdq_util.cuh"
#include "util.cuh"
#include "compat.cuh"

#define THREADS 32
#define BLOCKSIZE_Q 512
#define SUPER_BLOCKSIZE_Q (128 * 1024)
#define THREADS_Q (BLOCKSIZE_Q / 2)
#define HADAMARD_Q4

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

// -------------- FP16 -> Q4

inline __device__ void fp16_to_q4
(
    int t,
    const half* __restrict__ in,
    unsigned char* __restrict__ out,
    half* __restrict__ scales,
    int block_offset
)
{
    const half2* in2 = (const half2*) (in + block_offset);
    __shared__ uint32_t q_buffer[BLOCKSIZE_Q / 8];
    __shared__ half s_buffer[BLOCKSIZE_Q / 32];

    half2 w2 = in2[t];
    half2 o = w2;

    // Perform hadamard transform on two interleaved 32-element groups. Don't scale output by 1/sqrt(32) here, instead
    // scale by 1/32 when dequantizing

    #ifdef HADAMARD_Q4

        for (int i = 1; i < 32; i <<= 1)
        {
            half2 pw2 = __shfl_xor_sync(0xffffffff, w2, i);
            uint32_t* w2i = reinterpret_cast<uint32_t*>(&w2);
            int32_t sfm = -static_cast<int32_t>(t & i) >> 31;
            *w2i ^= (sfm & 0x80008000);
            w2 = __hadd2(w2, pw2);
        }

    #endif

    // Max abs value for lane_id 0..15, 16..31

    half2 absmax2 = __habs2(w2);
    half absmax = __hmax(__low2half(absmax2), __high2half(absmax2));
    absmax = __hmax(absmax, __shfl_xor_sync(0xffffffff, absmax, 8));
    absmax = __hmax(absmax, __shfl_xor_sync(0xffffffff, absmax, 4));
    absmax = __hmax(absmax, __shfl_xor_sync(0xffffffff, absmax, 2));
    absmax = __hmax(absmax, __shfl_xor_sync(0xffffffff, absmax, 1));
    absmax2 = __half2half2(absmax);

    // Normalize

    half2 c_8 = __half2half2(__float2half_rn(8));
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

__global__ void fp16_to_q4_kv_paged_kernel
(
    const half* __restrict__ k_in,
    unsigned char* __restrict__ k_out,
    half* __restrict__ k_scales,
    const half* __restrict__ v_in,
    unsigned char* __restrict__ v_out,
    half* __restrict__ v_scales,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ block_table,
    int pages_per_seq,
    int page_size,
    int dim,
    int q_len
)
{
    int t = threadIdx.x;
    int kv = blockIdx.z & 1;
    const half* in = kv ? v_in : k_in;
    half* scales = kv ? v_scales : k_scales;
    unsigned char* out = kv ? v_out : k_out;

    int x = blockIdx.x;
    int y = blockIdx.z >> 1;

    int page = block_table[pages_per_seq * y + x];
    int seqlen = cache_seqlens[y];
    int vx_a = page_size * x;
    int px_a = seqlen - vx_a;
    int px_b = px_a + q_len;
    px_a = max(px_a, 0);
    px_b = min(px_b, page_size);

    int block_a = (page * page_size + px_a) * dim;
    int block_b = (page * page_size + px_b) * dim;

    for (int i = block_a; i < block_b; i += SUPER_BLOCKSIZE_Q)
    {
        int j = i + blockIdx.y * BLOCKSIZE_Q;
        if (j >= block_b) continue;
        fp16_to_q4(t, in, out, scales, j);
    }
}

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

    fp16_to_q4(t, in, out, scales, block_offset);
}

void array_fp16_to_q4_kv_paged_cuda
(
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
    int q_len
)
{
    dim3 blockDim, gridDim;
    blockDim.x = THREADS_Q;
    gridDim.x = pages_per_seq;
    gridDim.y = SUPER_BLOCKSIZE_Q / BLOCKSIZE_Q;
    gridDim.z = batch_size * 2;

    fp16_to_q4_kv_paged_kernel<<<gridDim, blockDim>>>
    (
        k_in,
        k_out,
        k_scales,
        v_in,
        v_out,
        v_scales,
        cache_seqlens,
        block_table,
        pages_per_seq,
        page_size,
        dim,
        q_len
    );
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

// --------------- Q4 -> FP16

inline __device__ void q4_to_fp16
(
    int t,
    const unsigned char* __restrict__ in,
    const half* __restrict__ scales,
    half* __restrict__ out,
    int block_offset
)
{
//    __shared__ uint32_t q_buffer[BLOCKSIZE_Q / 8];
//    __shared__ half s_buffer[BLOCKSIZE_Q / 32];

    // Fetch

//    int4* in_q = (int4*) (in + block_offset / 2);
//    int4* in_s = (int4*) (scales + block_offset / 32);
//    int4* pq = (int4*) q_buffer;
//    int4* ps = (int4*) s_buffer;
//
//    if (t < BLOCKSIZE_Q / 32) pq[t] = in_q[t];
//    int t2 = t - BLOCKSIZE_Q / 32;
//    if (t2 >= 0 && t2 < BLOCKSIZE_Q / 256) ps[t2] = in_s[t2];
//    __syncthreads();

    const uint32_t* in_q = (const uint32_t*) (in + block_offset / 2);
    const half* in_s = (const half*) (scales + block_offset / 32);

    // Get scale

//    half scale = s_buffer[t / 16];
    half scale = __ldg(in_s + t / 16);
    half2 scale2 = __half2half2(scale);

    // Dequantize

    int shift0 = (t % 4) * 8;
    int shift1 = shift0 + 4;
//    uint32_t q = q_buffer[t / 4];
    uint32_t q = __ldg(in_q + t / 4);
    int q0 = ((int) ((q >> shift0) & 0x0f)) - 8;
    int q1 = ((int) ((q >> shift1) & 0x0f)) - 8;

    half w0 = __int2half_rn(q0);
    half w1 = __int2half_rn(q1);
    half2 w2 = __halves2half2(w0, w1);
    w2 = __hmul2(w2, scale2);

    // Perform hadamard transform on two interleaved 32-element groups. Skipped scaling when quantizing, so result
    // is scaled by 1/32 here

    #ifdef HADAMARD_Q4

        for (int i = 1; i < 32; i <<= 1)
        {
            half2 pw2 = __shfl_xor_sync(0xffffffff, w2, i);
            uint32_t* w2i = reinterpret_cast<uint32_t*>(&w2);
            int32_t sfm = -static_cast<int32_t>(t & i) >> 31;
            *w2i ^= (sfm & 0x80008000);
            w2 = __hadd2(w2, pw2);
        }
        w2 = __hmul2(w2, __float2half2_rn(1.0f/32.0f));

    #endif

    // Store

    half2* out2 = (half2*) (out + block_offset);
    out2[t] = w2;
}

__global__ void q4_to_fp16_kv_paged_kernel
(
    const unsigned char* __restrict__ k_in,
    const half* __restrict__ k_scales,
    half* __restrict__ k_out,
    const unsigned char* __restrict__ v_in,
    const half* __restrict__ v_scales,
    half* __restrict__ v_out,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ block_table,
    int pages_per_seq,
    int page_size,
    int dim
)
{
    int t = threadIdx.x;
    int kv = blockIdx.z & 1;
    const unsigned char* in = kv ? v_in : k_in;
    const half* scales = kv ? v_scales : k_scales;
    half* out = kv ? v_out : k_out;

    int x = blockIdx.x;
    int y = blockIdx.z >> 1;
    int page = block_table[pages_per_seq * y + x];
    int seqlen = cache_seqlens[y];
    int vx_a = page_size * x;
    int vx_b = min(vx_a + page_size, seqlen);
    int vnum = max(vx_b - vx_a, 0);
    int block_a = (page * page_size) * dim;
    int block_b = (page * page_size + vnum) * dim;

    for (int i = block_a; i < block_b; i += SUPER_BLOCKSIZE_Q)
    {
        int j = i + blockIdx.y * BLOCKSIZE_Q;
        if (j >= block_b) continue;
        q4_to_fp16(t, in, scales, out, j);
    }
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

    q4_to_fp16(t, in, scales, out, block_offset);
}

void array_q4_to_fp16_kv_paged_cuda
(
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
    int page_size
)
{
    dim3 blockDim, gridDim;
    blockDim.x = THREADS_Q;
    gridDim.x = pages_per_seq;
    gridDim.y = SUPER_BLOCKSIZE_Q / BLOCKSIZE_Q;
    gridDim.z = batch_size * 2;

    q4_to_fp16_kv_paged_kernel<<<gridDim, blockDim>>>
    (
        k_in,
        k_scales,
        k_out,
        v_in,
        v_scales,
        v_out,
        cache_seqlens,
        block_table,
        pages_per_seq,
        page_size,
        dim
    );
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
