
#define HADAMARD_Q

template <int wbits>
inline __device__ void fp16_to_q
(
    int t,
    const half* __restrict__ in,
    unsigned char* __restrict__ out,
    half* __restrict__ scales,
    int block_offset,
    const half* cal,
    int dim
)
{
    const half2* in2 = (const half2*) (in + block_offset);
    __shared__ uint32_t q_buffer[BLOCKSIZE_Q / (32 / wbits)];
    __shared__ half s_buffer[BLOCKSIZE_Q / 32];

    half2 w2 = in2[t];
    if (cal) w2 = __h2div(w2, *((half2*)(cal + (block_offset + t * 2) % dim)));

    // Perform hadamard transform on two interleaved 32-element groups. Don't scale output by 1/sqrt(32) here, instead
    // scale by 1/32 when dequantizing

    #ifdef HADAMARD_Q

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

    if constexpr (wbits == 4)
    {
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
    else
    {
        // Normalize

        half2 c_128 = __half2half2(__float2half_rn(128));
        half c_i = __float2half_rn(1.0f / 128.0f);

        w2 = __h2div(w2, absmax2);
        w2 = __hfma2(w2, c_128, c_128);

        // Quantize & pack

        int q0 = clamp(__half2int_rn(__low2half(w2)), 0, 255);
        int q1 = clamp(__half2int_rn(__high2half(w2)), 0, 255);
        uint32_t q = q0 | (q1 << 8);
        q |= (__shfl_down_sync(0x55555555, q, 1) << 16);

        if (t % 2 == 0) q_buffer[t / 2] = q;
        if (t % 16 == 0) s_buffer[t / 16] = __hmul(absmax, c_i);
        __syncthreads();

        // Store

        int4* pq = (int4*) q_buffer;
        int4* ps = (int4*) s_buffer;
        int4* out_q = (int4*) (out + block_offset);
        int4* out_s = (int4*) (scales + block_offset / 32);

        if (t < BLOCKSIZE_Q / 16) __stwb(&out_q[t], pq[t]);
        if (t < BLOCKSIZE_Q / 256) __stwb(&out_s[t], ps[t]);
    }
}

template <int wbits>
inline __device__ void q_to_fp16
(
    int t,
    const unsigned char* __restrict__ in,
    const half* __restrict__ scales,
    half* __restrict__ out,
    int block_offset,
    const half* cal,
    int dim
)
{
    const uint32_t* in_q = (const uint32_t*) (in + block_offset / (8 / wbits));
    const half* in_s = (const half*) (scales + block_offset / 32);

    // Get scale

    half scale = __ldg(in_s + t / 16);
    half2 scale2 = __half2half2(scale);

    half2 w2;

    if (wbits == 4)
    {
        // Dequantize

        int shift0 = (t % 4) * 8;
        int shift1 = shift0 + 4;
        uint32_t q = __ldg(in_q + t / 4);
        int q0 = ((int) ((q >> shift0) & 0x0f)) - 8;
        int q1 = ((int) ((q >> shift1) & 0x0f)) - 8;

        half w0 = __int2half_rn(q0);
        half w1 = __int2half_rn(q1);
        w2 = __halves2half2(w0, w1);
        w2 = __hmul2(w2, scale2);
    }
    else
    {
        // Dequantize

        int shift0 = (t % 2) * 16;
        int shift1 = shift0 + 8;
        uint32_t q = __ldg(in_q + t / 2);
        int q0 = ((int) ((q >> shift0) & 0xff)) - 128;
        int q1 = ((int) ((q >> shift1) & 0xff)) - 128;

        half w0 = __int2half_rn(q0);
        half w1 = __int2half_rn(q1);
        w2 = __halves2half2(w0, w1);

        w2 = __hmul2(w2, scale2);
    }

    // Perform hadamard transform on two interleaved 32-element groups. Skipped scaling when quantizing, so result
    // is scaled by 1/32 here

    #ifdef HADAMARD_Q

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
    if (cal) w2 = __hmul2(w2, *((half2*)(cal + (block_offset + t * 2) % dim)));
    __stcg(&out2[t], w2);
}


