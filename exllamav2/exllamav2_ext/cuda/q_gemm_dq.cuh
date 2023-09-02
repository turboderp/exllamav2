
#include "quant/qdq_2.cuh"
#include "quant/qdq_3.cuh"

__device__ __forceinline__ void dq_2bit_16(const uint32_t* b_ptr, half (&dq)[16], const int size_n)
{
    uint32_t q0 = *b_ptr;
    for (int i = 0; i < 16; i++) dq[i  ] = dq_ns(exb(    q0, i * 2    , 0x03),   2);
}

__device__ __forceinline__ void dq_3bit_32(const uint32_t* b_ptr, half (&dq)[32], const int size_n)
{
    uint32_t q0 = *b_ptr; b_ptr += size_n;
    uint32_t q1 = *b_ptr; b_ptr += size_n;
    uint32_t q2 = *b_ptr;
    int j = 0;
    for (int i = 0; i < 10; i++) dq[j++] = dq_ns(exb(    q0, i * 3    , 0x07),   4);
                                 dq[j++] = dq_ns(exb(q1, q0,        30, 0x07),   4);
    for (int i = 0; i < 10; i++) dq[j++] = dq_ns(exb(    q1, i * 3 + 1, 0x07),   4);
                                 dq[j++] = dq_ns(exb(q2, q1,        31, 0x07),   4);
    for (int i = 0; i < 10; i++) dq[j++] = dq_ns(exb(    q2, i * 3 + 2, 0x07),   4);
}

__device__ __forceinline__ void dq_4bit_8(const uint32_t* b_ptr, half (&dq)[8], const int size_n)
{
    uint32_t q0 = *b_ptr;
    for (int i = 0; i <  8; i++) dq[i  ] = dq_ns(exb(    q0, i * 4    , 0x0f),   8);
}

__device__ __forceinline__ void dq_5bit_32(const uint32_t* b_ptr, half (&dq)[32], const int size_n)
{
    uint32_t q0 = *b_ptr; b_ptr += size_n;
    uint32_t q1 = *b_ptr; b_ptr += size_n;
    uint32_t q2 = *b_ptr; b_ptr += size_n;
    uint32_t q3 = *b_ptr; b_ptr += size_n;
    uint32_t q4 = *b_ptr;
    int j = 0;
    for (int i = 0; i <  6; i++) dq[j++] = dq_ns(exb(    q0, i * 5    , 0x1f),  16);
                                 dq[j++] = dq_ns(exb(q1, q0,        30, 0x1f),  16);
    for (int i = 0; i <  5; i++) dq[j++] = dq_ns(exb(    q1, i * 5 + 3, 0x1f),  16);
                                 dq[j++] = dq_ns(exb(q2, q1,        28, 0x1f),  16);
    for (int i = 0; i <  6; i++) dq[j++] = dq_ns(exb(    q2, i * 5 + 1, 0x1f),  16);
                                 dq[j++] = dq_ns(exb(q3, q2,        31, 0x1f),  16);
    for (int i = 0; i <  5; i++) dq[j++] = dq_ns(exb(    q3, i * 5 + 4, 0x1f),  16);
                                 dq[j++] = dq_ns(exb(q4, q3,        29, 0x1f),  16);
    for (int i = 0; i <  6; i++) dq[j++] = dq_ns(exb(    q4, i * 5 + 2, 0x1f),  16);
}

__device__ __forceinline__ void dq_6bit_16(const uint32_t* b_ptr, half (&dq)[16], const int size_n)
{
    uint32_t q0 = *b_ptr; b_ptr += size_n;
    uint32_t q1 = *b_ptr; b_ptr += size_n;
    uint32_t q2 = *b_ptr;
    int j = 0;
    for (int i = 0; i <  5; i++) dq[j++] = dq_ns(exb(    q0, i * 6    , 0x3f),  32);
                                 dq[j++] = dq_ns(exb(q1, q0,        30, 0x3f),  32);
    for (int i = 0; i <  4; i++) dq[j++] = dq_ns(exb(    q1, i * 6 + 4, 0x3f),  32);
                                 dq[j++] = dq_ns(exb(q2, q1,        28, 0x3f),  32);
    for (int i = 0; i <  5; i++) dq[j++] = dq_ns(exb(    q2, i * 6 + 2, 0x3f),  32);
}

__device__ __forceinline__ void dq_8bit_4(const uint32_t* b_ptr, half (&dq)[4], const int size_n)
{
    uint32_t q0 = *b_ptr;
    for (int i = 0; i <  4; i++) dq[i  ] = dq_ns(exb(    q0, i * 8    , 0xff), 128);
}

__forceinline__ __device__ half dot_4(half (&dq)[4], const half* a_ptr, const half g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*) a_ptr;
    for (int i = 0; i < 4; i += 2) result = __hfma2(__halves2half2(dq[i], dq[i + 1]), *a2_ptr++, result);
    return __hfma(__hadd(result.x, result.y), qs_h, g_result);
}

__forceinline__ __device__ half dot_8(half (&dq)[8], const half* a_ptr, const half g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*) a_ptr;
    for (int i = 0; i < 8; i += 2) result = __hfma2(__halves2half2(dq[i], dq[i + 1]), *a2_ptr++, result);
    return __hfma(__hadd(result.x, result.y), qs_h, g_result);
}

__forceinline__ __device__ half dot_16(half (&dq)[16], const half* a_ptr, const half g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*) a_ptr;
    for (int i = 0; i < 16; i += 2) result = __hfma2(__halves2half2(dq[i], dq[i + 1]), *a2_ptr++, result);
    return __hfma(__hadd(result.x, result.y), qs_h, g_result);
}

__forceinline__ __device__ half dot_32(half (&dq)[32], const half* a_ptr, const half g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*) a_ptr;
    for (int i = 0; i < 32; i += 2) result = __hfma2(__halves2half2(dq[i], dq[i + 1]), *a2_ptr++, result);
    return __hfma(__hadd(result.x, result.y), qs_h, g_result);
}

__forceinline__ __device__ half2 dot2_32(half(&dq)[32], const half* a_ptr, const half2 g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    for (int i = 0; i < 32; i += 2) result = __hfma2(__halves2half2(dq[i], dq[i + 1]), *a2_ptr++, result);
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot2_16(half(&dq)[16], const half* a_ptr, const half2 g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    for (int i = 0; i < 16; i += 2) result = __hfma2(__halves2half2(dq[i], dq[i + 1]), *a2_ptr++, result);
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}


__forceinline__ __device__ half2 dot22_8(half2(&dq)[4], const half* a_ptr, const half2 g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}


__forceinline__ __device__ half2 dot22_16(half2(&dq)[8], const half* a_ptr, const half2 g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}


__forceinline__ __device__ half2 dot22_32(half2(&dq)[16], const half* a_ptr, const half2 g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}



__forceinline__ __device__ void advance_group
(
    int &k,
    int &group,
    int &nextgroup,
    const int &groupsize,
    int &n,
//     MatrixView_q4_row &b_q_scale_,
//     const half* &b_q_scale_max,
    half* scales,
    int &scales_idx,
    half &qs_h
)
{
    if (k == nextgroup)
    {
        group++;
        scales_idx++;
        qs_h = scales[scales_idx];
        //qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]);
        nextgroup += groupsize;
    }
}

__forceinline__ __device__ void load_2
(
    const uint32_t* &b_ptr,
    const int &size_n,
    uint32_t (&q)[2])
{
    #pragma unroll
    for (int i = 0; i < 2; i++, b_ptr += size_n) q[i] = *b_ptr;
}

__forceinline__ __device__ void load_3
(
    const uint32_t* &b_ptr,
    const int &size_n,
    uint32_t (&q)[3])
{
    #pragma unroll
    for (int i = 0; i < 3; i++, b_ptr += size_n) q[i] = *b_ptr;
}

__forceinline__ __device__ void load_4
(
    const uint32_t* &b_ptr,
    const int &size_n,
    uint32_t (&q)[4])
{
    #pragma unroll
    for (int i = 0; i < 4; i++, b_ptr += size_n) q[i] = *b_ptr;
}

__forceinline__ __device__ void load_5
(
    const uint32_t* &b_ptr,
    const int &size_n,
    uint32_t (&q)[5])
{
    #pragma unroll
    for (int i = 0; i < 5; i++, b_ptr += size_n) q[i] = *b_ptr;
}

__forceinline__ __device__ void load_6
(
    const uint32_t* &b_ptr,
    const int &size_n,
    uint32_t (&q)[6])
{
    #pragma unroll
    for (int i = 0; i < 6; i++, b_ptr += size_n) q[i] = *b_ptr;
}

__forceinline__ __device__ void load_8
(
    const uint32_t* &b_ptr,
    const int &size_n,
    uint32_t (&q)[8])
{
    #pragma unroll
    for (int i = 0; i < 8; i++, b_ptr += size_n) q[i] = *b_ptr;
}


// template <int m_count>
// __forceinline__ __device__ void qdot_2bit_32_lut
// (
//     int& k,
//     int& end_k_sg,
//     int& group,
//     int& nextgroup,
//     const int& groupsize,
//     int& n,
//     half* scales,
//     int& scales_idx,
//     half& qs_h,
//     half2* block_c,
//     const half*& a_ptr,
//     int& a_stride,
//     const uint32_t*& b_ptr,
//     const int& size_n,
//     uint32_t(&q_xx)[2],
//     uint32_t(&q_yy)[2],
//     const half2* lut2
// )
// {
//     if (k >= end_k_sg) return;
//
//     advance_group(k, group, nextgroup, groupsize, n, scales, scales_idx, qs_h);
//
//     if (k + 32 < end_k_sg) load_2(b_ptr, size_n, q_yy);
//
//     int i = 0;
//     half2 dq[16];
//
//     int l0 = (q_xx[0]      ) & 0xff;
//     int l1 = (q_xx[0] >>  8) & 0xff;
//     int l2 = (q_xx[0] >> 16) & 0xff;
//     int l3 = (q_xx[0] >> 24);
//     int l4 = (q_xx[1]      ) & 0xff;
//     int l5 = (q_xx[1] >>  8) & 0xff;
//     int l6 = (q_xx[1] >> 16) & 0xff;
//     int l7 = (q_xx[1] >> 24);
//
//     dq[ 0] = lut2[l0 * 2    ];
//     dq[ 1] = lut2[l0 * 2 + 1];
//     dq[ 2] = lut2[l1 * 2    ];
//     dq[ 3] = lut2[l1 * 2 + 1];
//     dq[ 4] = lut2[l2 * 2    ];
//     dq[ 5] = lut2[l2 * 2 + 1];
//     dq[ 6] = lut2[l3 * 2    ];
//     dq[ 7] = lut2[l3 * 2 + 1];
//     dq[ 8] = lut2[l4 * 2    ];
//     dq[ 9] = lut2[l4 * 2 + 1];
//     dq[10] = lut2[l5 * 2    ];
//     dq[11] = lut2[l5 * 2 + 1];
//     dq[12] = lut2[l6 * 2    ];
//     dq[13] = lut2[l6 * 2 + 1];
//     dq[14] = lut2[l7 * 2    ];
//     dq[15] = lut2[l7 * 2 + 1];
//
//     //for (int i = 0; i < 16; i++) dq[     i] = dq_ns(exb(q_xx[0], i * 2, 0x03), 2);
//     //for (int i = 0; i < 16; i++) dq[16 + i] = dq_ns(exb(q_xx[1], i * 2, 0x03), 2);
//
//     for (int m = 0; m < m_count; m++) block_c[m] = dot22_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);
//
//     a_ptr += 32;
//     k += 32;
// }

// template <int m_count>
// __forceinline__ __device__ void qdot_2bit_32_bf
// (
//     int &k,
//     int &end_k_sg,
//     int &group,
//     int &nextgroup,
//     const int &groupsize,
//     int &n,
//     half* scales,
//     int &scales_idx,
//     half &qs_h,
//     half2* block_c,
//     const half* &a_ptr,
//     int &a_stride,
//     const uint32_t* &b_ptr,
//     const int &size_n,
//     uint32_t (&q_xx)[2],
//     uint32_t (&q_yy)[2]
// )
// {
//     if (k >= end_k_sg) return;
//
//     advance_group(k, group, nextgroup, groupsize, n, scales, scales_idx, qs_h);
//
//     if (k + 32 < end_k_sg) load_2(b_ptr, size_n, q_yy);
//
//     for (int j = 0; j < 2; j++)
//     {
//         uint32_t dqp[1];
//         dqp[0] = q_xx[j];
//         half2 dq[8];
//
//         dequant_2bit_16(dqp, dq);
//
//         for (int m = 0; m < m_count; m++) block_c[m] = dot22_16(dq, a_ptr + m * a_stride, block_c[m], qs_h);
//
//         a_ptr += 16;
//         k += 16;
//     }
// }


// template <int m_count>
// __forceinline__ __device__ void qdot_2bit_32_bf2
// (
//     int &k,
//     int &end_k_sg,
//     int &group,
//     int &nextgroup,
//     const int &groupsize,
//     int &n,
//     half* scales,
//     int &scales_idx,
//     half &qs_h,
//     half2* block_c,
//     const half* &a_ptr,
//     int &a_stride,
//     const uint32_t* &b_ptr,
//     const int &size_n,
//     uint32_t (&q_xx)[2],
//     uint32_t (&q_yy)[2]
// )
// {
//     if (k >= end_k_sg) return;
//
//     advance_group(k, group, nextgroup, groupsize, n, scales, scales_idx, qs_h);
//
//     if (k + 32 < end_k_sg) load_2(b_ptr, size_n, q_yy);
//
//     const uint64_t lut = 0x3c000000bc00c000;  // half4(1, 0, -1, -2)
//
//     int i = 0;
//     half dq[32];
//
//     uint32_t q = q_xx[0];
//     #pragma unroll
//     while (i < 16)
//     {
//         uint32_t b = q & 3;
//         uint64_t d = lut >> (b << 4);
//         dq[i++] = __ushort_as_half((uint16_t)d);
//         q >>= 2;
//     }
//
//     q = q_xx[1];
//     #pragma unroll
//     while (i < 32)
//     {
//         uint32_t b = q & 3;
//         uint64_t d = lut >> (b << 4);
//         dq[i++] = __ushort_as_half((uint16_t)d);
//         q >>= 2;
//     }
//
//     //for (int i = 0; i < 16; i++) dq[     i] = dq_ns(exb(q_xx[0], i * 2, 0x03), 2);
//     //for (int i = 0; i < 16; i++) dq[16 + i] = dq_ns(exb(q_xx[1], i * 2, 0x03), 2);
//
//     for (int m = 0; m < m_count; m++) block_c[m] = dot2_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);
//
//     a_ptr += 32;
//     k += 32;
// }
//
// template <int m_count>
// __forceinline__ __device__ void qdot_2bit_32
// (
//     int &k,
//     int &end_k_sg,
//     int &group,
//     int &nextgroup,
//     const int &groupsize,
//     int &n,
//     half* scales,
//     int &scales_idx,
//     half &qs_h,
//     half2* block_c,
//     const half* &a_ptr,
//     int &a_stride,
//     const uint32_t* &b_ptr,
//     const int &size_n,
//     uint32_t (&q_xx)[2],
//     uint32_t (&q_yy)[2]
// )
// {
//     if (k >= end_k_sg) return;
//
//     advance_group(k, group, nextgroup, groupsize, n, scales, scales_idx, qs_h);
//
//     if (k + 32 < end_k_sg) load_2(b_ptr, size_n, q_yy);
//
//     half dq[32];
//     for (int i = 0; i < 16; i++) dq[     i] = dq_ns(exb(q_xx[0], i * 2, 0x03), 2);
//     for (int i = 0; i < 16; i++) dq[16 + i] = dq_ns(exb(q_xx[1], i * 2, 0x03), 2);
//
//     for (int m = 0; m < m_count; m++) block_c[m] = dot2_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);
//
//     a_ptr += 32;
//     k += 32;
// }

// template <int m_count>
// __forceinline__ __device__ void qdot_3bit_32
// (
//     int &k,
//     int &end_k_sg,
//     int &group,
//     int &nextgroup,
//     const int &groupsize,
//     int &n,
//     half* scales,
//     int &scales_idx,
//     half &qs_h,
//     half2* block_c,
//     const half* &a_ptr,
//     int &a_stride,
//     const uint32_t* &b_ptr,
//     const int &size_n,
//     uint32_t (&q_xx)[3],
//     uint32_t (&q_yy)[3]
// )
// {
//     if (k >= end_k_sg) return;
//
//     advance_group(k, group, nextgroup, groupsize, n, scales, scales_idx, qs_h);
//
//     if (k + 32 < end_k_sg) load_3(b_ptr, size_n, q_yy);
//
//     half dq[32];
//     for (int i = 0; i < 10; i++) dq[     i] = dq_ns(exb(         q_xx[0], i * 3    , 0x07), 4);
//                                  dq[10    ] = dq_ns(exb(q_xx[1], q_xx[0],        30, 0x07), 4);
//     for (int i = 0; i < 10; i++) dq[11 + i] = dq_ns(exb(         q_xx[1], i * 3 + 1, 0x07), 4);
//                                  dq[21    ] = dq_ns(exb(q_xx[2], q_xx[1],        31, 0x07), 4);
//     for (int i = 0; i < 10; i++) dq[22 + i] = dq_ns(exb(         q_xx[2], i * 3 + 2, 0x07), 4);
//
//     for (int m = 0; m < m_count; m++) block_c[m] = dot2_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);
//
//     a_ptr += 32;
//     k += 32;
// }
//

// template <int m_count>
// __forceinline__ __device__ void qdot_3bit_32_bf
// (
//     int &k,
//     int &end_k_sg,
//     int &group,
//     int &nextgroup,
//     const int &groupsize,
//     int &n,
//     half* scales,
//     int &scales_idx,
//     half &qs_h,
//     half2* block_c,
//     const half* &a_ptr,
//     int &a_stride,
//     const uint32_t* &b_ptr,
//     const int &size_n,
//     uint32_t (&q_xx)[3],
//     uint32_t (&q_yy)[3]
// )
// {
//     if (k >= end_k_sg) return;
//
//     advance_group(k, group, nextgroup, groupsize, n, scales, scales_idx, qs_h);
//     if (k + 32 < end_k_sg) load_3(b_ptr, size_n, q_yy);
//
//     uint32_t dqp[3];
//     dqp[0] = q_xx[0];
//     dqp[1] = q_xx[1];
//     dqp[2] = q_xx[2];
//     half2 dq[16];
//
//     dequant_3bit_32(dqp, dq);
//     for (int m = 0; m < m_count; m++) block_c[m] = dot22_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);
//
//     a_ptr += 32;
//     k += 32;
// }

template <int m_count>
__forceinline__ __device__ void qdot_4bit_32
(
    int &k,
    int &end_k_sg,
    int &group,
    int &nextgroup,
    const int &groupsize,
    int &n,
    half* scales,
    int &scales_idx,
    half &qs_h,
    half2* block_c,
    const half* &a_ptr,
    int &a_stride,
    const uint32_t* &b_ptr,
    const int &size_n,
    uint32_t (&q_xx)[4],
    uint32_t (&q_yy)[4]
)
{
    if (k >= end_k_sg) return;

    advance_group(k, group, nextgroup, groupsize, n, scales, scales_idx, qs_h);

    if (k + 32 < end_k_sg) load_4(b_ptr, size_n, q_yy);

    half dq[32];
    for (int i = 0; i <  8; i++) dq[     i] = dq_ns(exb(q_xx[0], i * 4, 0x0f), 8);
    for (int i = 0; i <  8; i++) dq[ 8 + i] = dq_ns(exb(q_xx[1], i * 4, 0x0f), 8);
    for (int i = 0; i <  8; i++) dq[16 + i] = dq_ns(exb(q_xx[2], i * 4, 0x0f), 8);
    for (int i = 0; i <  8; i++) dq[24 + i] = dq_ns(exb(q_xx[3], i * 4, 0x0f), 8);

    for (int m = 0; m < m_count; m++) block_c[m] = dot2_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);

    a_ptr += 32;
    k += 32;
}

template <int m_count>
__forceinline__ __device__ void qdot_4bit_64
(
    int &k,
    int &group,
    int &nextgroup,
    const int &groupsize,
    int &n,
    half* scales,
    int &scales_idx,
    half &qs_h,
    half2* block_c,
    const half* &a_ptr,
    int &a_stride,
    const uint32_t* &b_ptr,
    const int &size_n,
    uint32_t (&q_xx)[8]
)
{
    for (int j = 0; j < 2; j++)
    {
        advance_group(k, group, nextgroup, groupsize, n, scales, scales_idx, qs_h);
        half dq[32];
        for (int i = 0; i <  8; i++) dq[     i] = dq_ns(exb(q_xx[4 * j    ], i * 4, 0x0f), 8);
        for (int i = 0; i <  8; i++) dq[ 8 + i] = dq_ns(exb(q_xx[4 * j + 1], i * 4, 0x0f), 8);
        for (int i = 0; i <  8; i++) dq[16 + i] = dq_ns(exb(q_xx[4 * j + 2], i * 4, 0x0f), 8);
        for (int i = 0; i <  8; i++) dq[24 + i] = dq_ns(exb(q_xx[4 * j + 3], i * 4, 0x0f), 8);
        for (int m = 0; m < m_count; m++) block_c[m] = dot2_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);
        a_ptr += 32;
        k += 32;
    }
}

template <int m_count>
__forceinline__ __device__ void qdot_5bit_32
(
    int &k,
    int &end_k_sg,
    int &group,
    int &nextgroup,
    const int &groupsize,
    int &n,
    half* scales,
    int &scales_idx,
    half &qs_h,
    half2* block_c,
    const half* &a_ptr,
    int &a_stride,
    const uint32_t* &b_ptr,
    const int &size_n,
    uint32_t (&q_xx)[5],
    uint32_t (&q_yy)[5]
)
{
    if (k >= end_k_sg) return;

    advance_group(k, group, nextgroup, groupsize, n, scales, scales_idx, qs_h);

    if (k + 32 < end_k_sg) load_5(b_ptr, size_n, q_yy);

    half dq[32];
    for (int i = 0; i <  6; i++) dq[     i] = dq_ns(exb(         q_xx[0], i * 5    , 0x1f), 16);
                                 dq[ 6    ] = dq_ns(exb(q_xx[1], q_xx[0],        30, 0x1f), 16);
    for (int i = 0; i <  5; i++) dq[ 7 + i] = dq_ns(exb(         q_xx[1], i * 5 + 3, 0x1f), 16);
                                 dq[12    ] = dq_ns(exb(q_xx[2], q_xx[1],        28, 0x1f), 16);
    for (int i = 0; i <  6; i++) dq[13 + i] = dq_ns(exb(         q_xx[2], i * 5 + 1, 0x1f), 16);
                                 dq[19    ] = dq_ns(exb(q_xx[3], q_xx[2],        31, 0x1f), 16);
    for (int i = 0; i <  5; i++) dq[20 + i] = dq_ns(exb(         q_xx[3], i * 5 + 4, 0x1f), 16);
                                 dq[25    ] = dq_ns(exb(q_xx[4], q_xx[3],        29, 0x1f), 16);
    for (int i = 0; i <  6; i++) dq[26 + i] = dq_ns(exb(         q_xx[4], i * 5 + 2, 0x1f), 16);

    for (int m = 0; m < m_count; m++) block_c[m] = dot2_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);

    a_ptr += 32;
    k += 32;
}

template <int m_count>
__forceinline__ __device__ void qdot_6bit_32
(
    int &k,
    int &end_k_sg,
    int &group,
    int &nextgroup,
    const int &groupsize,
    int &n,
    half* scales,
    int &scales_idx,
    half &qs_h,
    half2* block_c,
    const half* &a_ptr,
    int &a_stride,
    const uint32_t* &b_ptr,
    const int &size_n,
    uint32_t (&q_xx)[6],
    uint32_t (&q_yy)[6]
)
{
    if (k >= end_k_sg) return;

    advance_group(k, group, nextgroup, groupsize, n, scales, scales_idx, qs_h);

    if (k + 32 < end_k_sg) load_6(b_ptr, size_n, q_yy);

    half dq[32];

    for (int i = 0; i <  5; i++) dq[     i] = dq_ns(exb(         q_xx[0], i * 6    , 0x3f), 32);
                                 dq[ 5    ] = dq_ns(exb(q_xx[1], q_xx[0],        30, 0x3f), 32);
    for (int i = 0; i <  4; i++) dq[ 6 + i] = dq_ns(exb(         q_xx[1], i * 6 + 4, 0x3f), 32);
                                 dq[10    ] = dq_ns(exb(q_xx[2], q_xx[1],        28, 0x3f), 32);
    for (int i = 0; i <  5; i++) dq[11 + i] = dq_ns(exb(         q_xx[2], i * 6 + 2, 0x3f), 32);
    for (int i = 0; i <  5; i++) dq[16 + i] = dq_ns(exb(         q_xx[3], i * 6    , 0x3f), 32);
                                 dq[21    ] = dq_ns(exb(q_xx[4], q_xx[3],        30, 0x3f), 32);
    for (int i = 0; i <  4; i++) dq[22 + i] = dq_ns(exb(         q_xx[4], i * 6 + 4, 0x3f), 32);
                                 dq[26    ] = dq_ns(exb(q_xx[5], q_xx[4],        28, 0x3f), 32);
    for (int i = 0; i <  5; i++) dq[27 + i] = dq_ns(exb(         q_xx[5], i * 6 + 2, 0x3f), 32);

    for (int m = 0; m < m_count; m++) block_c[m] = dot2_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);

    a_ptr += 32;
    k += 32;
}

template <int m_count>
__forceinline__ __device__ void qdot_8bit_32
(
    int &k,
    int &end_k_sg,
    int &group,
    int &nextgroup,
    const int &groupsize,
    int &n,
    half* scales,
    int &scales_idx,
    half &qs_h,
    half2* block_c,
    const half* &a_ptr,
    int &a_stride,
    const uint32_t* &b_ptr,
    const int &size_n,
    uint32_t (&q_xx)[8],
    uint32_t (&q_yy)[8]
)
{
    if (k >= end_k_sg) return;

    advance_group(k, group, nextgroup, groupsize, n, scales, scales_idx, qs_h);

    if (k + 32 < end_k_sg) load_8(b_ptr, size_n, q_yy);

    half dq[32];

    for (int i = 0; i <  4; i++) dq[     i] = dq_ns(exb(q_xx[0], i * 8, 0xff), 128);
    for (int i = 0; i <  4; i++) dq[ 4 + i] = dq_ns(exb(q_xx[1], i * 8, 0xff), 128);
    for (int i = 0; i <  4; i++) dq[ 8 + i] = dq_ns(exb(q_xx[2], i * 8, 0xff), 128);
    for (int i = 0; i <  4; i++) dq[12 + i] = dq_ns(exb(q_xx[3], i * 8, 0xff), 128);
    for (int i = 0; i <  4; i++) dq[16 + i] = dq_ns(exb(q_xx[4], i * 8, 0xff), 128);
    for (int i = 0; i <  4; i++) dq[20 + i] = dq_ns(exb(q_xx[5], i * 8, 0xff), 128);
    for (int i = 0; i <  4; i++) dq[24 + i] = dq_ns(exb(q_xx[6], i * 8, 0xff), 128);
    for (int i = 0; i <  4; i++) dq[28 + i] = dq_ns(exb(q_xx[7], i * 8, 0xff), 128);

    for (int m = 0; m < m_count; m++) block_c[m] = dot2_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);

    a_ptr += 32;
    k += 32;
}
