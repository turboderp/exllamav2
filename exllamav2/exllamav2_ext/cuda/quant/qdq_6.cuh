#ifndef _qdq_6_cuh
#define _qdq_6_cuh

#include "qdq_util.cuh"
#include "../../config.h"

#if QMODE_6BIT == 1

// Permutation:
//
// dddd3333 33111111  cccc2222 22000000
// ffff7777 77555555  eeee6666 66444444
// ffddbbbb bb999999  eeccaaaa aa888888

__forceinline__ __device__ void shuffle_6bit_16
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0 * stride];
    uint32_t qb = q[1 * stride];
    uint32_t qc = q[2 * stride];

    // qa: 55444444 33333322  22221111 11000000
    // qb: aaaa9999 99888888  77777766 66665555
    // qc: ffffffee eeeedddd  ddcccccc bbbbbbaa

    uint32_t q00 = (qa      ) & 0b111111;
    uint32_t q01 = (qa >>  6) & 0b111111;
    uint32_t q02 = (qa >> 12) & 0b111111;
    uint32_t q03 = (qa >> 18) & 0b111111;
    uint32_t q04 = (qa >> 24) & 0b111111;
    uint32_t q05 = ((qa >> 30) & 0b11) | ((qb & 0b1111) << 2);
    uint32_t q06 = (qb >>  4) & 0b111111;
    uint32_t q07 = (qb >> 10) & 0b111111;
    uint32_t q08 = (qb >> 16) & 0b111111;
    uint32_t q09 = (qb >> 22) & 0b111111;
    uint32_t q0a = ((qb >> 28) & 0b1111) | ((qc & 0b11) << 4);
    uint32_t q0b = (qc >>  2) & 0b111111;
    uint32_t q0c = (qc >>  8) & 0b111111;
    uint32_t q0d = (qc >> 14) & 0b111111;
    uint32_t q0e = (qc >> 20) & 0b111111;
    uint32_t q0f = (qc >> 26) & 0b111111;

    qa = q00 | (q01 << 16) | (q02 << 6) | (q03 << 22);
    qb = q04 | (q05 << 16) | (q06 << 6) | (q07 << 22);
    qc = q08 | (q09 << 16) | (q0a << 6) | (q0b << 22);

    // qa: ....3333 33111111  ....2222 22000000
    // qb: ....7777 77555555  ....6666 66444444
    // qc: ....bbbb bb999999  ....aaaa aa888888

    qa |= (q0c & 0b001111) << 12;
    qc |= (q0c & 0b110000) << 8;
    qa |= (q0d & 0b001111) << 28;
    qc |= (q0d & 0b110000) << 24;

    // qa: dddd3333 33111111  cccc2222 22000000
    // qb: ....7777 77555555  ....6666 66444444
    // qc: ..ddbbbb bb999999  ..ccaaaa aa888888

    qb |= (q0e & 0b001111) << 12;
    qc |= (q0e & 0b110000) << 10;
    qb |= (q0f & 0b001111) << 28;
    qc |= (q0f & 0b110000) << 26;

    // qa: dddd3333 33111111  cccc2222 22000000
    // qb: ffff7777 77555555  eeee6666 66444444
    // qc: ffddbbbb bb999999  eeccaaaa aa888888

    q[0 * stride] = qa;
    q[1 * stride] = qb;
    q[2 * stride] = qc;
}

__forceinline__ __device__ void dequant_6bit_16
(
    const uint32_t q_0,
    const uint32_t q_1,
    const uint32_t q_2,
    half2 (&dq)[8],
    int stride
)
{
    const uint32_t c0 = 0x64006400;
    const half z1_  = __float2half_rn(-1024.0f - 32.0f);
    const half2 z1  = __halves2half2(z1_, z1_);

    uint32_t qa = q_0;
    uint32_t qb = q_1;
    uint32_t qc = q_2;

    half2_uint32 q0((qa & 0x003f003f) | c0); // half2(q[ 0], q[ 1])      + 1024
    qa >>= 6;
    half2_uint32 q1((qa & 0x003f003f) | c0); // half2(q[ 2], q[ 3])      + 1024
    qa >>= 6;
    half2_uint32 q2((qb & 0x003f003f) | c0); // half2(q[ 4], q[ 5])      + 1024
    qb >>= 6;
    half2_uint32 q3((qb & 0x003f003f) | c0); // half2(q[ 6], q[ 7])      + 1024
    qb >>= 6;
    half2_uint32 q4((qc & 0x003f003f) | c0); // half2(q[ 8], q[ 9])      + 1024
    qc >>= 6;
    half2_uint32 q5((qc & 0x003f003f) | c0); // half2(q[10], q[11])      + 1024
    qc >>= 2;
    half2_uint32 q6((qa & 0x000f000f) | (qc & 0x00300030) | c0); // half2(q[12], q[13])      + 1024
    qc >>= 2;
    half2_uint32 q7((qb & 0x000f000f) | (qc & 0x00300030) | c0); // half2(q[14], q[15])      + 1024

    dq[0] = __hadd2(q0.as_half2, z1);
    dq[1] = __hadd2(q1.as_half2, z1);
    dq[2] = __hadd2(q2.as_half2, z1);
    dq[3] = __hadd2(q3.as_half2, z1);
    dq[4] = __hadd2(q4.as_half2, z1);
    dq[5] = __hadd2(q5.as_half2, z1);
    dq[6] = __hadd2(q6.as_half2, z1);
    dq[7] = __hadd2(q7.as_half2, z1);
}

#else

__forceinline__ __device__ void shuffle_6bit_16
(
    uint32_t* q,
    int stride
)
{
}

__forceinline__ __device__ void dequant_6bit_16
(
    const uint32_t q_0,
    const uint32_t q_1,
    const uint32_t q_2,
    half2 (&dq)[8],
    int stride
)
{
    half dqh[16];
    for (int i = 0; i < 5; i++) dqh[     i] = dq_ns(exb(     q_0, i * 6    , 0x3f), 32);
                                dqh[ 5    ] = dq_ns(exb(q_1, q_0,        30, 0x3f), 32);
    for (int i = 0; i < 4; i++) dqh[ 6 + i] = dq_ns(exb(     q_1, i * 6 + 4, 0x3f), 32);
                                dqh[10    ] = dq_ns(exb(q_2, q_1,        28, 0x3f), 32);
    for (int i = 0; i < 5; i++) dqh[11 + i] = dq_ns(exb(     q_2, i * 6 + 2, 0x3f), 32);

    for (int i = 0; i < 8; i++) dq[i] = __halves2half2(dqh[i * 2], dqh[i * 2 + 1]);
}

#endif

#endif


