#ifndef _qdq_4_cuh
#define _qdq_4_cuh

#include "qdq_util.cuh"
#include "../../config.h"

#if QMODE_4BIT == 1

// Permutation:
//
// 77775555 33331111  66664444 22220000

__forceinline__ __device__ void shuffle_4bit_8
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0];
    uint32_t qb = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        uint32_t qa0 = qa & 0x0f;
        uint32_t qa1 = (qa & 0xf0) >> 4;
        qa >>= 8;
        qb |= (qa1 << (i * 4 + 16));
        qb |= (qa0 << (i * 4));
    }
    q[0] = qb;
}

__forceinline__ __device__ void dequant_4bit_8
(
    const uint32_t* q,
    half2 (&dq)[4],
    int stride
)
{
//     uint32_t qt[1];
//     qt[0] = q[0];
//
//     shuffle_4bit_8(qt, 1);

    const uint32_t c0 = 0x64006400;
    const half y16_ = __float2half_rn(1.0f / 16.0f);
    const half2 y16 = __halves2half2(y16_, y16_);
    const half z1_  = __float2half_rn(-1024.0f         - 8.0f);
    const half z16_ = __float2half_rn(-1024.0f / 16.0f - 8.0f);
    const half2 z1  = __halves2half2(z1_,  z1_);
    const half2 z16 = __halves2half2(z16_, z16_);

    uint32_t qa = q[0];
    half2_uint32 q0((qa & 0x000f000f) | c0); // half2(q[ 0], q[ 1])      + 1024
    half2_uint32 q1((qa & 0x00f000f0) | c0); // half2(q[ 2], q[ 3]) * 16 + 1024
    qa >>= 8;
    half2_uint32 q2((qa & 0x000f000f) | c0); // half2(q[ 4], q[ 5])      + 1024
    half2_uint32 q3((qa & 0x00f000f0) | c0); // half2(q[ 6], q[ 7]) * 16 + 1024

    dq[0] = __hadd2(q0.as_half2, z1);
    dq[1] = __hfma2(q1.as_half2, y16, z16);
    dq[2] = __hadd2(q2.as_half2, z1);
    dq[3] = __hfma2(q3.as_half2, y16, z16);

//     half dqp[8];
//     for (int i = 0; i <  8; i++) dqp[i] = dq_ns(exb(q[0], i * 4, 0x0f), 8);
//
//     half2* dqp2 = (half2*) dqp;
//
//     if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
//     {
//         DBGX(q[0]);
//         DBGX(qt[0]);
//         DBGH2(dq[0].x, dqp2[0].x);
//         DBGH2(dq[1].x, dqp2[1].x);
//         DBGH2(dq[2].x, dqp2[2].x);
//         DBGH2(dq[3].x, dqp2[3].x);
//         DBGH2(dq[0].y, dqp2[0].y);
//         DBGH2(dq[1].y, dqp2[1].y);
//         DBGH2(dq[2].y, dqp2[2].y);
//         DBGH2(dq[3].y, dqp2[3].y);
//         printf("-----\n");
//     }
//
// //     dq[0] = dqp2[0];
// //     dq[1] = dqp2[1];
// //     dq[2] = dqp2[2];
// //     dq[3] = dqp2[3];
}

#else

__forceinline__ __device__ void shuffle_4bit_8
(
    uint32_t* q,
    int stride
)
{
}

__forceinline__ __device__ void dequant_4bit_8
(
    const uint32_t* q,
    half2 (&dq)[4],
    int stride
)
{
    half dqh[8];
    for (int i = 0; i < 8; i++) dqh[i] = dq_ns(exb(q[0 * stride], i * 4, 0x0f), 8);

    for (int i = 0; i < 4; i++) dq[i] = __halves2half2(dqh[i * 2], dqh[i * 2 + 1]);
}

#endif

#endif