/* Adated from the CUDA samples https://docs.nvidia.com/cuda/cuda-samples/index.html.
   Changed from "natural order" Hadamard transform (larger strides before
   smaller strides) to the standard Hadamard transform (smaller strides before
   larger strides).
 */

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <cooperative_groups.h>

namespace cg = cooperative_groups;


///////////////////////////////////////////////////////////////////////////////
// Elementary(for vectors less than elementary size) in-shared memory
// combined radix-2 + radix-4 Fast Walsh Transform
///////////////////////////////////////////////////////////////////////////////
#define ELEMENTARY_LOG2SIZE 11

__global__ void fwtBatch1Kernel(float *d_Output, float *d_Input, int log2N)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    const int    N = 1 << log2N;
    const int base = blockIdx.x << log2N;

    //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
    extern __shared__ float s_data[];
    float *d_Src = d_Input  + base;
    float *d_Dst = d_Output + base;

    for (int pos = threadIdx.x; pos < N; pos += blockDim.x)
    {
        s_data[pos] = d_Src[pos];
    }

    int stride = 1;
    //Do single radix-2 stage for odd power of two
    if (log2N & 1)
    {
        cg::sync(cta);

        for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x)
        {
            int i0 = pos << 1;
            int i1 = i0 + 1;

            float D0 = s_data[i0];
            float D1 = s_data[i1];
            s_data[i0] = D0 + D1;
            s_data[i1] = D0 - D1;
        }
        stride <<= 1;
    }

    //Main radix-4 stages
    const int pos = threadIdx.x;

    for (; stride <= N >> 2; stride <<= 2)
    {
        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        cg::sync(cta);
        float D0 = s_data[i0];
        float D1 = s_data[i1];
        float D2 = s_data[i2];
        float D3 = s_data[i3];

        float T;
        T = D0;
        D0         = D0 + D2;
        D2         = T - D2;
        T = D1;
        D1         = D1 + D3;
        D3         = T - D3;
        T = D0;
        s_data[i0] = D0 + D1;
        s_data[i1] = T - D1;
        T = D2;
        s_data[i2] = D2 + D3;
        s_data[i3] = T - D3;
    }

    cg::sync(cta);

    for (int pos = threadIdx.x; pos < N; pos += blockDim.x)
    {
        d_Dst[pos] = s_data[pos];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////
__global__ void fwtBatch2Kernel(
    float *d_Output,
    float *d_Input,
    int stride
)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int   N = blockDim.x *  gridDim.x * 4;

    float *d_Src = d_Input  + blockIdx.y * N;
    float *d_Dst = d_Output + blockIdx.y * N;

    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    float D0 = d_Src[i0];
    float D1 = d_Src[i1];
    float D2 = d_Src[i2];
    float D3 = d_Src[i3];

    float T;
    T = D0;
    D0        = D0 + D2;
    D2        = T - D2;
    T = D1;
    D1        = D1 + D3;
    D3        = T - D3;
    T = D0;
    d_Dst[i0] = D0 + D1;
    d_Dst[i1] = T - D1;
    T = D2;
    d_Dst[i2] = D2 + D3;
    d_Dst[i3] = T - D3;
}

////////////////////////////////////////////////////////////////////////////////
// Put everything together: batched Fast Walsh Transform CPU front-end
////////////////////////////////////////////////////////////////////////////////
void fwtBatchGPU(float *d_Data, int batchSize, int log2N, cudaStream_t stream)
{
    int nMixedRadixPasses = log2N > ELEMENTARY_LOG2SIZE ? ELEMENTARY_LOG2SIZE - (log2N - ELEMENTARY_LOG2SIZE) % 2 : log2N;
    int N = 1 << nMixedRadixPasses;
    int curBatchSize = batchSize << (log2N - nMixedRadixPasses);

    // (N + 3) / 4 to handle the case of N == 2
    fwtBatch1Kernel<<<curBatchSize, (N + 3) / 4, N * sizeof(float), stream>>>(
        d_Data,
        d_Data,
        nMixedRadixPasses
    );

    const int THREAD_N = 256;
    dim3 grid((1 << log2N) / (4 * THREAD_N), batchSize, 1);
    for (int logSize = nMixedRadixPasses + 2; logSize <= log2N; logSize += 2)
    {
      fwtBatch2Kernel<<<grid, THREAD_N, 0, stream>>>(d_Data, d_Data, (1 << logSize) / 4);
    }

}
