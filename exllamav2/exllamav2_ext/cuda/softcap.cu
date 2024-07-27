#include "softcap.cuh"
#include "util.cuh"
#include "../config.h"
#include "matrix_view.cuh"

#define NUM_THREADS 256

__global__ void cuda_softcap_kernel
(
    float* __restrict__ x,
    const uint64_t numel,
    const float scale
)
{
    uint64_t idx = (uint64_t)blockIdx.x * NUM_THREADS + (uint64_t)threadIdx.x;
    if (idx >= numel) return;

    float v = x[idx];
    v /= scale;
    v = tanhf(v);
    v *= scale;
    x[idx] = v;
}

void softcap_cuda_
(
    cudaStream_t stream,
    float* x,
    const uint64_t numel,
    const float scale
)
{
    dim3 blockDim, gridDim;
    blockDim.x = NUM_THREADS;
    gridDim.x = DIVIDE(numel, NUM_THREADS);

    cuda_softcap_kernel<<<gridDim, blockDim, 0, stream>>>(x, numel, scale);
}

// TODO: Profile

__global__ void h_cuda_softcap_kernel
(
    half* __restrict__ x,
    const uint64_t numel,
    const float scale
)
{
    uint64_t idx = (uint64_t)blockIdx.x * NUM_THREADS + (uint64_t)threadIdx.x;
    idx *= 2;
    if (idx >= numel) return;
    half2* x2 = (half2*)(x + idx);
    half2 v01 = *x2;
    float v0 = __low2float(v01);
    float v1 = __high2float(v01);
    v0 /= scale;
    v1 /= scale;
    v0 = tanhf(v0);
    v1 = tanhf(v1);
    v0 *= scale;
    v1 *= scale;
    v01 = __floats2half2_rn(v0, v1);
    *x2 = v01;
}

void h_softcap_cuda_
(
    cudaStream_t stream,
    half* x,
    const uint64_t numel,
    const float scale
)
{
    dim3 blockDim, gridDim;
    blockDim.x = NUM_THREADS;
    gridDim.x = DIVIDE(numel / 2, NUM_THREADS);

    h_cuda_softcap_kernel<<<gridDim, blockDim, 0, stream>>>(x, numel, scale);
}

