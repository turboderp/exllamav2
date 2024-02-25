#include "h_add.cuh"
#include "util.cuh"
#include "../config.h"
#include "matrix_view.cuh"

#define NUM_THREADS_X 32
#define NUM_THREADS_Y 16

__global__ void cuda_vector_add_kernel
(
    half* __restrict__ dest,
    const half* __restrict__ source,
    const int height,
    const int width
)
{
    MatrixView_half_rw dest_(dest, height, width);
    MatrixView_half source_(source, 1, width);

    int offset_x = blockIdx.x * NUM_THREADS_X * 2 + threadIdx.x * 2;
    if (offset_x >= width) return;

    int offset_y = blockIdx.y * NUM_THREADS_Y;
    int end_y = min(offset_y + NUM_THREADS_Y, height);

    half2 v = source_.item_half2(0, offset_x);
    for (int y = offset_y; y < end_y; ++y)
    {
        half2* ptr = (half2*) dest_.item_ptr(y, offset_x);
        *ptr = __hadd2(v, *ptr);
    }
}

void cuda_vector_add_
(
    half* dest,
    const half* source,
    int height,
    int width
)
{
    dim3 blockDim, gridDim;
    blockDim.x = NUM_THREADS_X;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x = DIVIDE(width, NUM_THREADS_X * 2);
    gridDim.y = DIVIDE(height, NUM_THREADS_Y);
    gridDim.z = 1;

    cuda_vector_add_kernel<<<gridDim, blockDim>>>(dest, source, height, width);
}
