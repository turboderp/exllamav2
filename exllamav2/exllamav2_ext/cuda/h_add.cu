#include "h_add.cuh"
#include "util.cuh"
#include "../config.h"
#include "matrix_view.cuh"

#define NUM_THREADS_X 32
#define NUM_THREADS_Y 16

#define NUM_EL_INT4 1024
#define NUM_THREADS_INT4 (NUM_EL_INT4 / 8)
#define NUM_THREADS_Y_INT4 2

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

__global__ void cuda_vector_set_kernel
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
        *ptr = v;
    }
}

typedef struct __align__(16)
{
    half2 x;
    half2 y;
    half2 z;
    half2 w;
} half8;

__global__ void cuda_vector_add_int4_kernel
(
    half* __restrict__ dest,
    const half* __restrict__ source,
    const int height,
    const int width
)
{
    int source_offset = blockIdx.x * NUM_EL_INT4;
    int dest_y = blockIdx.y * NUM_THREADS_Y_INT4;
    int max_y = min(height, dest_y + NUM_THREADS_Y_INT4);
    int dest_offset = source_offset + width * dest_y;
    int block_index = threadIdx.x * 8;

    if (source_offset + block_index >= width) return;

    half8* a_ptr = (half8*) (source + source_offset + block_index);
    half8 a;
    ((uint4*)&a)[0] = ((uint4*)a_ptr)[0];

    for (int i = dest_y; i < max_y; ++i, dest_offset += width)
    {
        half8* b_ptr = (half8*) (dest + dest_offset + block_index);
        half8 b;
        ((uint4*)&b)[0] = ((uint4*)b_ptr)[0];

        b.x = __hadd2(b.x, a.x);
        b.y = __hadd2(b.y, a.y);
        b.z = __hadd2(b.z, a.z);
        b.w = __hadd2(b.w, a.w);

        ((uint4*)b_ptr)[0] = ((uint4*)&b)[0];
    }
}

__global__ void cuda_vector_set_int4_kernel
(
    half* __restrict__ dest,
    const half* __restrict__ source,
    const int height,
    const int width
)
{
    int source_offset = blockIdx.x * NUM_EL_INT4;
    int dest_y = blockIdx.y * NUM_THREADS_Y_INT4;
    int max_y = min(height, dest_y + NUM_THREADS_Y_INT4);
    int dest_offset = source_offset + width * dest_y;
    int block_index = threadIdx.x * 8;

    if (source_offset + block_index >= width) return;

    int4* a_ptr = (int4*) (source + source_offset + block_index);
    int4 a = *a_ptr;

    for (int i = dest_y; i < max_y; ++i, dest_offset += width)
    {
        int4* b_ptr = (int4*) (dest + dest_offset + block_index);
        *b_ptr = a;
    }
}

void cuda_vector_add_
(
    cudaStream_t stream,
    half* dest,
    const half* source,
    int height,
    int width
)
{
    if (width % 8 == 0)
    {
        dim3 blockDim, gridDim;
        blockDim.x = min(NUM_THREADS_INT4, width / 8);
        gridDim.x = DIVIDE(width, NUM_EL_INT4);
        gridDim.y = DIVIDE(height, NUM_THREADS_Y_INT4);

        cuda_vector_add_int4_kernel<<<gridDim, blockDim, 0, stream>>>(dest, source, height, width);
    }
    else
    {
        dim3 blockDim, gridDim;
        blockDim.x = NUM_THREADS_X;
        gridDim.x = DIVIDE(width, NUM_THREADS_X * 2);
        gridDim.y = DIVIDE(height, NUM_THREADS_Y);

        cuda_vector_add_kernel<<<gridDim, blockDim, 0, stream>>>(dest, source, height, width);
    }
}

void cuda_vector_set_
(
    cudaStream_t stream,
    half* dest,
    const half* source,
    int height,
    int width
)
{
    if (width % 8 == 0)
    {
        dim3 blockDim, gridDim;
        blockDim.x = min(NUM_THREADS_INT4, width / 8);
        gridDim.x = DIVIDE(width, NUM_EL_INT4);
        gridDim.y = DIVIDE(height, NUM_THREADS_Y_INT4);

        cuda_vector_set_int4_kernel<<<gridDim, blockDim, 0, stream>>>(dest, source, height, width);
    }
    else
    {
        dim3 blockDim, gridDim;
        blockDim.x = NUM_THREADS_X;
        gridDim.x = DIVIDE(width, NUM_THREADS_X * 2);
        gridDim.y = DIVIDE(height, NUM_THREADS_Y);

        cuda_vector_set_kernel<<<gridDim, blockDim, 0, stream>>>(dest, source, height, width);
    }
}

