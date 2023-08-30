#include "pack_tensor.cuh"
#include "util.cuh"

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 16

// Pack rows:
// 0000 0000 0000 aaaa  0000 0000 0000 bbbb  0000 0000 0000 cccc  ...  -> hhhh gggg ffff eeee dddd cccc bbbb aaaa

__global__ void pack_rows_4_kernel
(
    const uint16_t* __restrict__ input,
    uint32_t* __restrict__ output,
    int rows,
    int out_columns
)
{
    int out_column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows) return;
    if (out_column >= out_columns) return;

    uint32_t packed = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        uint16_t x = input[row * out_columns * 8 + out_column * 8 + i];
        x -= 1;
        packed |= (((uint32_t)x) << (i * 4));
    }

    output[row * out_columns + out_column] = packed;
}

void pack_rows_4_cuda
(
    const uint16_t* input,
    uint32_t* output,
    const int rows,
    const int columns
)
{
    int out_columns = columns * 4 / 32;

    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocks(DIVIDE(out_columns, BLOCKSIZE_X), DIVIDE(rows, BLOCKSIZE_Y));

    pack_rows_4_kernel<<<blocks, threads>>>(input, output, rows, out_columns);
}

// Pack rows:
// 0000 0000 0000 aaaa  0000 0000 0000 bbbb  0000 0000 0000 cccc  ...  -> hhhh gggg ffff eeee dddd cccc bbbb aaaa

__global__ void pack_rows_6_kernel
(
    const uint16_t* __restrict__ input,
    uint32_t* __restrict__ output,
    int rows,
    int out_columns
)
{
    int out_column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows) return;
    if (out_column >= out_columns) return;

    uint32_t packed = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        uint16_t x = input[row * out_columns * 8 + out_column * 8 + i];
        x -= 1;
        packed |= (((uint32_t)x) << (i * 4));
    }

    output[row * out_columns + out_column] = packed;
}

void pack_rows_6_cuda
(
    const uint16_t* input,
    uint32_t* output,
    const int rows,
    const int columns
)
{
    int out_columns = columns * 6 / 32;

    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocks(DIVIDE(out_columns, BLOCKSIZE_X), DIVIDE(rows, BLOCKSIZE_Y));

    pack_rows_6_kernel<<<blocks, threads>>>(input, output, rows, out_columns);
}

// Pack columns

__forceinline__ __device__ uint32_t wshift(uint32_t x, int j)
{
    if (j < 0)
    {
        if (j <= -32) return 0;  // Else undefined in CUDA
        return x >> (-j);
    }
    else
    {
        if (j >= 32) return 0;  // Else undefined in CUDA
        return x << j;
    }
}

template<int bits>
__global__ void pack_columns_kernel
(
    const uint16_t* __restrict__ input,
    uint32_t* __restrict__ output,
    int out_rows,
    int columns
)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column >= columns) return;
    if (out_row >= out_rows) return;

    uint32_t x;

    if constexpr (bits == 2)
    {
        int row = out_row * 32 / 2;
        uint32_t packed = 0;

        # pragma unroll
        for (int i = 0, j = 0; i < 16; i++, j += 2)
        {
            x = (uint32_t) input[(row + i) * columns + column];
            packed |= (x << (i * 2));
        }
        output[out_row * columns + column] = packed;
    }

    if constexpr (bits == 3)
    {
        if (out_row % 3) return;  // Only run for every third row
        int row = out_row * 32 / 3;
        uint32_t packed0 = 0;
        uint32_t packed1 = 0;
        uint32_t packed2 = 0;

        #pragma unroll
        for (int i = 0, j = 0; i < 32; i++, j += 3)
        {
            x = (uint32_t) input[(row + i) * columns + column];
            packed0 |= wshift(x, j);
            packed1 |= wshift(x, j - 32);
            packed2 |= wshift(x, j - 64);
        }

        output[(out_row + 0) * columns + column] = packed0;
        output[(out_row + 1) * columns + column] = packed1;
        output[(out_row + 2) * columns + column] = packed2;
    }

    if constexpr (bits == 4)
    {
        int row = out_row * 32 / 4;
        uint32_t packed = 0;

        #pragma unroll
        for (int i = 0, j = 0; i < 8; i++, j += 4)
        {
            x = (uint32_t) input[(row + i) * columns + column];
            packed |= (x << j);
        }
        output[out_row * columns + column] = packed;
    }

    if constexpr (bits == 5)
    {
        if (out_row % 5) return;  // Only run for every fifth row
        int row = out_row * 32 / 5;
        uint32_t packed0 = 0;
        uint32_t packed1 = 0;
        uint32_t packed2 = 0;
        uint32_t packed3 = 0;
        uint32_t packed4 = 0;

        #pragma unroll
        for (int i = 0, j = 0; i < 32; i++, j += 5)
        {
            x = (uint32_t) input[(row + i) * columns + column];
            packed0 |= wshift(x, j);
            packed1 |= wshift(x, j - 32);
            packed2 |= wshift(x, j - 64);
            packed3 |= wshift(x, j - 96);
            packed4 |= wshift(x, j - 128);
        }

        output[(out_row + 0) * columns + column] = packed0;
        output[(out_row + 1) * columns + column] = packed1;
        output[(out_row + 2) * columns + column] = packed2;
        output[(out_row + 3) * columns + column] = packed3;
        output[(out_row + 4) * columns + column] = packed4;
    }

    if constexpr (bits == 6)
    {
        if (out_row % 3) return;  // Only run for every third row
        int row = out_row * 32 / 6;
        uint32_t packed0 = 0;
        uint32_t packed1 = 0;
        uint32_t packed2 = 0;

        #pragma unroll
        for (int i = 0, j = 0; i < 16; i++, j += 6)
        {
            x = (uint32_t) input[(row + i) * columns + column];
            packed0 |= wshift(x, j);
            packed1 |= wshift(x, j - 32);
            packed2 |= wshift(x, j - 64);
        }

        output[(out_row + 0) * columns + column] = packed0;
        output[(out_row + 1) * columns + column] = packed1;
        output[(out_row + 2) * columns + column] = packed2;
    }

    if constexpr (bits == 8)
    {
        int row = out_row * 32 / 8;
        uint32_t packed = 0;

        #pragma unroll
        for (int i = 0, j = 0; i < 4; i++, j += 8)
        {
            x = (uint32_t) input[(row + i) * columns + column];
            packed |= (x << j);
        }

        output[out_row * columns + column] = packed;
    }
}

void pack_columns_cuda
(
    const uint16_t* input,
    uint32_t* output,
    const int in_rows,
    const int out_rows,
    const int columns,
    const int bits
)
{
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocks(DIVIDE(columns, BLOCKSIZE_X), DIVIDE(out_rows, BLOCKSIZE_Y));

    if (bits == 2) pack_columns_kernel<2><<<blocks, threads>>>(input, output, out_rows, columns);
    if (bits == 3) pack_columns_kernel<3><<<blocks, threads>>>(input, output, out_rows, columns);
    if (bits == 4) pack_columns_kernel<4><<<blocks, threads>>>(input, output, out_rows, columns);
    if (bits == 5) pack_columns_kernel<5><<<blocks, threads>>>(input, output, out_rows, columns);
    if (bits == 6) pack_columns_kernel<6><<<blocks, threads>>>(input, output, out_rows, columns);
    if (bits == 8) pack_columns_kernel<8><<<blocks, threads>>>(input, output, out_rows, columns);
}

