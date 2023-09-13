#include "quantize.cuh"
#include "util.cuh"
#include <curand_kernel.h>
#include "compat.cuh"

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32

__global__ void quantize_kernel
(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scale,
    uint16_t* __restrict__ out_q,
    int rows,
    int columns,
    float qzero,
    float maxq
)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= columns) return;
    if (row >= rows) return;

    // Quantize

    float x = input[row * columns + column];
    float s = scale[column];
    x /= s;
    x = rintf(x);
    x += qzero;
    x = clamp(x, 0.0f, maxq);

    // Optionally save quant

    if (out_q)
    {
        uint16_t q = static_cast<uint16_t>(x);
        out_q[row * columns + column] = q;
    }

    half h_s = __float2half_rn(s);
    half h_x = __float2half_rn(x);
    half h_qzero = __float2half_rn(qzero);

    h_x = __hsub(h_x, h_qzero);
    h_x = __hmul(h_x, h_s);

    // Dequantize

//    x -= qzero;
//    x *= s;
    output[row * columns + column] = __half2float(h_x);
}

void quantize_cuda
(
    const float* input,
    float* output,
    const float* scale,
    uint16_t* out_q,
    int rows,
    int columns,
    float qzero,
    float maxq
)
{
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocks(DIVIDE(columns, BLOCKSIZE_X), DIVIDE(rows, BLOCKSIZE_Y));

//     DBGI2(rows, columns);
//     DBGF2(qzero, maxq);

    quantize_kernel<<<blocks, threads>>>
    (
        input,
        output,
        scale,
        out_q,
        rows,
        columns,
        qzero,
        maxq
    );
}

__global__ void quantize_err_kernel
(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scale,
    int rows,
    int columns,
    float qzero,
    float maxq,
    float err_norm,
    float min_p,
    float max_p,
    int p_grid
)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= columns) return;
    if (row >= rows) return;

    float w = input[row * columns + column];

    // Quantize

    for (int i = 0; i <= p_grid; i++)
    {
        float pi = __int2float_rn(i) / __int2float_rn(p_grid);
        float p = min_p * (1.0f - pi) + max_p * pi;

        float x = w;
        float s = scale[column] * p;
        x /= s;
        x = rintf(x);
        x += qzero;
        x = clamp(x, 0.0f, maxq);

        // Dequantize

        x -= qzero;
        x *= s;

        // Quantization error

        x = __powf(fabsf(x - w), err_norm);
        atomicAdd(&output[i * 128 + column % 128], x);
    }
}

void quantize_err_cuda
(
    const float* input,
    float* output,
    const float* scale,
    int rows,
    int columns,
    float qzero,
    float maxq,
    float err_norm,
    float min_p,
    float max_p,
    int p_grid
)
{
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocks(DIVIDE(columns, BLOCKSIZE_X), DIVIDE(rows, BLOCKSIZE_Y));

//     DBGI2(rows, columns);
//     DBGF2(qzero, maxq);

    quantize_err_kernel<<<blocks, threads>>>
    (
        input,
        output,
        scale,
        rows,
        columns,
        qzero,
        maxq,
        err_norm,
        min_p,
        max_p,
        p_grid
    );
}

__global__ void adjust_error_row_kernel
(
    const float* __restrict__ hessian_inv,
    float* __restrict__ error,
    const float* __restrict__ weights,
    const float* __restrict__ quant,
    int c,
    int columns,
    int hcolumns
)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;

    float d = hessian_inv[c * hcolumns + c];

    int idx = c * columns + column;
    float w = weights[idx];
    float q = quant[idx];
    error[idx] = (w - q) / d;
}

void adjust_error_row_cuda
(
    const float* hessian_inv,
    float* error,
    const float* weights,
    const float* quant,
    int c,
    int columns,
    int hcolumns
)
{
    dim3 threads(BLOCKSIZE_X, 1);
    dim3 blocks(DIVIDE(columns, BLOCKSIZE_X), 1);

    adjust_error_row_kernel<<<blocks, threads>>>(hessian_inv, error, weights, quant, c, columns, hcolumns);
}


// Compute z = z - x.T @ y

__global__ void vv_mul_sub_kernel
(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ z,
    int x_size,
    int y_size
)
{
    int y_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (y_idx >= y_size) return;
    if (x_idx >= x_size) return;
    int z_idx = y_size * x_idx + y_idx;

    float p = x[x_idx] * y[y_idx];

//     curandState state;
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     curand_init(1234, tid, clock64(), &state);
//     float r = curand_uniform(&state);
//     p *= r;

//    p *- 0.707106478;

    z[z_idx] -= p;
}

void vv_mul_sub_cuda
(
    const float* x,
    const float* y,
    float* z,
    int x_size,
    int y_size
)
{
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocks(DIVIDE(y_size, BLOCKSIZE_X), DIVIDE(x_size, BLOCKSIZE_Y));

    vv_mul_sub_kernel<<<blocks, threads>>>(x, y, z, x_size, y_size);
}
