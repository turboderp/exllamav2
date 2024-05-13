#include "quantize.cuh"
#include "util.cuh"
#include <curand_kernel.h>
#include "compat.cuh"

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32
#define MAX_P_GRID 64

__global__ void quantize_rtn_kernel
(
    float*       __restrict__ weights,          // input  weights           [rows, columns]
    const float* __restrict__ scale,            // input  scales            [1, columns]
    uint16_t*    __restrict__ out_q,            // output qweights          [rows, columns]
    int row,                                    // row index to quantize
    int rows,                                   // num rows
    int columns,                                // num columns
    float qzero,                                // 2^(bits - 1)
    float maxq                                  // (2^bits) - 1
)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;

    uint64_t idx = (uint64_t)row * (uint64_t)columns + (uint64_t)column;

    // Quantize

    float x = weights[idx];
    float s = scale[column];
    x /= s;
    x = rintf(x);
    x += qzero;
    x = clamp(x, 0.0f, maxq);

    // Optionally save quant

    if (out_q) out_q[idx] = static_cast<uint16_t>(x);

    // Downcast while quantizing

    half h_s = __float2half_rn(s);
    half h_x = __float2half_rn(x);
    half h_qzero = __float2half_rn(qzero);

    // Dequantize

    h_x = __hsub(h_x, h_qzero);
    h_x = __hmul(h_x, h_s);
    float q = __half2float(h_x);
    weights[idx] = q;
}

void quantize_rtn_cuda
(
    float*          weights,
    const float*    scale,
    uint16_t*       out_q,
    int row,
    int rows,
    int columns,
    float qzero,
    float maxq
)
{
    dim3 threads(BLOCKSIZE_X, 1);
    dim3 blocks(DIVIDE(columns, BLOCKSIZE_X), 1);

    quantize_rtn_kernel<<<blocks, threads>>>
    (
        weights,
        scale,
        out_q,
        row,
        rows,
        columns,
        qzero,
        maxq
    );
}

__global__ void fused_quantize_adjust_kernel
(
    const float* __restrict__ weights,          // input  weights           [rows, columns]
    float*       __restrict__ quant,            // output quantized weights [rows, columns]
    const float* __restrict__ scale,            // input  scales            [1, columns]
    uint16_t*    __restrict__ out_q,            // output qweights          [rows, columns]
    const float* __restrict__ hessian_inv,      // input hessian            [rows, rows]
    float*       __restrict__ error,            // output error             [rows, columns]
    int row,                                    // row index to quantize
    int rows,                                   // num rows
    int columns,                                // num columns
    float qzero,                                // 2^(bits - 1)
    float maxq                                  // (2^bits) - 1
)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;

    uint64_t idx = (uint64_t)row * (uint64_t)columns + (uint64_t)column;

    // Quantize

    float x = weights[idx];
    float s = scale[column];
    x /= s;
    x = rintf(x);
    x += qzero;
    x = clamp(x, 0.0f, maxq);

    // Optionally save quant

    if (out_q) out_q[idx] = static_cast<uint16_t>(x);

    // Downcast while quantizing

    half h_s = __float2half_rn(s);
    half h_x = __float2half_rn(x);
    half h_qzero = __float2half_rn(qzero);

    // Dequantize

    h_x = __hsub(h_x, h_qzero);
    h_x = __hmul(h_x, h_s);
    float q = __half2float(h_x);
    quant[idx] = q;

    // Adjust error

    uint64_t d_idx = (uint64_t)row * (uint64_t)rows + (uint64_t)row;
    float d = hessian_inv[d_idx];  // H diagonal
    float w = weights[idx];
    error[idx] = (w - q) / d;
}

void fused_quantize_adjust_cuda
(
    const float*    weights,
    float*          quant,
    const float*    scale,
    uint16_t*       out_q,
    const float*    hessian_inv,
    float*          error,
    int row,
    int rows,
    int columns,
    float qzero,
    float maxq
)
{
    dim3 threads(BLOCKSIZE_X, 1);
    dim3 blocks(DIVIDE(columns, BLOCKSIZE_X), 1);

    fused_quantize_adjust_kernel<<<blocks, threads>>>
    (
        weights,
        quant,
        scale,
        out_q,
        hessian_inv,
        error,
        row,
        rows,
        columns,
        qzero,
        maxq
    );
}

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

    uint64_t idx = (uint64_t)row * (uint64_t)columns + (uint64_t)column;
    float x = input[idx];
    float s = scale[column];
    x /= s;
    x = rintf(x);
    x += qzero;
    x = clamp(x, 0.0f, maxq);

    // Optionally save quant

    if (out_q)
    {
        uint16_t q = static_cast<uint16_t>(x);
        out_q[idx] = q;
    }

    half h_s = __float2half_rn(s);
    half h_x = __float2half_rn(x);
    half h_qzero = __float2half_rn(qzero);

    h_x = __hsub(h_x, h_qzero);
    h_x = __hmul(h_x, h_s);

    // Dequantize

    output[idx] = __half2float(h_x);
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
    int column_ = blockIdx.x * blockDim.x + threadIdx.x;
    int column = column_ * 4;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= columns) return;
    if (row >= rows) return;

    float clamp_min = -qzero;
    float clamp_max = maxq - qzero;

    uint64_t idx = (uint64_t)row * (uint64_t)columns + (uint64_t)column;
    float4 sc4 = *((float4*) (scale + column));
    float4 w4 = *((float4*) (input + idx));

    for (int i = 0; i <= p_grid; i++)
    {
        float pi = __int2float_rn(i) / __int2float_rn(p_grid);
        float p = min_p * (1.0f - pi) + max_p * pi;

        float4 s4 = sc4;
        s4.x *= p;
        s4.y *= p;
        s4.z *= p;
        s4.w *= p;

        float err = __powf(fabsf(clamp(rintf(w4.x / s4.x), clamp_min, clamp_max) * s4.x - w4.x), err_norm);
        err +=      __powf(fabsf(clamp(rintf(w4.y / s4.y), clamp_min, clamp_max) * s4.y - w4.y), err_norm);
        err +=      __powf(fabsf(clamp(rintf(w4.z / s4.z), clamp_min, clamp_max) * s4.z - w4.z), err_norm);
        err +=      __powf(fabsf(clamp(rintf(w4.w / s4.w), clamp_min, clamp_max) * s4.w - w4.w), err_norm);

        atomicAdd(&output[i * 128 + column_ % 128], err);
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
    dim3 blocks(DIVIDE(columns / 4, BLOCKSIZE_X), DIVIDE(rows, BLOCKSIZE_Y));

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
    int y_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int x_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (y_idx >= y_size) return;
    if (x_idx >= x_size) return;

    uint64_t z_idx = (uint64_t)y_size * (uint64_t)x_idx + (uint64_t)y_idx;

    float vx = x[x_idx];
    float4 vy = *((float4*) (y + y_idx));
    float4 vz = *((float4*) (z + z_idx));
    vz.x -= vy.x * vx;
    vz.y -= vy.y * vx;
    vz.z -= vy.z * vx;
    vz.w -= vy.w * vx;
    *((float4*) (z + z_idx)) = vz;
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
    dim3 blockDim, gridDim;
    blockDim.x = BLOCKSIZE_X;
    blockDim.y = BLOCKSIZE_Y;
    blockDim.z = 1;
    gridDim.x = DIVIDE(y_size / 4, BLOCKSIZE_X);
    gridDim.y = DIVIDE(x_size, BLOCKSIZE_Y);
    gridDim.z = 1;

    vv_mul_sub_kernel<<<gridDim, blockDim>>>(x, y, z, x_size, y_size);
}
