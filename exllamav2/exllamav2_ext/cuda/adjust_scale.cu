#include "adjust_scale.cuh"
#include "util.cuh"

#define WARPSIZE 32
#define BLOCKSIZE_Y 32

__global__ void adjust_scale_kernel
(
    const float* __restrict__ qscale,
    const float* __restrict__ x,
    float* __restrict__ grid,
    const float max_adjust,
    const float min_adjust,
    const int adjust_steps,
    const int rows,
    const int columns,
    const float norm,
    const float qzero,
    const float maxq
)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;

    int step = blockIdx.y * blockDim.y + threadIdx.y;
    if (step >= adjust_steps) return;

    float a = (float)step / (float)adjust_steps;
    float adjust = min_adjust * a + max_adjust * (1.0 - a);
    float qs = qscale[column] * adjust;

    for (int row = 0; row < rows; row++)
    {
        float w = x[row * columns + column];

        // Quantize

        float qw = w;
        qw /= qs;
        qw = rintf(qw);
        qw += qzero;
        qw = clamp(qw, 0, maxq);

        // Dequantize

        qw -= qzero;
        qw *= qs;

        // Error

        float err = fabs(qw - w);
        err = __powf(err, norm);
        atomicAdd(&grid[step], err);
    }
}

float adjust_scale_cuda
(
    const float* qscale,
    const float* x,
    float* grid,
    const float max_adjust,
    const float min_adjust,
    const int adjust_steps,
    const int rows,
    const int columns,
    const float norm,
    const int qzero,
    const int maxq
)
{
    dim3 threads(WARPSIZE, BLOCKSIZE_Y);
    dim3 blocks(DIVIDE(columns, threads.x), DIVIDE(adjust_steps, threads.y));

    adjust_scale_kernel<<<blocks, threads>>>
    (
        qscale,
        x,
        grid,
        max_adjust,
        min_adjust,
        adjust_steps,
        rows,
        columns,
        norm,
        static_cast<float>(qzero),
        static_cast<float>(maxq)
    );

    float* results = (float*) malloc(adjust_steps * sizeof(float));
    cudaMemcpy(results, grid, adjust_steps * sizeof(float), cudaMemcpyDeviceToHost);

    int besti = 0;
    float besterr = 9999999.0f;
    for (int i = 0; i < adjust_steps; i++)
    {
        if (results[i] < besterr)
        {
            besti = i;
            besterr = results[i];
        }
    }

    float a = (float)besti / (float)adjust_steps;
    float adjust = min_adjust * a + max_adjust * (1.0 - a);

    //DBGF(adjust);

    return adjust;
}