#ifndef _quantize_cuh
#define _quantize_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

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
);

void adjust_error_row_cuda
(
    const float* hessian_inv,
    float* error,
    const float* weights,
    const float* quant,
    int c,
    int columns,
    int hcolumns
);

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
);

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
);

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
);

void vv_mul_sub_cuda
(
    const float* x,
    const float* y,
    float* z,
    int x_size,
    int y_size
);

#endif