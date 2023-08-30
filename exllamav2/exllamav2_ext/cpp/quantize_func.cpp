#include "quantize_func.h"
#include "../cuda/quantize.cuh"

void quantize_range
(
    torch::Tensor quant,
    torch::Tensor scale,
    torch::Tensor out_q,
    float qzero,
    float maxq,
    torch::Tensor hessian_inv,
    torch::Tensor weights,
    torch::Tensor error,
    int a,
    int b
)
{
    int columns = weights.size(1);
    int hcolumns = hessian_inv.size(1);

    for (int c = a; c < b; c++)
    {
        quantize_cuda
        (
            ((const float*) weights.data_ptr()) + c * columns,
            ((float*) quant.data_ptr()) + c * columns,
            (const float*) scale.data_ptr(),
            out_q.device().is_meta() ? NULL : ((uint16_t*) out_q.data_ptr()) + c * columns,
            1,
            columns,
            qzero,
            maxq
        );

        adjust_error_row_cuda
        (
            (const float*) hessian_inv.data_ptr(),
            (float*) error.data_ptr(),
            (const float*) weights.data_ptr(),
            (const float*) quant.data_ptr(),
            c,
            columns,
            hcolumns
        );

        vv_mul_sub_cuda
        (
            ((const float*) hessian_inv.data_ptr()) + c * hcolumns + c,
            ((const float*) error.data_ptr()) + c * columns,
            ((float*) weights.data_ptr()) + c * columns,
            b - c,
            columns
        );
    }

    torch::Tensor x = hessian_inv.slice(0, a, b).slice(1, b).transpose(0, 1);
    torch::Tensor y = error.slice(0, a, b);
    weights.slice(0, b).addmm_(x, y, 1.0f, -1.0f);
}
