
#include "cpp/quantize_func.h"

void pack_columns
(
    torch::Tensor input,
    torch::Tensor output,
    int bits
);

void pack_rows_4
(
    torch::Tensor input,
    torch::Tensor output
);

void quantize_err
(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    float qzero,
    float maxq,
    float err_norm,
    float min_p,
    float max_p,
    int p_grid
);

void quantize
(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    torch::Tensor out_q,
    float qzero,
    float maxq
);
