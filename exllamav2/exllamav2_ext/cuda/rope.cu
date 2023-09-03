#include "rope.cuh"
#include "util.cuh"
#include "matrix_view.cuh"

const int THREADS_X = 32;
const int THREADS_Y = 4;
const int MAX_POS_EMBEDDINGS = 32768;   // Actual number doesn't matter
const int MAX_ROWS = 32768;             // Actual number doesn't matter

typedef void (*fp_rope_cuda_kernel)
(
    half*,
    const half*,
    const half*,
    int,
    int,
    int,
    int,
    const uint32_t*
);

template<bool use_half2>
__global__ void rope_cuda_kernel
(
    half* __restrict__ x,
    const half* __restrict__ sin,
    const half* __restrict__ cos,
    int rows_per_batch,
    int head_dim,
    int num_heads,
    int past_len,
    const uint32_t* __restrict__ past_lens
)
{
    MatrixView_half_rw x_(x, MAX_ROWS, head_dim);
    MatrixView_half sin_(sin, MAX_POS_EMBEDDINGS, head_dim);
    MatrixView_half cos_(cos, MAX_POS_EMBEDDINGS, head_dim);

    int column = (blockIdx.x * THREADS_X + threadIdx.x); if constexpr (use_half2) column *= 2;
    int half_dim = head_dim / 2;
    if (column >= half_dim) return;

    int row = blockIdx.y * THREADS_Y + threadIdx.y;
    if (row >= rows_per_batch) return;
    int batch_offset = blockIdx.z * rows_per_batch;
    int row_offset = batch_offset + row;

    // Get sin and cos

    if (past_len == -1) past_len = past_lens[blockIdx.z];
    int sincos_row = past_len + row / num_heads;

    if constexpr (use_half2)
    {
        half2 cos2_l = cos_.item_half2(sincos_row, column);
        half2 cos2_r = cos_.item_half2(sincos_row, column + half_dim);
        half2 sin2_l = sin_.item_half2(sincos_row, column);
        half2 sin2_r = sin_.item_half2(sincos_row, column + half_dim);
        sin2_l = __hneg2(sin2_l);

        // Apply embedding to row

        half2 item2_l = x_.item_half2(row_offset, column);
        half2 item2_r = x_.item_half2(row_offset, column + half_dim);
        half2 item2_ls = __hmul2(item2_r, sin2_l);
        half2 item2_rs = __hmul2(item2_l, sin2_r);
        item2_l = __hfma2(item2_l, cos2_l, item2_ls);
        item2_r = __hfma2(item2_r, cos2_r, item2_rs);
        x_.set_half2(row_offset, column, item2_l);
        x_.set_half2(row_offset, column + half_dim, item2_r);
    }
    else
    {
        half cos_l = cos_.item(sincos_row, column);
        half cos_r = cos_.item(sincos_row, column + half_dim);
        half sin_l = sin_.item(sincos_row, column);
        half sin_r = sin_.item(sincos_row, column + half_dim);
        sin_l = __hneg(sin_l);

        // Apply embedding to row

        half item_l = x_.item(row_offset, column);
        half item_r = x_.item(row_offset, column + half_dim);
        half item_ls = __hmul(item_r, sin_l);
        half item_rs = __hmul(item_l, sin_r);
        item_l = __hfma(item_l, cos_l, item_ls);
        item_r = __hfma(item_r, cos_r, item_rs);
        x_.set(row_offset, column, item_l);
        x_.set(row_offset, column + half_dim, item_r);
    }
}

fp_rope_cuda_kernel pick_rope_cuda_kernel(bool use_half2)
{
    if (use_half2) return rope_cuda_kernel<true>;
    else           return rope_cuda_kernel<false>;
};

void rope_cuda
(
    half* x,
    const half* sin,
    const half* cos,
    const int batch_size,
    const int rows_per_batch,
    const int head_dim,
    const int num_heads,
    const int past_len,
    const uint32_t* past_lens
)
{
    bool use_half2 = true;

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = THREADS_Y;
    gridDim.x = DIVIDE(head_dim, THREADS_X) / (use_half2 ? 2 : 1);
    gridDim.y = DIVIDE(rows_per_batch, THREADS_Y);
    gridDim.z = batch_size;

    fp_rope_cuda_kernel kernel = pick_rope_cuda_kernel(use_half2);
    kernel<<<gridDim, blockDim>>>(x, sin, cos, rows_per_batch, head_dim, num_heads, past_len, past_lens);
}
