#include "rope.cuh"
#include "util.cuh"
#include "matrix_view.cuh"

const int THREADS_X = 32;
const int THREADS_Y = 4;
const int MAX_POS_EMBEDDINGS = 32768;   // Actual number doesn't matter
const int MAX_ROWS = 32768;             // Actual number doesn't matter

__forceinline__ __device__ void rope_cuda_arr
(
    half* __restrict__ x,
    const half* __restrict__ sin,
    const half* __restrict__ cos,
    int rows_per_batch,
    int head_dim,
    int num_heads,
    int past_len,
    const int32_t* __restrict__ past_lens,
    int threads_y
)
{
    MatrixView_half_rw x_(x, MAX_ROWS, head_dim);
    MatrixView_half sin_(sin, MAX_POS_EMBEDDINGS, head_dim);
    MatrixView_half cos_(cos, MAX_POS_EMBEDDINGS, head_dim);

    int column = (blockIdx.x * THREADS_X + threadIdx.x) * 2;
    int half_dim = head_dim / 2;
    if (column >= half_dim) return;

    int row = blockIdx.y * threads_y + threadIdx.y;
    if (row >= rows_per_batch) return;
    int batch_offset = blockIdx.z * rows_per_batch;
    int row_offset = batch_offset + row;

    // Get sin and cos

    if (past_len == -1)
    {
        past_len = past_lens[blockIdx.z];
        past_len = max(past_len, 0);
    }
    else if (past_lens)
    {
        past_len += past_lens[blockIdx.z];
    }

    int sincos_row = past_len + row / num_heads;
    sincos_row = max(sincos_row, 0);

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

__global__ void rope_cuda_kernel
(
    half* __restrict__ x,
    const half* __restrict__ sin,
    const half* __restrict__ cos,
    int rows_per_batch,
    int head_dim,
    int num_heads,
    int past_len,
    const int32_t* __restrict__ past_lens,
    int threads_y
)
{
    rope_cuda_arr(x, sin, cos, rows_per_batch, head_dim, num_heads, past_len, past_lens, threads_y);
}

__global__ void rope_cuda_qk_kernel
(
    half* __restrict__ x_q,
    half* __restrict__ x_k,
    const half* __restrict__ sin,
    const half* __restrict__ cos,
    int rows_per_batch_q,
    int rows_per_batch_k,
    int head_dim,
    int num_heads_q,
    int num_heads_k,
    int past_len,
    const int32_t* __restrict__ past_lens,
    int threads_y
)
{
    rope_cuda_arr(x_q, sin, cos, rows_per_batch_q, head_dim, num_heads_q, past_len, past_lens, threads_y);
    rope_cuda_arr(x_k, sin, cos, rows_per_batch_k, head_dim, num_heads_k, past_len, past_lens, threads_y);
}

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
    const int32_t* past_lens
)
{
    // For large batch sizes we risk exceeding grid dimension of 65535, so shift to block dimension instead

    int threads_y = THREADS_Y;
    while (DIVIDE(rows_per_batch, threads_y) > 65535) threads_y *= 2;

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = threads_y;
    gridDim.x = DIVIDE(head_dim / 2, THREADS_X);
    gridDim.y = DIVIDE(rows_per_batch, threads_y);
    gridDim.z = batch_size;

    rope_cuda_kernel<<<gridDim, blockDim>>>
    (
        x,
        sin,
        cos,
        rows_per_batch,
        head_dim,
        num_heads,
        past_len,
        past_lens,
        threads_y
    );
}

void rope_cuda_qk
(
    half* x_q,
    half* x_k,
    const half* sin,
    const half* cos,
    const int batch_size,
    const int rows_per_batch_q,
    const int rows_per_batch_k,
    const int head_dim,
    const int num_heads_q,
    const int num_heads_k,
    const int past_len,
    const int32_t* past_lens
)
{
    // For large batch sizes we risk exceeding grid dimension of 65535, so shift to block dimension instead

    int threads_y = THREADS_Y;
    int rows_per_batch = max(rows_per_batch_q, rows_per_batch_k);
    while (DIVIDE(rows_per_batch, threads_y) > 65535) threads_y *= 2;

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = threads_y;
    gridDim.x = DIVIDE(head_dim / 2, THREADS_X);
    gridDim.y = DIVIDE(rows_per_batch, threads_y);
    gridDim.z = batch_size;

    rope_cuda_qk_kernel<<<gridDim, blockDim>>>
    (
        x_q,
        x_k,
        sin,
        cos,
        rows_per_batch_q,
        rows_per_batch_k,
        head_dim,
        num_heads_q,
        num_heads_k,
        past_len,
        past_lens,
        threads_y
    );
}


