#include "rope.cuh"
#include "util.cuh"
#include "matrix_view.cuh"

const int THREADS_X = 32;
const int THREADS_Y = 4;
const int MAX_POS_EMBEDDINGS = 32768;   // Actual number doesn't matter
const int MAX_ROWS = 32768;             // Actual number doesn't matter

__forceinline__ __device__ void rope_cuda_arr_neox
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
    half2 cos2_r = cos2_l; //cos_.item_half2(sincos_row, column + half_dim);
    half2 sin2_l = sin_.item_half2(sincos_row, column);
    half2 sin2_r = sin2_l; // sin_.item_half2(sincos_row, column + half_dim);
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

__forceinline__ __device__ void rope_cuda_arr_gptj
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
    if (column >= head_dim) return;

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

    half2 cos_01 = cos_.item_half2(sincos_row, column);   // cos[i], cos[i+1]
    half2_uint32 sin_01_ = half2_uint32(sin_.item_half2(sincos_row, column));
    sin_01_.as_uint32 ^= (1<<15);
    half2 sin_01 = sin_01_.as_half2;  // -sin[i], sin[i + 1]

    // Apply embedding to row

    half2 x_01 = x_.item_half2(row_offset, column);  // x[i], x[i+1]
    half2 x_10 = __lowhigh2highlow(x_01);  // x[i+1], x[i]
    half2 r = __hmul2(x_01, cos_01);
    r = __hfma2(x_10, sin_01, r);
    x_.set_half2(row_offset, column, r);
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
    int threads_y,
    const bool neox_style
)
{
    if (neox_style)
        rope_cuda_arr_neox(x, sin, cos, rows_per_batch, head_dim, num_heads, past_len, past_lens, threads_y);
    else
        rope_cuda_arr_gptj(x, sin, cos, rows_per_batch, head_dim, num_heads, past_len, past_lens, threads_y);
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
    int threads_y,
    const bool neox_style
)
{
    if (neox_style)
    {
        rope_cuda_arr_neox(x_q, sin, cos, rows_per_batch_q, head_dim, num_heads_q, past_len, past_lens, threads_y);
        rope_cuda_arr_neox(x_k, sin, cos, rows_per_batch_k, head_dim, num_heads_k, past_len, past_lens, threads_y);
    }
    else
    {
        rope_cuda_arr_gptj(x_q, sin, cos, rows_per_batch_q, head_dim, num_heads_q, past_len, past_lens, threads_y);
        rope_cuda_arr_gptj(x_k, sin, cos, rows_per_batch_k, head_dim, num_heads_k, past_len, past_lens, threads_y);
    }
}

void rope_cuda
(
    cudaStream_t stream,
    half* x,
    const half* sin,
    const half* cos,
    const int batch_size,
    const int rows_per_batch,
    const int head_dim,
    const int num_heads,
    const int past_len,
    const int32_t* past_lens,
    const bool neox_style
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

    rope_cuda_kernel<<<gridDim, blockDim, 0, stream>>>
    (
        x,
        sin,
        cos,
        rows_per_batch,
        head_dim,
        num_heads,
        past_len,
        past_lens,
        threads_y,
        neox_style
    );
}

void rope_cuda_qk
(
    cudaStream_t stream,
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
    const int32_t* past_lens,
    const bool neox_style,
    Graph* graph,
    int label
)
{
    // For large batch sizes we risk exceeding grid dimension of 65535, so shift to block dimension instead

    int threads_y = THREADS_Y;
    int rows_per_batch = max(rows_per_batch_q, rows_per_batch_k);
    while (DIVIDE(rows_per_batch, threads_y) > 65535) threads_y *= 2;

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = threads_y;
    gridDim.x = DIVIDE(head_dim / 2 / (neox_style ? 2 : 1), THREADS_X);
    gridDim.y = DIVIDE(rows_per_batch, threads_y);
    gridDim.z = batch_size;

    rope_cuda_qk_kernel<<<gridDim, blockDim, 0, stream>>>
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
        threads_y,
        neox_style
    );

    if (graph) graph->attach_label(stream, label, 0);
}

void rope_cuda_qk_update_q
(
    Graph* graph,
    int label,
    void* q
)
{
    graph->update_param_ptr(label, 0, 0, q);
}

void rope_cuda_qk_update_k
(
    Graph* graph,
    int label,
    void* k
)
{
    graph->update_param_ptr(label, 0, 1, k);
}

void rope_cuda_qk_update_past_len
(
    Graph* graph,
    int label,
    int past_len
)
{
    graph->update_param_int(label, 0, 9, past_len);
}

void rope_cuda_qk_update_past_lens
(
    Graph* graph,
    int label,
    void* past_lens
)
{
    graph->update_param_ptr(label, 0, 10, past_lens);
}
