#include "q_attn.cuh"
#include "q_gemm.cuh"
#include "rms_norm.cuh"
#include "layer_norm.cuh"
#include "head_norm.cuh"
#include "rope.cuh"
#include "util.cuh"
#include "lora.cuh"
#include "../config.h"

const int THREADS_X = 32;
const int THREADS_Y = 1;
const int THREADS_Z = 4;
const int BLOCKSIZE_X = 2; // 2*half == 1*uint32_t
const int BLOCKSIZE_Z = 4; // num_heads must be divisible by BLOCKSIZE_Z

enum KernelLabels
{
    NORM = 1,
    Q,
    K,
    V,
    Q_NORM,
    K_NORM,
    ROPE
};

__global__ void update_cache_kernel
(
    const half* __restrict__ key_states,
    const half* __restrict__ value_states,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    const int head_dim,
    const int num_kv_heads,
    const int q_len,
    const int cache_seq_len,
    const int past_len
)
{
    //int state_shape[]  = {              num_kv_heads,                     q_len, head_dim };
    int state_stride[] = {                  head_dim,   head_dim * num_kv_heads,        1 };
    int state_pos[]    = {                         0,                         0,        0 };

    //int cache_shape[]  = {              num_kv_heads,             cache_seq_len, head_dim };
    int cache_stride[] = {  cache_seq_len * head_dim,                  head_dim,        1 };
    int cache_pos[]    = {                         0,                  past_len,        0 };

    int size[]         = {              num_kv_heads,                     q_len, head_dim };

    int x = (blockIdx.x * THREADS_X + threadIdx.x) * BLOCKSIZE_X;
    int y = blockIdx.y * THREADS_Y + threadIdx.y;
    int z = (blockIdx.z * THREADS_Z + threadIdx.z) * BLOCKSIZE_Z;

    if (x >= size[2]) return;
    if (y >= size[1]) return;
    if (z >= size[0]) return;

    int state_offset = (z + state_pos[0]) * state_stride[0] + (y + state_pos[1]) * state_stride[1] + (x + state_pos[2]) * state_stride[2];
    int cache_offset = (z + cache_pos[0]) * cache_stride[0] + (y + cache_pos[1]) * cache_stride[1] + (x + cache_pos[2]) * cache_stride[2];

    const uint32_t* key_ptr   = (uint32_t*) (key_states   + state_offset);
    const uint32_t* value_ptr = (uint32_t*) (value_states + state_offset);
    uint32_t* key_cache_ptr   = (uint32_t*) (key_cache    + cache_offset);
    uint32_t* value_cache_ptr = (uint32_t*) (value_cache  + cache_offset);

    #pragma unroll
    for (int k = 0; k < BLOCKSIZE_Z; k++)
    {
        *key_cache_ptr = *key_ptr;
        key_ptr += state_stride[0] / BLOCKSIZE_X;
        key_cache_ptr += cache_stride[0] / BLOCKSIZE_X;
    }

    #pragma unroll
    for (int k = 0; k < BLOCKSIZE_Z; k++)
    {
        *value_cache_ptr = *value_ptr;
        value_ptr += state_stride[0] / BLOCKSIZE_X;
        value_cache_ptr += cache_stride[0] / BLOCKSIZE_X;
    }
}

QAttn::QAttn
(
    half* _layernorm,
    half* _layernorm_bias,
    bool _layernorm_is_rms,
    float _norm_epsilon,
    QMatrix* _q_proj,
    QMatrix* _k_proj,
    QMatrix* _v_proj,
    QMatrix* _o_proj,
    half* _temp_state,
//     half* _temp_q,
//     half* _temp_k,
//     half* _temp_v,
    half* _temp_dq,
    int _max_rows,
    int _hidden_size,
    int _num_heads,
    int _num_kv_heads,
    int _head_dim,
    int _max_seq_len,
    bool _has_residual,
    int _rope_style,
    half* _q_norm,
    half* _k_norm,
    half* _post_layernorm,
    half* _post_layernorm_bias,
    bool _residual_fp32,
    bool _use_graphs
):
    layernorm(_layernorm),
    layernorm_bias(_layernorm_bias),
    layernorm_is_rms(_layernorm_is_rms),
    norm_epsilon(_norm_epsilon),
    q_proj(_q_proj),
    k_proj(_k_proj),
    v_proj(_v_proj),
    o_proj(_o_proj),
    temp_state(_temp_state),
//     temp_q(_temp_q),
//     temp_k(_temp_k),
//     temp_v(_temp_v),
    temp_dq(_temp_dq),
    max_rows(_max_rows),
    hidden_size(_hidden_size),
    num_heads(_num_heads),
    num_kv_heads(_num_kv_heads),
    head_dim(_head_dim),
    max_seq_len(_max_seq_len),
    has_residual(_has_residual),
    rope_style(_rope_style),
    q_norm(_q_norm),
    k_norm(_k_norm),
    post_layernorm(_post_layernorm),
    post_layernorm_bias(_post_layernorm_bias),
    residual_fp32(_residual_fp32),
    use_graphs(_use_graphs)
{
}

QAttn::~QAttn()
{
    for (const auto& pair : graph_map) delete pair.second;
}

void QAttn::forward_cuda_1
(
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    void* x,
    int batch_size,
    int q_len,
    int past_len,
    int32_t* past_lens,
    half* temp_q,
    half* temp_k,
    half* temp_v,
    const half* sin,
    const half* cos,
    const std::vector<uintptr_t>& loras,
    half* lora_temp
)
{
    // Don't use graph if LoRAs enabled or if module might invoke cuBLAS

    if (!use_graphs || loras.size() || q_len * batch_size > MAX_Q_GEMM_ROWS)
    {
        forward_cuda_1_run(stream, cublas_handle, x, batch_size, q_len, past_len, past_lens, temp_q, temp_k, temp_v, sin, cos, loras, lora_temp);
        return;
    }

    QAttn_params_const pc = { q_len, batch_size };
    auto it = graph_map.find(pc);
    Graph* graph;
    if (it == graph_map.end())
    {
        graph = new Graph();
        graph_map[pc] = graph;
//        printf("**** new graph ****\n");
//        DBGI2(q_len, batch_size);
//        DBGX(x);
    }
    else graph = it->second;
    if (graph->count())
    {
        graph->begin_capture(stream);
        forward_cuda_1_run(stream, cublas_handle, x, batch_size, q_len, past_len, past_lens, temp_q, temp_k, temp_v, sin, cos, loras, lora_temp, graph);
        graph->end_capture(stream);
//        printf("**** record ****\n");
//        DBGI2(q_len, batch_size);
//        DBGX(x);
    }
    if (graph->ready())
    {
        if (layernorm)
        {
            if (layernorm_is_rms)
                rms_norm_cuda_update_x(graph, KernelLabels::NORM, x);
            else
                layer_norm_cuda_update_x(graph, KernelLabels::NORM, x);
        }
        else
        {
            q_gemm_cuda_update_a(graph, KernelLabels::Q, x);
            q_gemm_cuda_update_a(graph, KernelLabels::K, x);
            q_gemm_cuda_update_a(graph, KernelLabels::V, x);
        }

        q_gemm_cuda_update_c(graph, KernelLabels::Q, temp_q);
        q_gemm_cuda_update_c(graph, KernelLabels::K, temp_k);
        q_gemm_cuda_update_c(graph, KernelLabels::V, temp_v);

        if (q_norm)
        {
            head_norm_cuda_update_x(graph, KernelLabels::Q_NORM, temp_q);
            head_norm_cuda_update_y(graph, KernelLabels::Q_NORM, temp_q);
        }
        if (k_norm)
        {
            head_norm_cuda_update_x(graph, KernelLabels::K_NORM, temp_k);
            head_norm_cuda_update_y(graph, KernelLabels::K_NORM, temp_k);
        }

        if (rope_style != ROPE_STYLE_NONE)
        {
            rope_cuda_qk_update_q(graph, KernelLabels::ROPE, temp_q);
            rope_cuda_qk_update_k(graph, KernelLabels::ROPE, temp_k);
            rope_cuda_qk_update_past_len(graph, KernelLabels::ROPE, past_len);
            rope_cuda_qk_update_past_lens(graph, KernelLabels::ROPE, past_lens);
        }

        graph->launch(stream);
    }
    else
    {
        forward_cuda_1_run(stream, cublas_handle, x, batch_size, q_len, past_len, past_lens, temp_q, temp_k, temp_v, sin, cos, loras, lora_temp);
    }
}

void QAttn::forward_cuda_1_run
(
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    void* x,
    int batch_size,
    int q_len,
    int past_len,
    int32_t* past_lens,
    half* temp_q,
    half* temp_k,
    half* temp_v,
    const half* sin,
    const half* cos,
    const std::vector<uintptr_t>& loras,
    half* lora_temp,
    Graph* graph
)
{
    half* norm_state = (half*) x;

    if (layernorm)
    {
        if (layernorm_is_rms)
            rms_norm_cuda(stream, x, layernorm, temp_state, norm_epsilon, q_len * batch_size, hidden_size, false, residual_fp32, false, graph, KernelLabels::NORM);
        else
            layer_norm_cuda(stream, (half*)x, layernorm, layernorm_bias, temp_state, norm_epsilon, q_len * batch_size, hidden_size, false, graph, KernelLabels::NORM);
        norm_state = temp_state;
    }

    gemm_half_q_half_cuda(stream, cublas_handle, norm_state, q_proj, temp_q, q_len * batch_size, q_proj->width, hidden_size, true, temp_dq, false, NULL, 0, false, graph, KernelLabels::Q);
    gemm_half_q_half_cuda(stream, cublas_handle, norm_state, k_proj, temp_k, q_len * batch_size, k_proj->width, hidden_size, true, temp_dq, false, NULL, 0, false, graph, KernelLabels::K);
    gemm_half_q_half_cuda(stream, cublas_handle, norm_state, v_proj, temp_v, q_len * batch_size, v_proj->width, hidden_size, true, temp_dq, false, NULL, 0, false, graph, KernelLabels::V);

    apply_loras_cuda(stream, cublas_handle, q_proj_lora, loras, q_proj, norm_state, temp_q, lora_temp, q_len * batch_size);
    apply_loras_cuda(stream, cublas_handle, k_proj_lora, loras, k_proj, norm_state, temp_k, lora_temp, q_len * batch_size);
    apply_loras_cuda(stream, cublas_handle, v_proj_lora, loras, v_proj, norm_state, temp_v, lora_temp, q_len * batch_size);

    if (q_norm)
        head_norm_cuda(stream, temp_q, q_norm, NULL, temp_q, norm_epsilon, q_len * batch_size, num_heads, head_dim, graph, KernelLabels::Q_NORM);

    if (k_norm)
        head_norm_cuda(stream, temp_k, k_norm, NULL, temp_k, norm_epsilon, q_len * batch_size, num_kv_heads, head_dim, graph, KernelLabels::K_NORM);

//    rope_cuda(stream, temp_q, sin, cos, batch_size, q_len * num_heads,    head_dim, num_heads,    past_len, past_lens);
//    rope_cuda(stream, temp_k, sin, cos, batch_size, q_len * num_kv_heads, head_dim, num_kv_heads, past_len, past_lens);

    if (rope_style != ROPE_STYLE_NONE)
    {
        rope_cuda_qk
        (
            stream,
            temp_q,
            temp_k,
            sin,
            cos,
            batch_size,
            q_len * num_heads,
            q_len * num_kv_heads,
            head_dim,
            num_heads,
            num_kv_heads,
            past_len,
            past_lens,
            rope_style == ROPE_STYLE_NEOX,
            graph,
            KernelLabels::ROPE
        );
    }
}

void QAttn::forward_cuda_2
(
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    const half* attn_output,
    void* hidden_state,
    int q_len,
    int batch_size,
    const std::vector<uintptr_t>& loras,
    half* lora_temp
)
{
    if (!post_layernorm)
    {
        gemm_half_q_half_cuda(stream, cublas_handle, attn_output, o_proj, (half*) hidden_state, q_len * batch_size, o_proj->width, o_proj->height, !has_residual, temp_dq);
    }
    else
    {
        gemm_half_q_half_cuda(stream, cublas_handle, attn_output, o_proj, temp_state, q_len * batch_size, o_proj->width, o_proj->height, true, temp_dq);
        if (layernorm_is_rms)
            rms_norm_cuda(stream, temp_state, post_layernorm, hidden_state, norm_epsilon, q_len * batch_size, hidden_size, true, false, residual_fp32);
        else
            layer_norm_cuda(stream, temp_state, post_layernorm, post_layernorm_bias, (half*) hidden_state, norm_epsilon, q_len * batch_size, hidden_size, true);
    }

    apply_loras_cuda(stream, cublas_handle, o_proj_lora, loras, o_proj, attn_output, (half*) hidden_state, lora_temp, q_len * batch_size);
}
