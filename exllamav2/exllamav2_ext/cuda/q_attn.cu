#include "q_attn.cuh"
#include "q_gemm.cuh"
#include "rms_norm.cuh"
#include "layer_norm.cuh"
#include "head_norm.cuh"
#include "rope.cuh"
#include "util.cuh"
#include "lora.cuh"

const int THREADS_X = 32;
const int THREADS_Y = 1;
const int THREADS_Z = 4;
const int BLOCKSIZE_X = 2; // 2*half == 1*uint32_t
const int BLOCKSIZE_Z = 4; // num_heads must be divisible by BLOCKSIZE_Z

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
    bool _neox_style,
    half* _q_norm,
    half* _k_norm
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
    neox_style(_neox_style),
    q_norm(_q_norm),
    k_norm(_k_norm)
{
}

QAttn::~QAttn()
{
}

void QAttn::forward_cuda_1
(
    cublasHandle_t cublas_handle,
    half* x,
    int batch_size,
    int q_len,
    int past_len,
    const int32_t* past_lens,
    half* temp_q,
    half* temp_k,
    half* temp_v,
    const half* sin,
    const half* cos,
    const std::vector<uintptr_t>& loras,
    half* lora_temp
)
{
    half* norm_state = x;

    if (layernorm)
    {
        if (layernorm_is_rms)
            rms_norm_cuda(x, layernorm, temp_state, norm_epsilon, q_len * batch_size, hidden_size);
        else
            layer_norm_cuda(x, layernorm, layernorm_bias, temp_state, norm_epsilon, q_len * batch_size, hidden_size);
        norm_state = temp_state;
    }

    gemm_half_q_half_cuda(cublas_handle, norm_state, q_proj, temp_q, q_len * batch_size, q_proj->width, hidden_size, true, temp_dq);
    gemm_half_q_half_cuda(cublas_handle, norm_state, k_proj, temp_k, q_len * batch_size, k_proj->width, hidden_size, true, temp_dq);
    gemm_half_q_half_cuda(cublas_handle, norm_state, v_proj, temp_v, q_len * batch_size, v_proj->width, hidden_size, true, temp_dq);

    if (q_norm)
        head_norm_cuda(temp_q, q_norm, NULL, temp_q, norm_epsilon, q_len * batch_size, num_heads, head_dim);

    if (k_norm)
        head_norm_cuda(temp_k, k_norm, NULL, temp_k, norm_epsilon, q_len * batch_size, num_kv_heads, head_dim);

    apply_loras_cuda(cublas_handle, q_proj_lora, loras, q_proj, norm_state, temp_q, lora_temp, q_len * batch_size);
    apply_loras_cuda(cublas_handle, k_proj_lora, loras, k_proj, norm_state, temp_k, lora_temp, q_len * batch_size);
    apply_loras_cuda(cublas_handle, v_proj_lora, loras, v_proj, norm_state, temp_v, lora_temp, q_len * batch_size);

//    rope_cuda(temp_q, sin, cos, batch_size, q_len * num_heads,    head_dim, num_heads,    past_len, past_lens);
//    rope_cuda(temp_k, sin, cos, batch_size, q_len * num_kv_heads, head_dim, num_kv_heads, past_len, past_lens);
    rope_cuda_qk
    (
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
        neox_style
    );
}

void QAttn::forward_cuda_2
(
    cublasHandle_t cublas_handle,
    const half* attn_output,
    half* hidden_state,
    int q_len,
    int batch_size,
    const std::vector<uintptr_t>& loras,
    half* lora_temp
)
{
    gemm_half_q_half_cuda(cublas_handle, attn_output, o_proj, hidden_state, q_len * batch_size, o_proj->width, o_proj->height, !has_residual, temp_dq);

    apply_loras_cuda(cublas_handle, o_proj_lora, loras, o_proj, attn_output, hidden_state, lora_temp, q_len * batch_size);
}
