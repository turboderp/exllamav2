#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"
#include "ext_qattn.h"

#include "cuda/q_attn.cuh"

#include "cpp/util.h"

uintptr_t make_q_attn
(
    torch::Tensor layernorm,
    torch::Tensor layernorm_bias,
    bool layernorm_is_rms,
    float norm_epsilon,
    uintptr_t q_q_proj,
    uintptr_t q_k_proj,
    uintptr_t q_v_proj,
    uintptr_t q_o_proj,
    torch::Tensor temp_state,
//    torch::Tensor temp_q,
//    torch::Tensor temp_k,
//    torch::Tensor temp_v,
    torch::Tensor temp_dq,
    int max_rows,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    bool has_residual,
    bool neox_style,
    torch::Tensor q_norm,
    torch::Tensor k_norm
)
{
    QMatrix* qm_q_proj = reinterpret_cast<QMatrix*> (q_q_proj);
    QMatrix* qm_k_proj = reinterpret_cast<QMatrix*> (q_k_proj);
    QMatrix* qm_v_proj = reinterpret_cast<QMatrix*> (q_v_proj);
    QMatrix* qm_o_proj = reinterpret_cast<QMatrix*> (q_o_proj);

    TORCH_CHECK_DTYPE_OPT(layernorm, kHalf);

    if (qm_q_proj && !layernorm.is_meta()) TORCH_CHECK(qm_q_proj->height == layernorm.size(0), "q_proj is wrong shape")
    if (qm_k_proj && !layernorm.is_meta()) TORCH_CHECK(qm_k_proj->height == layernorm.size(0), "k_proj is wrong shape")
    if (qm_v_proj && !layernorm.is_meta()) TORCH_CHECK(qm_v_proj->height == layernorm.size(0), "v_proj is wrong shape")
    if (!layernorm.is_meta()) TORCH_CHECK(qm_o_proj->width == layernorm.size(0), "o_proj is wrong shape")

    QAttn* attn = new QAttn
    (
        (half*) layernorm.is_meta() ? NULL : (half*) layernorm.data_ptr(),
        (half*) layernorm_bias.is_meta() ? NULL : (half*) layernorm_bias.data_ptr(),
        layernorm_is_rms,
        norm_epsilon,
        qm_q_proj,
        qm_k_proj,
        qm_v_proj,
        qm_o_proj,
        (half*) temp_state.data_ptr(),
//        (half*) temp_q.data_ptr(),
//        (half*) temp_k.data_ptr(),
//        (half*) temp_v.data_ptr(),
        (half*) temp_dq.data_ptr(),
        max_rows,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len,
        has_residual,
        neox_style,
        (half*) q_norm.is_meta() ? NULL : (half*) q_norm.data_ptr(),
        (half*) k_norm.is_meta() ? NULL : (half*) k_norm.data_ptr()
    );

    return reinterpret_cast<uintptr_t> (attn);
}

void free_q_attn
(
    uintptr_t handle
)
{
    QAttn* attn = reinterpret_cast<QAttn*> (handle);
    delete attn;
}

void q_attn_forward_1
(
    uintptr_t q_attn,
    torch::Tensor x,
    int batch_size,
    int q_len,
    int past_len,
    torch::Tensor past_lens,
    torch::Tensor q_temp,
    torch::Tensor k_temp,
    torch::Tensor v_temp,
    torch::Tensor sin,
    torch::Tensor cos,
    const std::vector<uintptr_t>& loras,
    torch::Tensor loras_temp
)
{
    QAttn* attn = reinterpret_cast<QAttn*> (q_attn);
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE_OPT(past_lens, kInt);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();

    attn->forward_cuda_1
    (
        cublas_handle,
        (half*) x.data_ptr(),
        batch_size,
        q_len,
        past_len,
        past_lens.device().is_meta() ? NULL : (int32_t*) past_lens.data_ptr(),
        (half*) q_temp.data_ptr(),
        (half*) k_temp.data_ptr(),
        (half*) v_temp.data_ptr(),
        (half*) sin.data_ptr(),
        (half*) cos.data_ptr(),
        loras,
        loras_temp.device().is_meta() ? NULL : (half*) loras_temp.data_ptr()
    );
}

void q_attn_forward_2
(
    uintptr_t q_attn,
    torch::Tensor x,
    torch::Tensor attn_output,
    int batch_size,
    int q_len,
    const std::vector<uintptr_t>& loras,
    torch::Tensor loras_temp
)
{
    QAttn* attn = reinterpret_cast<QAttn*> (q_attn);
    TORCH_CHECK_DTYPE(x, kHalf);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();

    attn->forward_cuda_2
    (
        cublas_handle,
        (const half*) attn_output.data_ptr(),
        (half*) x.data_ptr(),
        q_len,
        batch_size,
        loras,
        loras_temp.device().is_meta() ? NULL : (half*) loras_temp.data_ptr()
    );
}

int q_attn_set_loras
(
    uintptr_t q_attn,
    std::unordered_map<uintptr_t, torch::Tensor>& q_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& q_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& k_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& k_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& v_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& v_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& o_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& o_proj_lora_b
)
{
    QAttn* attn = reinterpret_cast<QAttn*> (q_attn);

    attn->q_proj_lora.clear();
    attn->k_proj_lora.clear();
    attn->v_proj_lora.clear();
    attn->o_proj_lora.clear();

    int max_rank = 0;

    for (const auto& pair : q_proj_lora_a)
    {
        int rank = pair.second.size(-1);
        if (rank > max_rank) max_rank = rank;
        half* a = (half*) pair.second.data_ptr();
        half* b = (half*) q_proj_lora_b[pair.first].data_ptr();
        attn->q_proj_lora[pair.first] = std::make_tuple(a, b, rank);
    }

    for (const auto& pair : k_proj_lora_a)
    {
        int rank = pair.second.size(-1);
        if (rank > max_rank) max_rank = rank;
        half* a = (half*) pair.second.data_ptr();
        half* b = (half*) k_proj_lora_b[pair.first].data_ptr();
        attn->k_proj_lora[pair.first] = std::make_tuple(a, b, rank);
    }

    for (const auto& pair : v_proj_lora_a)
    {
        int rank = pair.second.size(-1);
        if (rank > max_rank) max_rank = rank;
        half* a = (half*) pair.second.data_ptr();
        half* b = (half*) v_proj_lora_b[pair.first].data_ptr();
        attn->v_proj_lora[pair.first] = std::make_tuple(a, b, rank);
    }

    for (const auto& pair : o_proj_lora_a)
    {
        int rank = pair.second.size(-1);
        if (rank > max_rank) max_rank = rank;
        half* a = (half*) pair.second.data_ptr();
        half* b = (half*) o_proj_lora_b[pair.first].data_ptr();
        attn->o_proj_lora[pair.first] = std::make_tuple(a, b, rank);
    }

    return max_rank;
}
