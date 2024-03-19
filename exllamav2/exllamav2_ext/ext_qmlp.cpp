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
#include "ext_qmlp.h"

#include "cuda/q_mlp.cuh"

#include "cpp/util.h"

uintptr_t make_q_mlp
(
    torch::Tensor layernorm,
    torch::Tensor layernorm_bias,
    bool layernorm_is_rms,
    float norm_epsilon,
    uintptr_t q_gate,
    uintptr_t q_up,
    uintptr_t q_down,
    torch::Tensor temp_state,
    torch::Tensor temp_a,
    torch::Tensor temp_b,
    torch::Tensor temp_dq,
    int max_rows,
    bool act_gelu,
    bool has_residual
)
{
    QMatrix* qm_gate = reinterpret_cast<QMatrix*> (q_gate);
    QMatrix* qm_up = reinterpret_cast<QMatrix*> (q_up);
    QMatrix* qm_down = reinterpret_cast<QMatrix*> (q_down);

    TORCH_CHECK_DTYPE_OPT(layernorm, kHalf);
    if (qm_gate && !layernorm.is_meta()) TORCH_CHECK(qm_gate->height == layernorm.size(0), "gate_proj is wrong shape")
    if (!layernorm.is_meta()) TORCH_CHECK(qm_up->height == layernorm.size(0), "up_proj is wrong shape")

    QMLP* mlp = new QMLP
    (
        (half*) layernorm.is_meta() ? NULL : (half*) layernorm.data_ptr(),
        (half*) layernorm_bias.is_meta() ? NULL : (half*) layernorm_bias.data_ptr(),
        layernorm_is_rms,
        norm_epsilon,
        qm_gate,
        qm_up,
        qm_down,
        (half*) temp_state.data_ptr(),
        (half*) temp_a.data_ptr(),
        (half*) temp_b.data_ptr(),
        (half*) temp_dq.data_ptr(),
        max_rows,
        act_gelu,
        has_residual
    );

    return reinterpret_cast<uintptr_t> (mlp);
}

void free_q_mlp
(
   uintptr_t handle
)
{
    QMLP* mlp = reinterpret_cast<QMLP*> (handle);
    delete mlp;
}

void q_mlp_forward_
(
    uintptr_t q_mlp,
    torch::Tensor x,
    const std::vector<uintptr_t>& loras,
    torch::Tensor loras_temp
)
{
    QMLP* mlp = reinterpret_cast<QMLP*> (q_mlp);
    TORCH_CHECK_DTYPE(x, kHalf);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    TORCH_CHECK(x.size(1) == mlp->up->height, "x is wrong shape");
    TORCH_CHECK(x.size(0) <= mlp->max_rows, "Too many rows in x");

    mlp->forward_
    (
        at::cuda::getCurrentCUDABlasHandle(),
        (half*) x.data_ptr(),
        x.size(0), // rows
        x.size(1), // columns == hidden_size
        loras,
        loras_temp.device().is_meta() ? NULL : (half*) loras_temp.data_ptr()
    );
}

int q_mlp_set_loras
(
    uintptr_t q_mlp,
    std::unordered_map<uintptr_t, torch::Tensor>& gate_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& gate_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& up_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& up_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& down_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& down_proj_lora_b
)
{
    QMLP* mlp = reinterpret_cast<QMLP*> (q_mlp);

    mlp->gate_proj_lora.clear();
    mlp->up_proj_lora.clear();
    mlp->down_proj_lora.clear();

    int max_rank = 0;

    for (const auto& pair : gate_proj_lora_a)
    {
        int rank = pair.second.size(-1);
        if (rank > max_rank) max_rank = rank;
        half* a = (half*) pair.second.data_ptr();
        half* b = (half*) gate_proj_lora_b[pair.first].data_ptr();
        mlp->gate_proj_lora[pair.first] = std::make_tuple(a, b, rank);
    }

    for (const auto& pair : up_proj_lora_a)
    {
        int rank = pair.second.size(-1);
        if (rank > max_rank) max_rank = rank;
        half* a = (half*) pair.second.data_ptr();
        half* b = (half*) up_proj_lora_b[pair.first].data_ptr();
        mlp->up_proj_lora[pair.first] = std::make_tuple(a, b, rank);
    }

    for (const auto& pair : down_proj_lora_a)
    {
        int rank = pair.second.size(-1);
        if (rank > max_rank) max_rank = rank;
        half* a = (half*) pair.second.data_ptr();
        half* b = (half*) down_proj_lora_b[pair.first].data_ptr();
        mlp->down_proj_lora[pair.first] = std::make_tuple(a, b, rank);
    }

    return max_rank;
}

// Quant MoE MLP

uintptr_t make_q_moe_mlp
(
    torch::Tensor layernorm,
    torch::Tensor layernorm_bias,
    bool layernorm_is_rms,
    float norm_epsilon,
    torch::Tensor gate,
    int num_experts,
    int num_experts_per_token,
    const std::vector<uintptr_t>& w1,
    const std::vector<uintptr_t>& w2,
    const std::vector<uintptr_t>& w3,
    torch::Tensor temp_state,
    torch::Tensor temp_gathered_state,
    torch::Tensor temp_a,
    torch::Tensor temp_b,
    torch::Tensor temp_logits,
    torch::Tensor temp_dq,
    int max_rows,
    bool act_gelu
)
{
    std::vector<QMatrix*> qm_w1;
    std::vector<QMatrix*> qm_w2;
    std::vector<QMatrix*> qm_w3;

    for (int i = 0; i < (int)w1.size(); ++i)
    {
        qm_w1.push_back(reinterpret_cast<QMatrix*> (w1[i]));
        qm_w2.push_back(reinterpret_cast<QMatrix*> (w2[i]));
        qm_w3.push_back(reinterpret_cast<QMatrix*> (w3[i]));
    }

    TORCH_CHECK_DTYPE(layernorm, kHalf);
    TORCH_CHECK_SHAPES(layernorm, 0, gate, 1, 1);  // gate is transposed
    TORCH_CHECK(gate.size(0) == num_experts, "gate output features != num_experts");

    int hidden_dim = gate.size(1);

    QMoEMLP* moe_mlp = new QMoEMLP
    (
        (half*) layernorm.is_meta() ? NULL : (half*) layernorm.data_ptr(),
        (half*) layernorm_bias.is_meta() ? NULL : (half*) layernorm_bias.data_ptr(),
        layernorm_is_rms,
        norm_epsilon,
        (half*) gate.data_ptr(),
        num_experts,
        num_experts_per_token,
        qm_w1,
        qm_w2,
        qm_w3,
        (half*) temp_state.data_ptr(),
        (half*) temp_gathered_state.data_ptr(),
        (half*) temp_a.data_ptr(),
        (half*) temp_b.data_ptr(),
        (half*) temp_logits.data_ptr(),
        (half*) temp_dq.data_ptr(),
        max_rows,
        hidden_dim,
        act_gelu
    );

    return reinterpret_cast<uintptr_t> (moe_mlp);
}

void free_q_moe_mlp
(
   uintptr_t handle
)
{
    QMoEMLP* moe_mlp = reinterpret_cast<QMoEMLP*> (handle);
    delete moe_mlp;
}

void q_moe_mlp_forward_
(
    uintptr_t q_moe_mlp,
    torch::Tensor x
//    const std::vector<uintptr_t>& loras,
//    torch::Tensor loras_temp
)
{
    QMoEMLP* moe_mlp = reinterpret_cast<QMoEMLP*> (q_moe_mlp);
    TORCH_CHECK_DTYPE(x, kHalf);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    TORCH_CHECK(x.size(1) == moe_mlp->hidden_dim, "x is wrong shape");
    TORCH_CHECK(x.size(0) <= moe_mlp->max_rows, "Too many rows in x");

    moe_mlp->forward_
    (
        at::cuda::getCurrentCUDABlasHandle(),
        (half*) x.data_ptr(),
        x.size(0), // rows
        x.size(1) // columns == hidden_size
//        loras,
//        loras_temp.device().is_meta() ? NULL : (half*) loras_temp.data_ptr()
    );
}

//int q_moe_mlp_set_loras
//(
//    uintptr_t q_moe_mlp,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w1_lora_a,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w1_lora_b,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w2_lora_a,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w2_lora_b,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w3_lora_a,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w3_lora_b
//)
//{
//    QMoEMLP* moe_mlp = reinterpret_cast<QMoEMLP*> (q_moe_mlp);
//
//    int max_rank = 0;
//
//    for (int i = 0; i < moe_mlp->num_experts; ++i)
//    {
//        moe_mlp->w1_lora[i].clear();
//        moe_mlp->w2_lora[i].clear();
//        moe_mlp->w3_lora[i].clear();
//
//        for (const auto& pair : w1_lora_a[i])
//        {
//            int rank = pair.second.size(-1);
//            if (rank > max_rank) max_rank = rank;
//            half* a = (half*) pair.second.data_ptr();
//            half* b = (half*) w1_lora_b[i][pair.first].data_ptr();
//            moe_mlp->w1_lora[i][pair.first] = std::make_tuple(a, b, rank);
//        }
//
//        for (const auto& pair : w2_lora_a[i])
//        {
//            int rank = pair.second.size(-1);
//            if (rank > max_rank) max_rank = rank;
//            half* a = (half*) pair.second.data_ptr();
//            half* b = (half*) w2_lora_b[i][pair.first].data_ptr();
//            moe_mlp->w2_lora[i][pair.first] = std::make_tuple(a, b, rank);
//        }
//
//        for (const auto& pair : w3_lora_a[i])
//        {
//            int rank = pair.second.size(-1);
//            if (rank > max_rank) max_rank = rank;
//            half* a = (half*) pair.second.data_ptr();
//            half* b = (half*) w3_lora_b[i][pair.first].data_ptr();
//            moe_mlp->w3_lora[i][pair.first] = std::make_tuple(a, b, rank);
//        }
//    }
//
//    return max_rank;
//}
