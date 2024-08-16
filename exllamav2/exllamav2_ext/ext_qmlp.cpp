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
#include "cuda/rms_norm.cuh"
#include "cuda/q_gemm.cuh"

#include "cpp/util.h"
#include "ext_tp.h"
#include "ext_qmatrix.h"

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
    bool has_residual,
    torch::Tensor post_layernorm,
    torch::Tensor post_layernorm_bias,
    bool residual_fp32,
    bool use_graphs
)
{
    QMatrix* qm_gate = reinterpret_cast<QMatrix*> (q_gate);
    QMatrix* qm_up = reinterpret_cast<QMatrix*> (q_up);
    QMatrix* qm_down = reinterpret_cast<QMatrix*> (q_down);

    TORCH_CHECK_DTYPE_OPT(layernorm, kHalf);
    TORCH_CHECK_DTYPE_OPT(post_layernorm, kHalf);
    if (qm_gate && !layernorm.is_meta()) TORCH_CHECK(qm_gate->height == layernorm.size(0), "gate_proj is wrong shape")
    if (!layernorm.is_meta()) TORCH_CHECK(qm_up->height == layernorm.size(0), "up_proj is wrong shape")

    QMLP* mlp = new QMLP
    (
        layernorm.is_meta() ? NULL : (half*) layernorm.data_ptr(),
        layernorm_bias.is_meta() ? NULL : (half*) layernorm_bias.data_ptr(),
        layernorm_is_rms,
        norm_epsilon,
        qm_gate,
        qm_up,
        qm_down,
        temp_state.is_meta() ? NULL : (half*) temp_state.data_ptr(),
        temp_a.is_meta() ? NULL : (half*) temp_a.data_ptr(),
        temp_b.is_meta() ? NULL : (half*) temp_b.data_ptr(),
        temp_dq.is_meta() ? NULL : (half*) temp_dq.data_ptr(),
        max_rows,
        act_gelu,
        has_residual,
        post_layernorm.is_meta() ? NULL : (half*) post_layernorm.data_ptr(),
        post_layernorm_bias.is_meta() ? NULL : (half*) post_layernorm_bias.data_ptr(),
        residual_fp32,
        use_graphs
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
    if (mlp->residual_fp32) { TORCH_CHECK_DTYPE(x, kFloat); }
    else                    { TORCH_CHECK_DTYPE(x, kHalf); }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int dim = x.size(-1);
    int rows = x.numel() / dim;

    TORCH_CHECK(dim == mlp->up->height, "x is wrong shape");
    TORCH_CHECK(rows <= mlp->max_rows, "Too many rows in x");

    mlp->forward_
    (
        stream,
        at::cuda::getCurrentCUDABlasHandle(),
        (void*) x.data_ptr(),
        rows,
        dim,
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(x.size(1) == moe_mlp->hidden_dim, "x is wrong shape");
    TORCH_CHECK(x.size(0) <= moe_mlp->max_rows, "Too many rows in x");

    moe_mlp->forward_
    (
        stream,
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

void tp_mlp_forward_
(
    uintptr_t tp_context,
    torch::Tensor hidden_states,
    const std::vector<torch::Tensor> &temp_bc0_,
    const std::vector<torch::Tensor> &temp_bc1_,
    const std::vector<torch::Tensor> &temp_bc2_,
    const std::vector<torch::Tensor> &temp_gate_,
    const std::vector<torch::Tensor> &temp_up_,
    const std::vector<torch::Tensor> &temp_down_,
    const std::vector<torch::Tensor> &pre_layernorm,
    float norm_epsilon,
    const std::vector<uintptr_t> &gate,
    const std::vector<uintptr_t> &up,
    const std::vector<uintptr_t> &down,
    bool act_gelu
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    int rows = hidden_states.size(0);
    int interm_dim = temp_bc2_[0].size(1);
    int hidden_dim = temp_bc1_[0].size(1);

    std::vector<torch::Tensor> temp_bc0;
    std::vector<torch::Tensor> temp_bc1;
    std::vector<torch::Tensor> temp_bc2;
    std::vector<torch::Tensor> temp_gate;
    std::vector<torch::Tensor> temp_up;
    std::vector<torch::Tensor> temp_down;
    for (const auto &item : temp_bc0_) temp_bc0.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_bc1_) temp_bc1.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_bc2_) temp_bc2.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_gate_) temp_gate.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_up_) temp_up.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_down_) temp_down.push_back(item.narrow(0, 0, rows));

    auto run_thread = [&] (int t_device, Barrier* sync) -> void
    {
        #ifdef TP_MULTITHREADED
            at::InferenceMode guard(true);
        #endif

        // Broadcast

        tp_broadcast(tp_context, 0, hidden_states, BROADCAST_ID, temp_bc0, 1, t_device);

        // Layernorm

        for (int i = 0; i < pre_layernorm.size(); ++i)
        {
            int dev = temp_bc0[i].device().index();
            if (t_device != -1 && t_device != dev) continue;

            cudaSetDevice(dev);
            rms_norm_cuda
            (
                ctx->streams[dev],
                (void*) temp_bc0[i].data_ptr(),
                (half*) pre_layernorm[i].data_ptr(),
                (void*) temp_bc1[i].data_ptr(),
                norm_epsilon,
                rows,
                hidden_dim,
                false,
                false,  // TODO: FP32 residual
                false
            );
        }

        // Up, gate

        gemm_half_q_half_tp(temp_bc1, gate, temp_gate, false, tp_context, t_device);
        gemm_half_q_half_tp(temp_bc1, up, temp_up, false, tp_context, t_device);

        // Act/mul

        for (int i = 0; i < temp_bc1.size(); ++i)
        {
            int dev = temp_bc1[i].device().index();
            if (t_device != -1 && t_device != dev) continue;

            cudaSetDevice(dev);
            act_mul_cuda
            (
                ctx->streams[dev],
                (half*) temp_gate[i].data_ptr(),
                (half*) temp_up[i].data_ptr(),
                rows,
                temp_gate[i].size(1),
                act_gelu
            );
        }

        // Allgather

        tp_gather_barrier(tp_context, 1, temp_gate, BROADCAST_ID, temp_bc2, BROADCAST_ID, 1, t_device, sync);

        // Down

        gemm_half_q_half_tp(temp_bc2, down, temp_down, false, tp_context, t_device);

        // Add residual
        // TODO: libtorch adds a bit of overhead here that could be removed with a custom strided add_ kernel
        // TODO: Currently runs only on the first thread, seems libtorch in-place operations are not threadsafe?

        if (t_device == -1 || t_device == ctx->all_devices[0])
        {
            int offset = 0;
            for (int i = 0; i < temp_bc0.size(); ++i)
            {
                int dev = temp_bc0[i].device().index();
                cudaSetDevice(dev);

                auto stream = at::cuda::getStreamFromExternal(ctx->streams[dev], dev);
                at::cuda::setCurrentCUDAStream(stream);

                int w = temp_down[i].size(1);
                auto res_slice = temp_bc0[i].narrow(1, offset, w);
                temp_down[i].add_(res_slice);
                offset += w;
            }
        }

        if (t_device != -1)
            sync->arrive_and_wait();

        // Gather

        tp_gather_barrier(tp_context, 0, temp_down, BROADCAST_RS, temp_down, -1, 1, t_device, sync);

    };

    #ifdef TP_MULTITHREADED

        std::vector<std::future<void>> threads;
        Barrier sync_point(ctx->all_devices.size());
        for (const auto &dev : ctx->all_devices)
            threads.push_back(ctx->thread_pool->enqueue(run_thread, dev, &sync_point));
        for (auto &t : threads)
            t.get();

    #else

        run_thread(-1, NULL);

    #endif

}
