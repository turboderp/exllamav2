#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include "config.h"
#include "ext_qattn.h"

#include "cuda/q_attn.cuh"
#include "cuda/rms_norm.cuh"
#include "cuda/q_gemm.cuh"
#include "cuda/rope.cuh"

#include "cpp/util.h"
#include "ext_tp.h"
#include "ext_qmatrix.h"

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
    int rope_style,
    torch::Tensor q_norm,
    torch::Tensor k_norm,
    torch::Tensor post_layernorm,
    torch::Tensor post_layernorm_bias,
    bool residual_fp32,
    bool use_graphs
)
{
    QMatrix* qm_q_proj = reinterpret_cast<QMatrix*> (q_q_proj);
    QMatrix* qm_k_proj = reinterpret_cast<QMatrix*> (q_k_proj);
    QMatrix* qm_v_proj = reinterpret_cast<QMatrix*> (q_v_proj);
    QMatrix* qm_o_proj = reinterpret_cast<QMatrix*> (q_o_proj);

    TORCH_CHECK_DTYPE_OPT(layernorm, kHalf);
    TORCH_CHECK_DTYPE_OPT(post_layernorm, kHalf);

    if (qm_q_proj && !layernorm.is_meta()) TORCH_CHECK(qm_q_proj->height == layernorm.size(0), "q_proj is wrong shape")
    if (qm_k_proj && !layernorm.is_meta()) TORCH_CHECK(qm_k_proj->height == layernorm.size(0), "k_proj is wrong shape")
    if (qm_v_proj && !layernorm.is_meta()) TORCH_CHECK(qm_v_proj->height == layernorm.size(0), "v_proj is wrong shape")
    if (!layernorm.is_meta()) TORCH_CHECK(qm_o_proj->width == layernorm.size(0), "o_proj is wrong shape")

    QAttn* attn = new QAttn
    (
        layernorm.is_meta() ? NULL : (half*) layernorm.data_ptr(),
        layernorm_bias.is_meta() ? NULL : (half*) layernorm_bias.data_ptr(),
        layernorm_is_rms,
        norm_epsilon,
        qm_q_proj,
        qm_k_proj,
        qm_v_proj,
        qm_o_proj,
        temp_state.is_meta() ? NULL : (half*) temp_state.data_ptr(),
//        (half*) temp_q.data_ptr(),
//        (half*) temp_k.data_ptr(),
//        (half*) temp_v.data_ptr(),
        temp_dq.is_meta() ? NULL : (half*) temp_dq.data_ptr(),
        max_rows,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len,
        has_residual,
        rope_style,
        q_norm.is_meta() ? NULL : (half*) q_norm.data_ptr(),
        k_norm.is_meta() ? NULL : (half*) k_norm.data_ptr(),
        post_layernorm.is_meta() ? NULL : (half*) post_layernorm.data_ptr(),
        post_layernorm_bias.is_meta() ? NULL : (half*) post_layernorm_bias.data_ptr(),
        residual_fp32,
        use_graphs
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
    if (attn->residual_fp32) { TORCH_CHECK_DTYPE(x, kFloat); }
    else                     { TORCH_CHECK_DTYPE(x, kHalf); }

    TORCH_CHECK_DTYPE_OPT(past_lens, kInt);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();

    attn->forward_cuda_1
    (
        stream,
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
    if (attn->residual_fp32) { TORCH_CHECK_DTYPE(x, kFloat); }
    else                     { TORCH_CHECK_DTYPE(x, kHalf); }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();

    attn->forward_cuda_2
    (
        stream,
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

//MHAFwdKVCacheFunc flash_attn_func = nullptr;
//py::object fwd_kvcache_func;
void set_flash_attn_func()
{
//    fwd_kvcache_func = py::module_::import("flash_attn_2_cuda").attr("fwd_kvcache");
}

void tp_attn_forward_paged_
(
    uintptr_t tp_context,
    torch::Tensor hidden_states,
    const std::vector<torch::Tensor> &temp_bc0_,
    const std::vector<torch::Tensor> &temp_bc1_,
    const std::vector<torch::Tensor> &temp_bc2_,
    const std::vector<torch::Tensor> &temp_q_,
    const std::vector<torch::Tensor> &temp_k_,
    const std::vector<torch::Tensor> &temp_v_,
    const std::vector<torch::Tensor> &temp_o_,
    const std::vector<torch::Tensor> &k_cache,
    const std::vector<torch::Tensor> &v_cache,
    const std::vector<torch::Tensor> &pre_layernorm,
    float norm_epsilon,
    const std::vector<uintptr_t> &q_proj,
    const std::vector<uintptr_t> &k_proj,
    const std::vector<uintptr_t> &v_proj,
    const std::vector<uintptr_t> &o_proj,
    int head_dim,
    int rope_style,
    int batch_size,
    int q_len,
    const std::vector<torch::Tensor> &sin,
    const std::vector<torch::Tensor> &cos,
    const std::vector<torch::Tensor> &past_lens,
    const std::vector<torch::Tensor> &block_index,
    float scaling
)
{
    auto fwd_kvcache_func = py::module_::import("flash_attn_2_cuda").attr("fwd_kvcache");

    #ifdef TP_MULTITHREADED
        pybind11::gil_scoped_release release;
    #endif

    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    int rows = batch_size * q_len;
    int hidden_dim = temp_bc0_[0].size(1);

    std::vector<torch::Tensor> temp_bc0;
    std::vector<torch::Tensor> temp_bc1;
    std::vector<torch::Tensor> temp_bc2;
    std::vector<torch::Tensor> temp_q;
    std::vector<torch::Tensor> temp_k;
    std::vector<torch::Tensor> temp_v;
    std::vector<torch::Tensor> temp_o;
    for (const auto &item : temp_bc0_) temp_bc0.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_bc1_) temp_bc1.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_bc2_) temp_bc2.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_q_) temp_q.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_k_) temp_k.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_v_) temp_v.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_o_) temp_o.push_back(item.narrow(0, 0, rows));

    auto run_thread = [&] (int t_device, Barrier* sync) -> void
    {
        #ifdef TP_MULTITHREADED
            at::InferenceMode guard(true);
        #endif

        // Broadcast

        tp_broadcast(tp_context, 0, hidden_states, BROADCAST_Q, temp_bc0, head_dim, t_device);

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

        // Q, K, V

        gemm_half_q_half_tp(temp_bc1, q_proj, temp_q, false, tp_context, t_device);
        gemm_half_q_half_tp(temp_bc1, k_proj, temp_k, false, tp_context, t_device);
        gemm_half_q_half_tp(temp_bc1, v_proj, temp_v, false, tp_context, t_device);

        // RoPE

        if (rope_style != ROPE_STYLE_NONE)
        {
            for (int i = 0; i < temp_q.size(); ++i)
            {
                int dev = temp_q[i].device().index();
                if (t_device != -1 && t_device != dev) continue;
                cudaSetDevice(dev);

                int num_heads = temp_q[i].size(1) / head_dim;
                int num_kv_heads = temp_k[i].size(1) / head_dim;
                int q_len = rows / batch_size;

                rope_cuda_qk
                (
                    ctx->streams[dev],
                    (half*) temp_q[i].data_ptr(),
                    (half*) temp_k[i].data_ptr(),
                    (half*) sin[dev].data_ptr(),
                    (half*) cos[dev].data_ptr(),
                    batch_size,
                    q_len * num_heads,
                    q_len * num_kv_heads,
                    head_dim,
                    num_heads,
                    num_kv_heads,
                    0, //past_len,
                    (int32_t*) past_lens[i].data_ptr(),
                    rope_style == ROPE_STYLE_NEOX
                );
            }
        }

        // Attn

        for (int i = 0; i < temp_q.size(); ++i)
        {
            int dev = temp_q[i].device().index();
            if (t_device != -1 && t_device != dev) continue;
            cudaSetDevice(dev);

            auto stream = at::cuda::getStreamFromExternal(ctx->streams[dev], dev);
            at::cuda::setCurrentCUDAStream(stream);

            std::vector<int64_t> attn_shape_qo = {batch_size, q_len, temp_q[i].size(1) / head_dim, head_dim};
            std::vector<int64_t> attn_shape_kv = {batch_size, q_len, temp_k[i].size(1) / head_dim, head_dim};

            torch::Tensor q = temp_q[i].view(attn_shape_qo);
            torch::Tensor k = temp_k[i].view(attn_shape_kv);
            torch::Tensor v = temp_v[i].view(attn_shape_kv);
            torch::Tensor o = temp_o[i].view(attn_shape_qo);

            {
                #ifdef TP_MULTITHREADED
                    pybind11::gil_scoped_acquire acquire;
                #endif

                auto none = py::none();

                fwd_kvcache_func
                (
                    q,
                    k_cache[i],
                    v_cache[i],
                    k,
                    v,
                    past_lens[i],  // cache_seqlens
                    none,  // rotary_cos
                    none,  // rotary_sin
                    none,  // cache_batch_idx
                    none,  // cache_leftpad
                    block_index[i],  // block_table
                    none,  // alibi_slopes
                    o, // output
                    scaling,  // softmax_scale
                    true,  // causal
                    10000000,  // window_size[0]
                    -1,  // window_size[1]
                    0.0,  // softcap
                    true,  // rotary_interleaved
                    0  // num_splits
                );
            }

        }

        // Allgather

        tp_gather_barrier(tp_context, 1, temp_o, BROADCAST_Q, temp_bc2, BROADCAST_Q, head_dim, t_device, sync);

        // Output projection

        gemm_half_q_half_tp(temp_bc2, o_proj, temp_o, false, tp_context, t_device);

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

                int w = temp_o[i].size(1);
                auto res_slice = temp_bc0[i].narrow(1, offset, w);
                temp_o[i].add_(res_slice);
                offset += w;
            }
        }

        if (t_device != -1)
            sync->arrive_and_wait();

        // Gather

        tp_gather_barrier(tp_context, 0, temp_o, BROADCAST_Q, temp_o, -1, head_dim, t_device, sync);

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

void tp_attn_forward_
(
    uintptr_t tp_context,
    torch::Tensor hidden_states,
    const std::vector<torch::Tensor> &temp_bc0_,
    const std::vector<torch::Tensor> &temp_bc1_,
    const std::vector<torch::Tensor> &temp_bc2_,
    const std::vector<torch::Tensor> &temp_q_,
    const std::vector<torch::Tensor> &temp_k_,
    const std::vector<torch::Tensor> &temp_v_,
    const std::vector<torch::Tensor> &temp_o_,
    const std::vector<torch::Tensor> &k_cache,
    const std::vector<torch::Tensor> &v_cache,
    const std::vector<torch::Tensor> &pre_layernorm,
    float norm_epsilon,
    const std::vector<uintptr_t> &q_proj,
    const std::vector<uintptr_t> &k_proj,
    const std::vector<uintptr_t> &v_proj,
    const std::vector<uintptr_t> &o_proj,
    int head_dim,
    int rope_style,
    int batch_size,
    int q_len,
    const std::vector<torch::Tensor> &sin,
    const std::vector<torch::Tensor> &cos,
    const std::vector<torch::Tensor> &past_len_tp,
    float scaling
)
{
    auto fwd_kvcache_func = py::module_::import("flash_attn_2_cuda").attr("fwd_kvcache");

    #ifdef TP_MULTITHREADED
        pybind11::gil_scoped_release release;
    #endif

    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    int rows = batch_size * q_len;
    int hidden_dim = temp_bc0_[0].size(1);

    std::vector<torch::Tensor> temp_bc0;
    std::vector<torch::Tensor> temp_bc1;
    std::vector<torch::Tensor> temp_bc2;
    std::vector<torch::Tensor> temp_q;
    std::vector<torch::Tensor> temp_k;
    std::vector<torch::Tensor> temp_v;
    std::vector<torch::Tensor> temp_o;
    for (const auto &item : temp_bc0_) temp_bc0.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_bc1_) temp_bc1.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_bc2_) temp_bc2.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_q_) temp_q.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_k_) temp_k.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_v_) temp_v.push_back(item.narrow(0, 0, rows));
    for (const auto &item : temp_o_) temp_o.push_back(item.narrow(0, 0, rows));

    auto run_thread = [&] (int t_device, Barrier* sync) -> void
    {
        #ifdef TP_MULTITHREADED
            at::InferenceMode guard(true);
        #endif

        // Broadcast

        tp_broadcast(tp_context, 0, hidden_states, BROADCAST_Q, temp_bc0, head_dim, t_device);

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

        // Q, K, V

        gemm_half_q_half_tp(temp_bc1, q_proj, temp_q, false, tp_context, t_device);
        gemm_half_q_half_tp(temp_bc1, k_proj, temp_k, false, tp_context, t_device);
        gemm_half_q_half_tp(temp_bc1, v_proj, temp_v, false, tp_context, t_device);

        // RoPE

        if (rope_style != ROPE_STYLE_NONE)
        {
            for (int i = 0; i < temp_q.size(); ++i)
            {
                int dev = temp_q[i].device().index();
                if (t_device != -1 && t_device != dev) continue;
                cudaSetDevice(dev);

                int num_heads = temp_q[i].size(1) / head_dim;
                int num_kv_heads = temp_k[i].size(1) / head_dim;
                int q_len = rows / batch_size;

                rope_cuda_qk
                (
                    ctx->streams[dev],
                    (half*) temp_q[i].data_ptr(),
                    (half*) temp_k[i].data_ptr(),
                    (half*) sin[dev].data_ptr(),
                    (half*) cos[dev].data_ptr(),
                    batch_size,
                    q_len * num_heads,
                    q_len * num_kv_heads,
                    head_dim,
                    num_heads,
                    num_kv_heads,
                    0, //past_len,
                    (int32_t*) past_len_tp[i].data_ptr(),
                    rope_style == ROPE_STYLE_NEOX
                );
            }
        }

        // Attn

        for (int i = 0; i < temp_q.size(); ++i)
        {
            int dev = temp_q[i].device().index();
            if (t_device != -1 && t_device != dev) continue;
            cudaSetDevice(dev);

            auto stream = at::cuda::getStreamFromExternal(ctx->streams[dev], dev);
            at::cuda::setCurrentCUDAStream(stream);

            std::vector<int64_t> attn_shape_qo = {batch_size, q_len, temp_q[i].size(1) / head_dim, head_dim};
            std::vector<int64_t> attn_shape_kv = {batch_size, q_len, temp_k[i].size(1) / head_dim, head_dim};

            torch::Tensor q = temp_q[i].view(attn_shape_qo);
            torch::Tensor k = temp_k[i].view(attn_shape_kv);
            torch::Tensor v = temp_v[i].view(attn_shape_kv);
            torch::Tensor o = temp_o[i].view(attn_shape_qo);

            {
                #ifdef TP_MULTITHREADED
                    pybind11::gil_scoped_acquire acquire;
                #endif

                auto none = py::none();

                fwd_kvcache_func
                (
                    q,
                    k_cache[i],
                    v_cache[i],
                    k,
                    v,
                    past_len_tp[i],  // cache_seqlens
                    none,  // rotary_cos
                    none,  // rotary_sin
                    none,  // cache_batch_idx
                    none,  // cache_leftpad
                    none,  // block_table
                    none,  // alibi_slopes
                    o, // output
                    scaling,  // softmax_scale
                    true,  // causal
                    10000000,  // window_size[0]
                    -1,  // window_size[1]
                    0.0,  // softcap
                    true,  // rotary_interleaved
                    0  // num_splits
                );
            }

        }

        // Allgather

        tp_gather_barrier(tp_context, 1, temp_o, BROADCAST_Q, temp_bc2, BROADCAST_Q, head_dim, t_device, sync);

        // Output projection

        gemm_half_q_half_tp(temp_bc2, o_proj, temp_o, false, tp_context, t_device);

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

                int w = temp_o[i].size(1);
                auto res_slice = temp_bc0[i].narrow(1, offset, w);
                temp_o[i].add_(res_slice);
                offset += w;
            }
        }

        if (t_device != -1)
            sync->arrive_and_wait();

        // Gather

        tp_gather_barrier(tp_context, 0, temp_o, BROADCAST_Q, temp_o, -1, head_dim, t_device, sync);

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
