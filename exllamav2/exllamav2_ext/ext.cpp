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

#include "cuda/pack_tensor.cuh"
#include "cuda/quantize.cuh"
#include "cuda/q_matrix.cuh"
#include "cuda/q_attn.cuh"
#include "cuda/q_mlp.cuh"
#include "cuda/q_gemm.cuh"
#include "cuda/rms_norm.cuh"
#include "cuda/rope.cuh"
#include "cuda/cache.cuh"

#include "cpp/quantize_func.h"
#include "cpp/sampling.h"

#include "cpp/util.h"

// Some decluttering macros

#define TORCH_CHECK_DTYPE(__x, __dtype) TORCH_CHECK((__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_DTYPE_OPT(__x, __dtype) TORCH_CHECK((__x).device().is_meta() || (__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_SHAPES(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_SHAPES_OPT(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).device().is_meta() || (__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")


// Packing functions

void pack_rows_4
(
    torch::Tensor input,
    torch::Tensor output
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    TORCH_CHECK_DTYPE(input, kShort);
    TORCH_CHECK_DTYPE(output, kInt);
    TORCH_CHECK_SHAPES(input, 0, output, 0, 1);
    TORCH_CHECK_SHAPES(input, 1, output, 1, 8);

    int rows = input.size(0);
    int columns = input.size(1);

    pack_rows_4_cuda
    (
        (uint16_t*) input.data_ptr(),
        (uint32_t*) output.data_ptr(),
        rows,
        columns
    );
}

void pack_columns
(
    torch::Tensor input,
    torch::Tensor output,
    int bits
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    TORCH_CHECK_DTYPE(input, kShort);
    TORCH_CHECK_DTYPE(output, kInt);
    TORCH_CHECK_SHAPES(input, 1, output, 1, 1);

    int in_rows = input.size(0);
    int columns = input.size(1);
    int out_rows = output.size(0);
    int exp_out_rows = in_rows * bits / 32;
    TORCH_CHECK(out_rows == exp_out_rows, "Wrong output shape for input and bitrate")

    pack_columns_cuda
    (
        (uint16_t*) input.data_ptr(),
        (uint32_t*) output.data_ptr(),
        in_rows,
        out_rows,
        columns,
        bits
    );
}


// Quantization functions

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
)
{
    TORCH_CHECK_DTYPE(input, kFloat);
    TORCH_CHECK_DTYPE(output, kFloat);
    // TORCH_CHECK_SHAPES(input, 0, output, 0, 1);
    // TORCH_CHECK_SHAPES(input, 1, output, 1, 1);
    TORCH_CHECK_SHAPES(input, 1, scale, 0, 1);
    TORCH_CHECK(output.size(0) == p_grid + 1, "Output vector shape doesn't match grid")

    int rows = input.size(0);
    int columns = input.size(1);

    quantize_err_cuda
    (
        (float*) input.data_ptr(),
        (float*) output.data_ptr(),
        (float*) scale.data_ptr(),
        rows,
        columns,
        qzero,
        maxq,
        err_norm,
        min_p,
        max_p,
        p_grid
    );
}

void quantize
(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    torch::Tensor out_q,
    float qzero,
    float maxq
)
{
    TORCH_CHECK_DTYPE(input, kFloat);
    TORCH_CHECK_DTYPE(output, kFloat);
    TORCH_CHECK_SHAPES(input, 0, output, 0, 1);
    TORCH_CHECK_SHAPES(input, 1, output, 1, 1);
    TORCH_CHECK_SHAPES(input, 1, scale, 0, 1);

    int rows = input.size(0);
    int columns = input.size(1);

    quantize_cuda
    (
        (float*) input.data_ptr(),
        (float*) output.data_ptr(),
        (float*) scale.data_ptr(),
        out_q.device().is_meta() ? NULL : (uint16_t*) out_q.data_ptr(),
        rows,
        columns,
        qzero,
        maxq
    );
}


// Quant matrix

uintptr_t make_q_matrix
(
    torch::Tensor q_weight,
    torch::Tensor q_perm,
    torch::Tensor q_invperm,
    torch::Tensor q_scale,
    torch::Tensor q_scale_max,
    torch::Tensor q_groups,
    torch::Tensor gptq_qzeros,
    torch::Tensor gptq_scales,
    torch::Tensor gptq_g_idx,
    torch::Tensor temp_dq
)
{
    TORCH_CHECK_DTYPE(q_weight, kInt);
    TORCH_CHECK_DTYPE_OPT(q_perm, kShort);
    TORCH_CHECK_DTYPE_OPT(q_invperm, kShort);
    TORCH_CHECK_DTYPE_OPT(q_scale, kInt);
    TORCH_CHECK_DTYPE_OPT(q_scale_max, kHalf);
    TORCH_CHECK_DTYPE_OPT(q_groups, kShort);
    TORCH_CHECK_DTYPE_OPT(gptq_qzeros, kInt);
    TORCH_CHECK_DTYPE_OPT(gptq_scales, kHalf);
    TORCH_CHECK_DTYPE_OPT(gptq_g_idx, kInt);

    TORCH_CHECK_SHAPES(q_perm, 0, q_invperm, 0, 1);

    int device = q_weight.device().index();
    int width = q_weight.size(1);
    int groups;
    int height;

    if (!q_scale.device().is_meta())
    {
        TORCH_CHECK_SHAPES(q_weight, 1, q_scale, 1, 8);
        TORCH_CHECK_SHAPES(q_scale_max, 0, q_scale, 0, 1);
        groups = q_scale.size(0);
        height = q_invperm.size(0);
    }
    else
    {
        TORCH_CHECK_SHAPES(q_weight, 1, gptq_qzeros, 1, 8);
        TORCH_CHECK_SHAPES(q_weight, 1, gptq_scales, 1, 1);
        groups = gptq_qzeros.size(0);
        height = q_weight.size(0) * 8;
    }

    TORCH_CHECK(temp_dq.size(0) >= width * height, "Insufficient size of temp_dq buffer")

    QMatrix* m = new QMatrix
    (
        device,
        height,
        width,
        groups,
        (uint32_t*) q_weight.data_ptr(),
        q_perm.device().is_meta() ? NULL : (uint16_t*) q_perm.data_ptr(),
        q_invperm.device().is_meta() ? NULL : (uint16_t*) q_invperm.data_ptr(),
        q_scale.device().is_meta() ? NULL : (uint32_t*) q_scale.data_ptr(),
        q_scale_max.device().is_meta() ? NULL : (half*) q_scale_max.data_ptr(),
        q_groups.device().is_meta() ? NULL : (uint16_t*) q_groups.data_ptr(),
        gptq_qzeros.device().is_meta() ? NULL : (uint32_t*) gptq_qzeros.data_ptr(),
        gptq_scales.device().is_meta() ? NULL : (half*) gptq_scales.data_ptr(),
        gptq_g_idx.device().is_meta() ? NULL : (uint32_t*) gptq_g_idx.data_ptr(),
        (half*) temp_dq.data_ptr()
    );

    if (m->failed) throw std::runtime_error("CUDA out of memory");

    return reinterpret_cast<uintptr_t> (m);
}

void free_q_matrix
(
    uintptr_t handle
)
{
    QMatrix* m = reinterpret_cast<QMatrix*> (handle);
    delete m;
}

void reconstruct
(
    uintptr_t q_handle,
    torch::Tensor output
)
{
    QMatrix* qm = reinterpret_cast<QMatrix*> (q_handle);
    TORCH_CHECK(qm->height == output.size(0) && qm->width == output.size(1), "Output tensor doesn't match shape of QMatrix")
    TORCH_CHECK_DTYPE(output, kHalf);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(output));

    qm->reconstruct((half*) output.data_ptr());
}


// Matmul

void gemm_half_q_half
(
    torch::Tensor a,
    uintptr_t b,
    torch::Tensor c,
    bool force_cuda
)
{
    QMatrix* qm = reinterpret_cast<QMatrix*> (b);

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(c, kHalf);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK(qm->height == a.size(1), "a and b have incompatible shapes")
    TORCH_CHECK(qm->width == c.size(1), "b and c have incompatible shapes")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

    gemm_half_q_half_cuda
    (
        at::cuda::getCurrentCUDABlasHandle(),
        (const half*) a.data_ptr(),
        qm,
        (half*) c.data_ptr(),
        c.size(0), // m
        c.size(1), // n
        a.size(1), // k
        true,
        NULL,
        force_cuda
    );
}


// Quant attention

uintptr_t make_q_attn
(
    torch::Tensor layernorm,
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
    int max_seq_len
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
    if (!layernorm.is_meta()) TORCH_CHECK(qm_o_proj->height == layernorm.size(0), "o_proj is wrong shape")

    QAttn* attn = new QAttn
    (
        (half*) layernorm.is_meta() ? NULL : (half*) layernorm.data_ptr(),
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
        max_seq_len
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

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();

    attn->forward_cuda_1
    (
        cublas_handle,
        (half*) x.data_ptr(),
        batch_size,
        q_len,
        past_len,
        past_lens.device().is_meta() ? NULL : (uint32_t*) past_lens.data_ptr(),
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

// Quant MLP

uintptr_t make_q_mlp
(
    torch::Tensor layernorm,
    float norm_epsilon,
    uintptr_t q_gate,
    uintptr_t q_up,
    uintptr_t q_down,
    torch::Tensor temp_state,
    torch::Tensor temp_a,
    torch::Tensor temp_b,
    torch::Tensor temp_dq,
    int max_rows
)
{
    QMatrix* qm_gate = reinterpret_cast<QMatrix*> (q_gate);
    QMatrix* qm_up = reinterpret_cast<QMatrix*> (q_up);
    QMatrix* qm_down = reinterpret_cast<QMatrix*> (q_down);

    TORCH_CHECK_DTYPE(layernorm, kHalf);
    TORCH_CHECK(qm_gate->height == layernorm.size(0), "gate_proj is wrong shape")
    TORCH_CHECK(qm_up->height == layernorm.size(0), "up_proj is wrong shape")

    QMLP* mlp = new QMLP
    (
        (half*) layernorm.data_ptr(),
        norm_epsilon,
        qm_gate,
        qm_up,
        qm_down,
        (half*) temp_state.data_ptr(),
        (half*) temp_a.data_ptr(),
        (half*) temp_b.data_ptr(),
        (half*) temp_dq.data_ptr(),
        max_rows
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

    TORCH_CHECK(x.size(1) == mlp->gate->height, "x is wrong shape");
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


// RoPE rotary positional embeddings, in-place

void rope_
(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos,
    int past_len,
    int num_heads,
    int head_dim
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(sin, kHalf);
    TORCH_CHECK_DTYPE(cos, kHalf);
    TORCH_CHECK(head_dim == cos.size(-1), "cos table does not match head_dim");
    TORCH_CHECK(head_dim == sin.size(-1), "sin table does not match head_dim");

    int batch_size = x.size(0);
    int rows_per_batch = x.numel() / head_dim / batch_size;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    rope_cuda
    (
        (half*) x.data_ptr(),
        (const half*) sin.data_ptr(),
        (const half*) cos.data_ptr(),
        batch_size,
        rows_per_batch,
        head_dim,
        num_heads,
        past_len,
        NULL
    );
}


// RMS layernorm

void rms_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    float epsilon
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);
    TORCH_CHECK_SHAPES(x, 0, y, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, y, 1, 1);

    int rows = x.size(0);
    int dim = x.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    rms_norm_cuda
    (
        (half*) x.data_ptr(),
        (half*) w.data_ptr(),
        (half*) y.data_ptr(),
        epsilon,
        rows,
        dim
    );
}

void rms_norm_
(
    torch::Tensor x,
    torch::Tensor w,
    float epsilon
)
{
    rms_norm(x, w, x, epsilon);
}


// Sampling

void apply_rep_penalty
(
    torch::Tensor sequence,
    float penalty_max,
    int sustain,
    int decay,
    torch::Tensor logits
)
{
    TORCH_CHECK_DTYPE(sequence, kLong);
    TORCH_CHECK_DTYPE(logits, kFloat);
    TORCH_CHECK_SHAPES(sequence, 0, logits, 0, 1);

    int vocab_size = logits.size(-1);
    int bsz = sequence.size(0);
    int seq_len = sequence.size(-1);

    for (int i = 0; i < bsz; i++)
    {
        apply_rep_penalty_cpu
        (
            vocab_size,
            ((uint64_t*) sequence.data_ptr()) + i * seq_len,
            penalty_max,
            sustain,
            decay,
            seq_len,
            ((float*) logits.data_ptr()) + i * vocab_size
        );
    }
}

std::vector<float> sample_basic
(
    torch::Tensor logits,           // shape [bsz, vocab_size]
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    float tfs,
    float typical,
    float random,
    torch::Tensor output_tokens,    // shape [bsz, 1]
    torch::Tensor output_probs,     // shape [bsz, 1]
    torch::Tensor logit_filter,     // shape [bsz, vocab_size]
    bool mirostat,
    std::vector<float>& mirostat_mu,
    float mirostat_tau,
    float mirostat_eta,
    float post_temperature
)
{
    TORCH_CHECK_DTYPE(logits, kFloat);
    TORCH_CHECK_DTYPE(output_tokens, kLong);
    TORCH_CHECK_DTYPE(output_probs, kFloat);
    TORCH_CHECK_DTYPE(logits, kFloat);
    TORCH_CHECK_DTYPE(logit_filter, kBool);

    TORCH_CHECK_SHAPES(logit_filter, 0, logits, 0, 1);
    TORCH_CHECK_SHAPES(logit_filter, 1, logits, 1, 1);

    int vocab_size = logits.size(-1);
    int bsz = logits.size(0);

    float* temp_probs = (float*) malloc(vocab_size * sizeof(float));
    int* temp_indices = (int*) malloc(vocab_size * sizeof(int));

//    int64_t* output_tokens_ptr = (int64_t*) output_tokens.data_ptr();
//    float* output_probs_ptr = (float*) output_tokens.data_ptr();
    float* logits_ptr = (float*) logits.data_ptr();

    bool* logits_filter_ptr = (bool*) logit_filter.data_ptr();

    for (int i = 0; i < bsz; i++)
    {
        softmax_cpu
        (
            vocab_size,
            temperature,
            logits_ptr + i * vocab_size,
            logits_filter_ptr + i * vocab_size,
            temp_probs
        );

        if (top_k == 1)
        {
            int index = greedy_sample(vocab_size, logits_ptr + i * vocab_size, logits_filter_ptr + i * vocab_size);
            output_tokens[i] = index;
            output_probs[i] = temp_probs[index];
            continue;
        }

        for (int j = 0; j < vocab_size; j++) temp_indices[j] = j;
        int num_candidates = vocab_size;

        if (top_k > 0 && top_k < vocab_size)
        {
            num_candidates = top_k_cpu(num_candidates, temp_probs, temp_indices, top_k);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (top_p > 0.0f && top_p < 1.0f)
        {
            num_candidates = top_p_cpu(num_candidates, temp_probs, temp_indices, top_p);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (min_p > 0.0f && min_p < 1.0f)
        {
            num_candidates = min_p_cpu(num_candidates, temp_probs, temp_indices, min_p);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (tfs > 0.0f && tfs < 1.0f)
        {
            num_candidates = tfs_cpu(num_candidates, temp_probs, temp_indices, tfs);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (typical > 0.0f && typical < 1.0f)
        {
            num_candidates = typical_cpu(num_candidates, temp_probs, temp_indices, typical);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (mirostat)
        {
            num_candidates = mirostat_pre_cpu(num_candidates, temp_probs, temp_indices, mirostat_mu[i], mirostat_tau, mirostat_eta);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (post_temperature != 1.0f)
        {
            num_candidates = post_softmax_temperature(num_candidates, temp_probs, temp_indices, post_temperature);
        }

        num_candidates = multinomial_cpu(num_candidates, temp_probs, temp_indices, random);
        output_tokens[i] = temp_indices[0];
        output_probs[i] = temp_probs[0];

        if (mirostat)
        {
            mirostat_mu[i] = mirostat_post_cpu(num_candidates, temp_probs, temp_indices, mirostat_mu[i], mirostat_tau, mirostat_eta);
        }
    }

    free(temp_probs);
    free(temp_indices);

    return mirostat_mu;
}


// Filtering

void logit_filter_exclusive
(
    torch::Tensor filter,                                       // shape [bsz, vocab_size]
    const std::vector<std::vector<int>> &exclusive_lists
)
{
    TORCH_CHECK_DTYPE(filter, kBool);
    TORCH_CHECK((uint64_t) filter.size(0) == exclusive_lists.size(), "Number of lists does not match batch size")

    bool* filter_ptr = (bool*) filter.data_ptr();
    unsigned int vocab_size = filter.size(1);

    for(const auto& list : exclusive_lists)
    {
        unsigned int id = 0;
        unsigned int next_id_idx = 0;
        unsigned int next_id = list[next_id_idx];

        while (id < vocab_size)
        {
            while (id < next_id)
            {
                filter_ptr[id] = false;
                id++;
            }
            id++;
            next_id_idx++;
            if (next_id_idx >= list.size()) next_id = vocab_size;
            else next_id = list[next_id_idx];
        }

        filter_ptr += vocab_size;
    }
}

// For cache conversion

void fp16_to_fp8(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width)
{
    TORCH_CHECK_DTYPE(in_tensor, kHalf);
    TORCH_CHECK_DTYPE(out_tensor, kUInt8);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(in_tensor));

    TORCH_CHECK_SHAPES(in_tensor, 0, out_tensor, 0, 1);
    TORCH_CHECK_SHAPES(in_tensor, 1, out_tensor, 1, 1);
    TORCH_CHECK_SHAPES(in_tensor, 2, out_tensor, 2, 1);
    TORCH_CHECK_SHAPES(in_tensor, 3, out_tensor, 3, 1);

    int stride = in_tensor.size(1) * in_tensor.size(2) * in_tensor.size(3);
    int height = batch_size;

    int tsize = in_tensor.size(2) * in_tensor.size(3);
    offset *= tsize;
    width *= tsize;

    array_fp16_to_fp8_cuda((const half*) (in_tensor.data_ptr()), (unsigned char*)(out_tensor.data_ptr()), stride, height, offset, width);
}

void fp8_to_fp16(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width)
{
    TORCH_CHECK_DTYPE(in_tensor, kUInt8);
    TORCH_CHECK_DTYPE(out_tensor, kHalf);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(in_tensor));

    TORCH_CHECK_SHAPES(in_tensor, 0, out_tensor, 0, 1);
    TORCH_CHECK_SHAPES(in_tensor, 1, out_tensor, 1, 1);
    TORCH_CHECK_SHAPES(in_tensor, 2, out_tensor, 2, 1);
    TORCH_CHECK_SHAPES(in_tensor, 3, out_tensor, 3, 1);

    int stride = in_tensor.size(1) * in_tensor.size(2) * in_tensor.size(3);
    int height = batch_size;

    int tsize = in_tensor.size(2) * in_tensor.size(3);
    offset *= tsize;
    width *= tsize;

    array_fp8_to_fp16_cuda((const unsigned char*)(in_tensor.data_ptr()), (half*)(out_tensor.data_ptr()), stride, height, offset, width);
}

//void array_fp16_to_fp8_ref(torch::Tensor in_tensor, torch::Tensor out_tensor, int size)
//{
//    TORCH_CHECK_DTYPE(in_tensor, kHalf);
//    TORCH_CHECK_DTYPE(out_tensor, kUInt8);
//    array_fp16_to_fp8_ref_cuda((const half*) (in_tensor.data_ptr()), (unsigned char*)(out_tensor.data_ptr()), size);
//}
//
//void array_fp8_to_fp16_ref(torch::Tensor in_tensor, torch::Tensor out_tensor, int size)
//{
//    TORCH_CHECK_DTYPE(in_tensor, kUInt8);
//    TORCH_CHECK_DTYPE(out_tensor, kHalf);
//    array_fp8_to_fp16_ref_cuda((const unsigned char*)(in_tensor.data_ptr()), (half*)(out_tensor.data_ptr()), size);
//}

// Bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pack_rows_4", &pack_rows_4, "pack_rows_4");
    m.def("pack_columns", &pack_columns, "pack_columns");
    m.def("quantize_err", &quantize_err, "quantize_err");
    m.def("quantize", &quantize, "quantize");
    m.def("make_q_matrix", &make_q_matrix, "make_q_matrix");
    m.def("free_q_matrix", &free_q_matrix, "free_q_matrix");
    m.def("reconstruct", &reconstruct, "reconstruct");
    m.def("make_q_mlp", &make_q_mlp, "make_q_mlp");
    m.def("free_q_mlp", &free_q_mlp, "free_q_mlp");
    m.def("q_mlp_forward_", &q_mlp_forward_, "q_mlp_forward_");
    m.def("q_mlp_set_loras", &q_mlp_set_loras, "q_mlp_set_loras");
    m.def("make_q_attn", &make_q_attn, "make_q_attn");
    m.def("free_q_attn", &free_q_attn, "free_q_attn");
    m.def("q_attn_forward_1", &q_attn_forward_1, "q_attn_forward_1");
    m.def("q_attn_forward_2", &q_attn_forward_2, "q_attn_forward_2");
    m.def("q_attn_set_loras", &q_attn_set_loras, "q_attn_set_loras");
    m.def("quantize_range", &quantize_range, "quantize_range");
    m.def("gemm_half_q_half", &gemm_half_q_half, "gemm_half_q_half");
    m.def("rms_norm", &rms_norm, "rms_norm");
    m.def("rms_norm_", &rms_norm_, "rms_norm_");
    m.def("rope_", &rope_, "rope_");
    m.def("apply_rep_penalty", &apply_rep_penalty, "apply_rep_penalty");
    m.def("sample_basic", &sample_basic, "sample_basic");
    m.def("logit_filter_exclusive", &logit_filter_exclusive, "logit_filter_exclusive");
    m.def("fp16_to_fp8", &fp16_to_fp8, "fp16_to_fp8");
    m.def("fp8_to_fp16", &fp8_to_fp16, "fp8_to_fp16");
//    m.def("array_fp16_to_fp8_ref", &array_fp16_to_fp8_ref, "array_fp16_to_fp8_ref");
//    m.def("array_fp8_to_fp16_ref", &array_fp8_to_fp16_ref, "array_fp8_to_fp16_ref");
}
