#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#include "config.h"

#include "cuda/pack_tensor.cuh"
#include "cuda/quantize.cuh"
#include "cuda/q_matrix.cuh"
#include "cuda/q_attn.cuh"
#include "cuda/q_mlp.cuh"
#include "cuda/q_gemm.cuh"
#include "cuda/rms_norm.cuh"
#include "cuda/rope.cuh"

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

    return reinterpret_cast<uintptr_t> (m);
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

    TORCH_CHECK_DTYPE(layernorm, kHalf);

    TORCH_CHECK(qm_q_proj->height == layernorm.size(0), "q_proj is wrong shape")
    TORCH_CHECK(qm_k_proj->height == layernorm.size(0), "k_proj is wrong shape")
    TORCH_CHECK(qm_v_proj->height == layernorm.size(0), "v_proj is wrong shape")
    TORCH_CHECK(qm_o_proj->height == layernorm.size(0), "o_proj is wrong shape")

    QAttn* attn = new QAttn
    (
        (half*) layernorm.data_ptr(),
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
    torch::Tensor cos
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
        (half*) cos.data_ptr()
    );
}

void q_attn_forward_2
(
    uintptr_t q_attn,
    torch::Tensor x,
    torch::Tensor attn_output,
    int batch_size,
    int q_len
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
        batch_size
    );
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

void q_mlp_forward_
(
    uintptr_t q_mlp,
    torch::Tensor x
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
        x.size(1)  // columns == hidden_size
    );
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

void sample_basic
(
    torch::Tensor logits,           // shape [bsz, vocab_size]
    float temperature,
    int top_k,
    float top_p,
    float random,
    torch::Tensor output_tokens,    // shape [bsz, 1]
    torch::Tensor output_probs      // shape [bsz, 1]
)
{
    TORCH_CHECK_DTYPE(logits, kFloat);
    TORCH_CHECK_DTYPE(output_tokens, kLong);
    TORCH_CHECK_DTYPE(output_probs, kFloat);
    TORCH_CHECK_DTYPE(logits, kFloat);

    int vocab_size = logits.size(-1);
    int bsz = logits.size(0);

    float* temp_probs = (float*) malloc(vocab_size * sizeof(float));
    int* temp_indices = (int*) malloc(vocab_size * sizeof(int));

    int64_t* output_tokens_ptr = (int64_t*) output_tokens.data_ptr();
    float* output_probs_ptr = (float*) output_tokens.data_ptr();
    float* logits_ptr = (float*) logits.data_ptr();

    for (int i = 0; i < bsz; i++)
    {
        softmax_cpu(vocab_size, temperature, logits_ptr + i * vocab_size, temp_probs);

        if (top_k == 1)
        {
            int index = greedy_sample(vocab_size, logits_ptr + i * vocab_size);
            output_tokens[i] = index;
            output_probs[i] = temp_probs[index];
            continue;
        }

//        if (top_k == 1)
//        {
//            int index = greedy_sample(vocab_size, logits_ptr + i * vocab_size);
//            output_tokens[i] = index;
//            output_probs[i] = temp_probs[index];
//            continue;
//        }
//
//        softmax_cpu(vocab_size, temperature, logits_ptr + i * vocab_size, temp_probs);

        for (int j = 0; j < vocab_size; j++) temp_indices[j] = j;
        int num_candidates = vocab_size;

        sort_descending(num_candidates, temp_probs, temp_indices, top_k);

        if (top_k > 0)
        {
            num_candidates = top_k_cpu(num_candidates, temp_probs, temp_indices, top_k);
            normalize_cpu(num_candidates, temp_probs);
        }

        if (top_p > 0.0f)
        {
            num_candidates = top_p_cpu(num_candidates, temp_probs, temp_indices, top_p);
            normalize_cpu(num_candidates, temp_probs);
        }

        num_candidates = multinomial_cpu(num_candidates, temp_probs, temp_indices, random);
        output_tokens[i] = temp_indices[0];
        output_probs[i] = temp_probs[0];
    }

    free(temp_probs);
    free(temp_indices);
}


// Bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pack_rows_4", &pack_rows_4, "pack_rows_4");
    m.def("pack_columns", &pack_columns, "pack_columns");
    m.def("quantize_err", &quantize_err, "quantize_err");
    m.def("quantize", &quantize, "quantize");
    m.def("make_q_matrix", &make_q_matrix, "make_q_matrix");
    m.def("reconstruct", &reconstruct, "reconstruct");
    m.def("make_q_mlp", &make_q_mlp, "make_q_mlp");
    m.def("q_mlp_forward_", &q_mlp_forward_, "q_mlp_forward_");
    m.def("make_q_attn", &make_q_attn, "make_q_attn");
    m.def("q_attn_forward_1", &q_attn_forward_1, "q_attn_forward_1");
    m.def("q_attn_forward_2", &q_attn_forward_2, "q_attn_forward_2");
    m.def("quantize_range", &quantize_range, "quantize_range");
    m.def("gemm_half_q_half", &gemm_half_q_half, "gemm_half_q_half");
    m.def("rms_norm", &rms_norm, "rms_norm");
    m.def("rms_norm_", &rms_norm_, "rms_norm_");
    m.def("rope_", &rope_, "rope_");
    m.def("apply_rep_penalty", &apply_rep_penalty, "apply_rep_penalty");
    m.def("sample_basic", &sample_basic, "sample_basic");
}
