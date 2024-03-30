#include "q_mlp.cuh"
#include "q_gemm.cuh"
#include "rms_norm.cuh"
#include "layer_norm.cuh"
#include "util.cuh"
#include "matrix_view.cuh"
#include "lora.cuh"
#include "quant/qdq_util.cuh"
#include "../config.h"
#include "compat.cuh"

const int THREADS_X = 32;
const int THREADS_Y = 4;

#include "q_mlp_softmax.cuh"
#include "q_mlp_activation.cuh"

// const int MAX_DIMENSION = 8192;

QMLP::QMLP
(
    half* _layernorm,
    half* _layernorm_bias,
    bool _layernorm_is_rms,
    float _norm_epsilon,
    QMatrix* _gate,
    QMatrix* _up,
    QMatrix* _down,
    half* _temp_state,
    half* _temp_a,
    half* _temp_b,
    half* _temp_dq,
    int _max_rows,
    bool _act_gelu,
    bool _has_residual
):
    layernorm(_layernorm),
    layernorm_bias(_layernorm_bias),
    layernorm_is_rms(_layernorm_is_rms),
    norm_epsilon(_norm_epsilon),
    gate(_gate),
    up(_up),
    down(_down),
    temp_state(_temp_state),
    temp_a(_temp_a),
    temp_b(_temp_b),
    temp_dq(_temp_dq),
    max_rows(_max_rows),
    act_gelu(_act_gelu),
    has_residual(_has_residual)
{
}

QMLP::~QMLP() {
}

void QMLP::forward_
(
    cublasHandle_t cublas_handle,
    half* x,
    int rows,
    int columns,
    const std::vector<uintptr_t>& loras,
    half* lora_temp
)
{
    bool use_half2 = true;
    int intermediate_size = up->width;

    // Activation kernel dims

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = THREADS_Y;
    gridDim.x = DIVIDE(up->width, THREADS_X) / (use_half2 ? 2 : 1);
    gridDim.y = DIVIDE(rows, THREADS_Y);

    // Layernorm

    half* norm_state = x;

    if (layernorm)
    {
        if (layernorm_is_rms)
            rms_norm_cuda(x, layernorm, temp_state, norm_epsilon, rows, columns);
        else
            layer_norm_cuda(x, layernorm, layernorm_bias, temp_state, norm_epsilon, rows, columns);
        norm_state = temp_state;
    }

    // Up proj with gate

    if (gate)
    {
        gemm_half_q_half_cuda(cublas_handle, norm_state, gate, temp_a, rows, intermediate_size, columns, true, temp_dq);
        gemm_half_q_half_cuda(cublas_handle, norm_state, up,   temp_b, rows, intermediate_size, columns, true, temp_dq);

        apply_loras_cuda(cublas_handle, gate_proj_lora, loras, gate, norm_state, temp_a, lora_temp, rows);
        apply_loras_cuda(cublas_handle, up_proj_lora,   loras, up,   norm_state, temp_b, lora_temp, rows);

        fp_act_mul_kernel kernel = pick_act_mul_kernel(use_half2, false, act_gelu);
        kernel<<<gridDim, blockDim>>>(temp_a, temp_b, rows, intermediate_size, NULL, 0);
    }

    // Up proj without gate

    else
    {
        gemm_half_q_half_cuda(cublas_handle, norm_state, up,   temp_a, rows, intermediate_size, columns, true, temp_dq);

        apply_loras_cuda(cublas_handle, up_proj_lora,   loras, up,   norm_state, temp_a, lora_temp, rows);

        fp_act_kernel kernel = pick_act_kernel(use_half2, false, act_gelu);
        kernel<<<gridDim, blockDim>>>(temp_a, rows, intermediate_size, NULL, 0);
    }

    // Down proj

    gemm_half_q_half_cuda(cublas_handle, temp_a, down, x, rows, columns, intermediate_size, !has_residual, temp_dq);

    apply_loras_cuda(cublas_handle, down_proj_lora, loras, down, temp_a, x, lora_temp, rows);
}


QMoEMLP::QMoEMLP
(
    half* _layernorm,
    half* _layernorm_bias,
    bool _layernorm_is_rms,
    float _norm_epsilon,
    half* _gate,
    int _num_experts,
    int _num_experts_per_token,
    std::vector<QMatrix*>& _w1,
    std::vector<QMatrix*>& _w2,
    std::vector<QMatrix*>& _w3,
    half* _temp_state,
    half* _temp_gathered_state,
    half* _temp_a,
    half* _temp_b,
    half* _temp_logits,
    half* _temp_dq,
    int _max_rows,
    int _hidden_dim,
    bool _act_gelu
):
    layernorm(_layernorm),
    layernorm_bias(_layernorm_bias),
    layernorm_is_rms(_layernorm_is_rms),
    norm_epsilon(_norm_epsilon),
    gate(_gate),
    num_experts(_num_experts),
    num_experts_per_token(_num_experts_per_token),
    w1(_w1),
    w2(_w2),
    w3(_w3),
    temp_state(_temp_state),
    temp_gathered_state(_temp_gathered_state),
    temp_a(_temp_a),
    temp_b(_temp_b),
    temp_logits(_temp_logits),
    temp_dq(_temp_dq),
    max_rows(_max_rows),
    hidden_dim(_hidden_dim),
    act_gelu(_act_gelu)
{
//    for (int i = 0; i < num_experts; ++i)
//    {
//        std::unordered_map<uintptr_t, std::tuple<half*, half*, int>> w1;
//        std::unordered_map<uintptr_t, std::tuple<half*, half*, int>> w2;
//        std::unordered_map<uintptr_t, std::tuple<half*, half*, int>> w3;
//        w1_lora.push_back(w1);
//        w2_lora.push_back(w2);
//        w3_lora.push_back(w3);
//    }
}

QMoEMLP::~QMoEMLP() {
}

void QMoEMLP::forward_
(
    cublasHandle_t cublas_handle,
    half* x,
    int rows,
    int columns
//    const std::vector<uintptr_t>& loras,
//    half* lora_temp
)
{
    if (num_experts != 4 && num_experts != 8 && num_experts != 16)
    {
        printf(" ## num_experts must be 4, 8 or 16\n");
        return;
    }

    bool use_half2 = true;

    // Norm

    if (layernorm_is_rms)
        rms_norm_cuda(x, layernorm, temp_state, norm_epsilon, rows, columns);
    else
        layer_norm_cuda(x, layernorm, layernorm_bias, temp_state, norm_epsilon, rows, columns);

    // Compute gate logits

    half alpha_ = __float2half(1.0f);
    half beta_ = __float2half(0.0f);
    cublasHgemm(cublas_handle,
                CUBLAS_OP_T, // gate is column-major
                CUBLAS_OP_N,
                num_experts, rows, hidden_dim,
                &alpha_,
                gate, hidden_dim,
                temp_state, hidden_dim,
                &beta_,
                temp_logits, num_experts);

    // Compute softmax filter to and normalize top-k outputs

    dim3 blockDim, gridDim;
    blockDim.x = WARPS;
    blockDim.y = 1;
    gridDim.x = 1;
    gridDim.y = DIVIDE(rows, WARPS);
    if (num_experts == 4)
        softmax4_topk_norm_kernel<<<gridDim, blockDim>>>(temp_logits, rows, num_experts_per_token);
    else if (num_experts == 8)
        softmax8_topk_norm_kernel<<<gridDim, blockDim>>>(temp_logits, rows, num_experts_per_token);
    else if (num_experts == 16)
        softmax16_topk_norm_kernel<<<gridDim, blockDim>>>(temp_logits, rows, num_experts_per_token);

    // For small no. rows, execute all kernels but pass the routing weights. Rows with a weight of zero will skip dot
    // product accum and kernels launched with only zero-weights will exit prematurely.

    if (rows <= MAX_Q_GEMM_WEIGHTS)
    {
        int intermediate_size = w1[0]->width;
        fp_act_mul_kernel kernel = pick_act_mul_kernel(use_half2, true, act_gelu);

        for (int i = 0; i < num_experts; i++)
        {
            gemm_half_q_half_cuda(cublas_handle, temp_state, w1[i], temp_a, rows, intermediate_size, columns, true, temp_dq, true, temp_logits + i, num_experts, false);
            gemm_half_q_half_cuda(cublas_handle, temp_state, w3[i], temp_b, rows, intermediate_size, columns, true, temp_dq, true, temp_logits + i, num_experts, false);

//            apply_loras_cuda(cublas_handle, w1_lora[i], loras, w1[i], temp_state, temp_a, lora_temp, rows);
//            apply_loras_cuda(cublas_handle, w3_lora[i], loras, w3[i], temp_state, temp_b, lora_temp, rows);

            blockDim.x = THREADS_X;
            blockDim.y = THREADS_Y;
            gridDim.x = DIVIDE(intermediate_size, THREADS_X) / (use_half2 ? 2 : 1);
            gridDim.y = DIVIDE(rows, THREADS_Y);
            kernel<<<gridDim, blockDim>>>(temp_a, temp_b, rows, intermediate_size, temp_logits + i, num_experts);

            gemm_half_q_half_cuda(cublas_handle, temp_a, w2[i], x, rows, columns, intermediate_size, false, temp_dq, true, temp_logits + i, num_experts, true);

//            apply_loras_cuda(cublas_handle, w2_lora[i], loras, w2[i], temp_a, x, lora_temp, rows);
        }
    }

    // Gather larger number of rows in separate batches according to which experts they trigger, evaluate each MLP
    // only on the affected rows and scale by routing weights while adding back directly onto the residual hidden state

    else
    {
        printf(" ## ropws > %i not implemented\n", MAX_Q_GEMM_WEIGHTS);
        DBGI(rows);
    }
}
