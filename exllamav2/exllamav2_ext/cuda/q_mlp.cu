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

#include <iostream>

enum KernelLabels
{
    NORM = 1,
    GATE,
    UP,
    DOWN,
    POST_NORM
};

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
    bool _has_residual,
    half* _post_layernorm,
    half* _post_layernorm_bias,
    bool _residual_fp32,
    bool _use_graphs
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
    has_residual(_has_residual),
    post_layernorm(_post_layernorm),
    post_layernorm_bias(_post_layernorm_bias),
    residual_fp32(_residual_fp32),
    use_graphs(_use_graphs)
{
}

QMLP::~QMLP()
{
    for (const auto& pair : graph_map) delete pair.second;
}

void QMLP::forward_
(
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    void* x,
    int rows,
    int columns,
    const std::vector<uintptr_t>& loras,
    half* lora_temp
)
{
    // Don't use graph if LoRAs enabled or if module might invoke cuBLAS

    if (!use_graphs || loras.size() || rows > MAX_Q_GEMM_ROWS)
    {
        forward_run_(stream, cublas_handle, x, rows, columns, loras, lora_temp);
        return;
    }

    QMLP_params_const pc = { rows, columns };
    auto it = graph_map.find(pc);
    Graph* graph;
    if (it == graph_map.end())
    {
        graph = new Graph();
        graph_map[pc] = graph;
//        printf("**** new graph ****\n");
//        DBGI2(rows, columns);
//        DBGX(x);
    }
    else graph = it->second;
    if (graph->count())
    {
        graph->begin_capture(stream);
        forward_run_(stream, cublas_handle, (void*) x, rows, columns, loras, lora_temp, graph);
        graph->end_capture(stream);
//        printf("**** record ****\n");
//        DBGI2(rows, columns);
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
            q_gemm_cuda_update_a(graph, KernelLabels::GATE, x);
            q_gemm_cuda_update_a(graph, KernelLabels::UP, x);
        }

        if (!post_layernorm)
        {
            q_gemm_cuda_update_c(graph, KernelLabels::DOWN, x);
        }
        else
        {
            if (layernorm_is_rms)
                rms_norm_cuda_update_y(graph, KernelLabels::POST_NORM, x);
            else
                layer_norm_cuda_update_y(graph, KernelLabels::POST_NORM, x);
        }

        graph->launch(stream);
    }
    else
    {
        forward_run_(stream, cublas_handle, x, rows, columns, loras, lora_temp);
    }
}

void QMLP::forward_run_
(
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    void* x,
    int rows,
    int columns,
    const std::vector<uintptr_t>& loras,
    half* lora_temp,
    Graph* graph
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

    half* norm_state = (half*) x;

    if (layernorm)
    {
        if (layernorm_is_rms)
            rms_norm_cuda(stream, x, layernorm, temp_state, norm_epsilon, rows, columns, false, residual_fp32, false, graph, KernelLabels::NORM);
        else
            layer_norm_cuda(stream, (half*) x, layernorm, layernorm_bias, temp_state, norm_epsilon, rows, columns, false, graph, KernelLabels::NORM);
        norm_state = temp_state;
    }

    // Up proj with gate

    if (gate)
    {
        gemm_half_q_half_cuda(stream, cublas_handle, norm_state, gate, temp_a, rows, intermediate_size, columns, true, temp_dq, false, NULL, 0, false, graph, KernelLabels::GATE);
        gemm_half_q_half_cuda(stream, cublas_handle, norm_state, up,   temp_b, rows, intermediate_size, columns, true, temp_dq, false, NULL, 0, false, graph, KernelLabels::UP);

        apply_loras_cuda(stream, cublas_handle, gate_proj_lora, loras, gate, norm_state, temp_a, lora_temp, rows);
        apply_loras_cuda(stream, cublas_handle, up_proj_lora,   loras, up,   norm_state, temp_b, lora_temp, rows);

        fp_act_mul_kernel kernel = pick_act_mul_kernel(use_half2, false, act_gelu);
        kernel<<<gridDim, blockDim, 0, stream>>>(temp_a, temp_b, rows, intermediate_size, NULL, 0);
        if (graph) graph->attach_label(stream, 0, 0);
    }

    // Up proj without gate

    else
    {
        gemm_half_q_half_cuda(stream, cublas_handle, norm_state, up,   temp_a, rows, intermediate_size, columns, true, temp_dq, false, NULL, 0, false, graph, KernelLabels::GATE);

        apply_loras_cuda(stream, cublas_handle, up_proj_lora,   loras, up,   norm_state, temp_a, lora_temp, rows);

        fp_act_kernel kernel = pick_act_kernel(use_half2, false, act_gelu);
        kernel<<<gridDim, blockDim, 0, stream>>>(temp_a, rows, intermediate_size, NULL, 0);
        if (graph) graph->attach_label(stream, 0, 0);
    }

    // Down proj without post_layernorm

    if (!post_layernorm)
    {
        gemm_half_q_half_cuda(stream, cublas_handle, temp_a, down, (half*) x, rows, columns, intermediate_size, !has_residual, temp_dq, false, NULL, 0, false, graph, KernelLabels::DOWN);
    }

    // Down proj with post_layernorm

    else
    {
        gemm_half_q_half_cuda(stream, cublas_handle, temp_a, down, temp_state, rows, columns, intermediate_size, true, temp_dq, false, NULL, 0, false, graph, 0);
        if (layernorm_is_rms)
            rms_norm_cuda(stream, temp_state, post_layernorm, x, norm_epsilon, rows, columns, true, false, residual_fp32, graph, KernelLabels::POST_NORM);
        else
            layer_norm_cuda(stream, temp_state, post_layernorm, post_layernorm_bias, (half*) x, norm_epsilon, rows, columns, true, graph, KernelLabels::POST_NORM);
    }

    apply_loras_cuda(stream, cublas_handle, down_proj_lora, loras, down, temp_a, (half*) x, lora_temp, rows);
}

void act_mul_cuda
(
    cudaStream_t stream,
    half* gate,
    half* up,
    int rows,
    int dim,
    bool act_gelu
)
{
    bool use_half2 = true;

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = THREADS_Y;
    gridDim.x = DIVIDE(dim, THREADS_X) / (use_half2 ? 2 : 1);
    gridDim.y = DIVIDE(rows, THREADS_Y);

    fp_act_mul_kernel kernel = pick_act_mul_kernel(use_half2, false, act_gelu);
    kernel<<<gridDim, blockDim, 0, stream>>>(gate, up, rows, dim, NULL, 0);
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
    cudaStream_t stream,
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
        rms_norm_cuda(stream, x, layernorm, temp_state, norm_epsilon, rows, columns);
    else
        layer_norm_cuda(stream, x, layernorm, layernorm_bias, temp_state, norm_epsilon, rows, columns);

    // Compute gate logits

    half alpha_ = __float2half(1.0f);
    half beta_ = __float2half(0.0f);
    cublasSetStream(cublas_handle, stream);
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
        softmax4_topk_norm_kernel<<<gridDim, blockDim, 0, stream>>>(temp_logits, rows, num_experts_per_token);
    else if (num_experts == 8)
        softmax8_topk_norm_kernel<<<gridDim, blockDim, 0, stream>>>(temp_logits, rows, num_experts_per_token);
    else if (num_experts == 16)
        softmax16_topk_norm_kernel<<<gridDim, blockDim, 0, stream>>>(temp_logits, rows, num_experts_per_token);

    // For small no. rows, execute all kernels but pass the routing weights. Rows with a weight of zero will skip dot
    // product accum and kernels launched with only zero-weights will exit prematurely.

    if (rows <= MAX_Q_GEMM_WEIGHTS)
    {
        int intermediate_size = w1[0]->width;
        fp_act_mul_kernel kernel = pick_act_mul_kernel(use_half2, true, act_gelu);

        for (int i = 0; i < num_experts; i++)
        {
            gemm_half_q_half_cuda(stream, cublas_handle, temp_state, w1[i], temp_a, rows, intermediate_size, columns, true, temp_dq, true, temp_logits + i, num_experts, false);
            gemm_half_q_half_cuda(stream, cublas_handle, temp_state, w3[i], temp_b, rows, intermediate_size, columns, true, temp_dq, true, temp_logits + i, num_experts, false);

//            apply_loras_cuda(cublas_handle, w1_lora[i], loras, w1[i], temp_state, temp_a, lora_temp, rows);
//            apply_loras_cuda(cublas_handle, w3_lora[i], loras, w3[i], temp_state, temp_b, lora_temp, rows);

            blockDim.x = THREADS_X;
            blockDim.y = THREADS_Y;
            gridDim.x = DIVIDE(intermediate_size, THREADS_X) / (use_half2 ? 2 : 1);
            gridDim.y = DIVIDE(rows, THREADS_Y);
            kernel<<<gridDim, blockDim, 0, stream>>>(temp_a, temp_b, rows, intermediate_size, temp_logits + i, num_experts);

            gemm_half_q_half_cuda(stream, cublas_handle, temp_a, w2[i], x, rows, columns, intermediate_size, false, temp_dq, true, temp_logits + i, num_experts, true);

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
