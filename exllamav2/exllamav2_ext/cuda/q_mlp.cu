#include "q_mlp.cuh"
#include "q_gemm.cuh"
#include "rms_norm.cuh"
#include "layer_norm.cuh"
#include "util.cuh"
#include "matrix_view.cuh"
#include "lora.cuh"
#include "quant/qdq_util.cuh"
#include "../config.h"

#include "q_mlp_softmax.cuh"

#if defined(USE_ROCM)
__device__ __forceinline__ __half2 __compat_h2rcp(__half2 x) {
    return _Float16_2{static_cast<_Float16>(__builtin_amdgcn_rcph(static_cast<__half2_raw>(x).data.x)),
        static_cast<_Float16>(__builtin_amdgcn_rcph(static_cast<__half2_raw>(x).data.y))};
}
#define h2rcp __compat_h2rcp
#endif

const int THREADS_X = 32;
const int THREADS_Y = 4;
// const int MAX_DIMENSION = 8192;

__device__ __forceinline__ half silu(half x)
{
    half one = __float2half(1.0f);
    half neg_x = __hneg(x);
    half e = hexp(neg_x);
    half sum = __hadd(one, e);
    half r = hrcp(sum);
    half result = __hmul(x, r);
    return result;
}

__device__ __forceinline__ half2 silu(half2 x)
{
    half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 e = h2exp(neg_x);
    half2 sum = __hadd2(one, e);
    half2 r = h2rcp(sum);
    half2 result = __hmul2(x, r);
    return result;
}

__device__ __forceinline__ half gelu(half x)
{
    float xf = __half2float(x);
    const float c = 0.797884560803f;  // sqrt(2/Pi)
    float tanh_arg = c * (xf + 0.044715f * pow(xf, 3));
    xf = 0.5f * xf * (1.0 + tanh(tanh_arg));
    return __float2half_rn(xf);
}

__device__ __forceinline__ half2 gelu(half2 x)
{
    return __halves2half2(gelu(__low2half(x)), gelu(__high2half(x)));
}

typedef void (*fp_silu_mul_kernel)
(
    half*,
    const half*,
    const int,
    const int,
    const half*,
    const int
);

template <bool use_half2, bool use_r_weights, bool act_fn_gelu>
__global__ void silu_mul_kernel
(
    half* __restrict__ x,
    const half* __restrict__ y,
    const int height,
    const int width,
    const half* r_weights,
    const int r_weights_stride
)
{
    MatrixView_half_rw x_(x, height, width);
    MatrixView_half y_(y, height, width);

    int column = (THREADS_X * blockIdx.x + threadIdx.x); if constexpr (use_half2) column *= 2;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= height) return;

    if constexpr (use_r_weights)
    {
        half_uint16 weight(r_weights[row * r_weights_stride]);
        if (!weight.as_uint16)
        {
//             half2 ppp = __float2half2_rn(6.9f);
//             x_.set_half2(row, column, ppp);
            return;
        }
    }

    // silu(x) * y

    if constexpr (use_half2)
    {
        half2 x_item = x_.item_half2(row, column);
        half2 y_item = y_.item_half2(row, column);

        if constexpr (act_fn_gelu)
            x_item = gelu(x_item);
        else
            x_item = silu(x_item);

        x_item = __hmul2(x_item, y_item);

        x_.set_half2(row, column, x_item);
    }
    else
    {
        half x_item = x_.item(row, column);
        half y_item = y_.item(row, column);

        if constexpr (act_fn_gelu)
            x_item = gelu(x_item);
        else
            x_item = silu(x_item);

        x_item = __hmul(x_item, y_item);

        x_.set(row, column, x_item);
    }
}

fp_silu_mul_kernel pick_silu_mul_kernel(bool use_half2, bool mul_r_weights, bool act_fn_gelu)
{
    if (act_fn_gelu)
    {
        if ( use_half2 && !mul_r_weights) return silu_mul_kernel< true, false,  true>;
        if ( use_half2 &&  mul_r_weights) return silu_mul_kernel< true,  true,  true>;
        if (!use_half2 && !mul_r_weights) return silu_mul_kernel<false, false,  true>;
        if (!use_half2 &&  mul_r_weights) return silu_mul_kernel<false,  true,  true>;
    }
    else
    {
        if ( use_half2 && !mul_r_weights) return silu_mul_kernel< true, false, false>;
        if ( use_half2 &&  mul_r_weights) return silu_mul_kernel< true,  true, false>;
        if (!use_half2 && !mul_r_weights) return silu_mul_kernel<false, false, false>;
        if (!use_half2 &&  mul_r_weights) return silu_mul_kernel<false,  true, false>;
    }
    return NULL;
};

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
    bool _act_gelu
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
    act_gelu(_act_gelu)
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
    int intermediate_size = gate->width;

    if (layernorm_is_rms)
        rms_norm_cuda(x, layernorm, temp_state, norm_epsilon, rows, columns);
    else
        layer_norm_cuda(x, layernorm, layernorm_bias, temp_state, norm_epsilon, rows, columns);

    gemm_half_q_half_cuda(cublas_handle, temp_state, gate, temp_a, rows, intermediate_size, columns, true, temp_dq);
    gemm_half_q_half_cuda(cublas_handle, temp_state, up,   temp_b, rows, intermediate_size, columns, true, temp_dq);

    apply_loras_cuda(cublas_handle, gate_proj_lora, loras, gate, temp_state, temp_a, lora_temp, rows);
    apply_loras_cuda(cublas_handle, up_proj_lora,   loras, up,   temp_state, temp_b, lora_temp, rows);

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = THREADS_Y;
    gridDim.x = DIVIDE(up->width, THREADS_X) / (use_half2 ? 2 : 1);
    gridDim.y = DIVIDE(rows, THREADS_Y);

    fp_silu_mul_kernel kernel = pick_silu_mul_kernel(use_half2, false, act_gelu);
    kernel<<<gridDim, blockDim>>>(temp_a, temp_b, rows, intermediate_size, NULL, 0);

    gemm_half_q_half_cuda(cublas_handle, temp_a, down, x, rows, columns, intermediate_size, false, temp_dq);

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
    if (num_experts != 8 && num_experts != 4)
    {
        printf(" ## num_experts != 4 or 8 not implemented\n");
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

    // For small no. rows, execute all kernels but pass the routing weights. Rows with a weight of zero will skip dot
    // product accum and kernels launched with only zero-weights will exit prematurely.

    if (rows <= MAX_Q_GEMM_WEIGHTS)
    {
        int intermediate_size = w1[0]->width;
        fp_silu_mul_kernel kernel = pick_silu_mul_kernel(use_half2, true, act_gelu);

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
