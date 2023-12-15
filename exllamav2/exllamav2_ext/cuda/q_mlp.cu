#include "q_mlp.cuh"
#include "q_gemm.cuh"
#include "rms_norm.cuh"
#include "util.cuh"
#include "matrix_view.cuh"
#include "lora.cuh"
#include "quant/qdq_util.cuh"
#include "../config.h"

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

typedef void (*fp_silu_mul_kernel)
(
    half*,
    const half*,
    const int,
    const int,
    const half*,
    const int
);

template <bool use_half2, bool use_r_weights>
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

        x_item = silu(x_item);
        x_item = __hmul2(x_item, y_item);

        x_.set_half2(row, column, x_item);
    }
    else
    {
        half x_item = x_.item(row, column);
        half y_item = y_.item(row, column);

        x_item = silu(x_item);
        x_item = __hmul(x_item, y_item);

        x_.set(row, column, x_item);
    }
}

fp_silu_mul_kernel pick_silu_mul_kernel(bool use_half2, bool mul_r_weights)
{
    if ( use_half2 && !mul_r_weights) return silu_mul_kernel< true, false>;
    if ( use_half2 &&  mul_r_weights) return silu_mul_kernel< true,  true>;
    if (!use_half2 && !mul_r_weights) return silu_mul_kernel<false, false>;
    if (!use_half2 &&  mul_r_weights) return silu_mul_kernel<false,  true>;
    return NULL;
};

QMLP::QMLP
(
    half* _layernorm,
    float _norm_epsilon,
    QMatrix* _gate,
    QMatrix* _up,
    QMatrix* _down,
    half* _temp_state,
    half* _temp_a,
    half* _temp_b,
    half* _temp_dq,
    int _max_rows
):
    layernorm(_layernorm),
    norm_epsilon(_norm_epsilon),
    gate(_gate),
    up(_up),
    down(_down),
    temp_state(_temp_state),
    temp_a(_temp_a),
    temp_b(_temp_b),
    temp_dq(_temp_dq),
    max_rows(_max_rows)
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

    rms_norm_cuda(x, layernorm, temp_state, norm_epsilon, rows, columns);
    gemm_half_q_half_cuda(cublas_handle, temp_state, gate, temp_a, rows, intermediate_size, columns, true, temp_dq);
    gemm_half_q_half_cuda(cublas_handle, temp_state, up,   temp_b, rows, intermediate_size, columns, true, temp_dq);

    apply_loras_cuda(cublas_handle, gate_proj_lora, loras, gate, temp_state, temp_a, lora_temp, rows);
    apply_loras_cuda(cublas_handle, up_proj_lora,   loras, up,   temp_state, temp_b, lora_temp, rows);

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = THREADS_Y;
    gridDim.x = DIVIDE(up->width, THREADS_X) / (use_half2 ? 2 : 1);
    gridDim.y = DIVIDE(rows, THREADS_Y);

    fp_silu_mul_kernel kernel = pick_silu_mul_kernel(use_half2, false);
    kernel<<<gridDim, blockDim>>>(temp_a, temp_b, rows, intermediate_size, NULL, 0);

    gemm_half_q_half_cuda(cublas_handle, temp_a, down, x, rows, columns, intermediate_size, false, temp_dq);

    apply_loras_cuda(cublas_handle, down_proj_lora, loras, down, temp_a, x, lora_temp, rows);
}


QMoEMLP::QMoEMLP
(
    half* _layernorm,
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
    int _hidden_dim
):
    layernorm(_layernorm),
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
    hidden_dim(_hidden_dim)
{
}

QMoEMLP::~QMoEMLP() {
}

#define WARPS 32

__global__ void softmax8_topk_norm_kernel
(
    half* __restrict__ x,
    const int rows,
    const int topk
)
{
    int row = blockIdx.y * WARPS + threadIdx.x;
    if (row >= rows) return;

    // Softmax

    int4* row_ptr = (int4*) (x + row * 8);
    int4 logits_int4 = *row_ptr;
    half2_uint32 l01(logits_int4.x);
    half2_uint32 l23(logits_int4.y);
    half2_uint32 l45(logits_int4.z);
    half2_uint32 l67(logits_int4.w);
    float f[] =
    {
        __low2float(l01.as_half2),
        __high2float(l01.as_half2),
        __low2float(l23.as_half2),
        __high2float(l23.as_half2),
        __low2float(l45.as_half2),
        __high2float(l45.as_half2),
        __low2float(l67.as_half2),
        __high2float(l67.as_half2)
    };

    float maxf1 = fmaxf(f[0], f[1]);
    float maxf2 = fmaxf(f[2], f[3]);
    float maxf3 = fmaxf(f[4], f[5]);
    float maxf4 = fmaxf(f[6], f[7]);
    maxf1 = fmaxf(maxf1, maxf2);
    maxf2 = fmaxf(maxf3, maxf4);
    maxf1 = fmaxf(maxf1, maxf2);

    float sum = 0;
    for (int i = 0; i < 8; ++i)
    {
        float e = expf(f[i] - maxf1);
        sum += e;
        f[i] = e;
    }
    float epsilon = 1e-8;
    float isum = 1.0f / (sum + 8 * epsilon);
    for (int i = 0; i < 8; ++i) f[i] = f[i] * isum + epsilon;

    // This is awful but surely faster than synchronizing or launching more kernels (??)

    sum = 1.0f;
    for (int i = 0; i < 8 - topk; ++i)
    {
        float minf = 1.0f;
        int minj = -1;
        for (int j = 0; j < 8; ++j)
        {
            if (f[j] > 0 && f[j] < minf)
            {
                minf = f[j];
                minj = j;
            }
        }
        sum -= f[minj];
        f[minj] = 0.0f;
    }

    __syncthreads();

    isum = 1.0f / sum;
    for (int i = 0; i < 8; ++i) f[i] *= isum;

    l01.as_half2 = __floats2half2_rn(f[0], f[1]);
    l23.as_half2 = __floats2half2_rn(f[2], f[3]);
    l45.as_half2 = __floats2half2_rn(f[4], f[5]);
    l67.as_half2 = __floats2half2_rn(f[6], f[7]);
    logits_int4.x = l01.as_uint32;
    logits_int4.y = l23.as_uint32;
    logits_int4.z = l45.as_uint32;
    logits_int4.w = l67.as_uint32;
    *row_ptr = logits_int4;
}

void QMoEMLP::forward_
(
    cublasHandle_t cublas_handle,
    half* x,
    int rows,
    int columns
//     const std::vector<uintptr_t>& loras,
//     half* lora_temp
)
{
    if (num_experts != 8)
    {
        printf(" ## num_experts != 8 not implemented\n");
        return;
    }

    bool use_half2 = true;

    // Norm

    rms_norm_cuda(x, layernorm, temp_state, norm_epsilon, rows, columns);

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
    softmax8_topk_norm_kernel<<<gridDim, blockDim>>>(temp_logits, rows, 2);

    // For small no. rows, execute all kernels but pass the routing weights. Rows with a weight of zero will skip dot
    // product accum and kernels launched with only zero-weights will exit prematurely.

    if (rows <= MAX_Q_GEMM_WEIGHTS)
    {
        int intermediate_size = w1[0]->width;
        fp_silu_mul_kernel kernel = pick_silu_mul_kernel(use_half2, true);

        for (int i = 0; i < num_experts; i++)
        {
            gemm_half_q_half_cuda(cublas_handle, temp_state, w1[i], temp_a, rows, intermediate_size, columns, true, temp_dq, true, temp_logits + i, num_experts, false);
            gemm_half_q_half_cuda(cublas_handle, temp_state, w3[i], temp_b, rows, intermediate_size, columns, true, temp_dq, true, temp_logits + i, num_experts, false);

            blockDim.x = THREADS_X;
            blockDim.y = THREADS_Y;
            gridDim.x = DIVIDE(intermediate_size, THREADS_X) / (use_half2 ? 2 : 1);
            gridDim.y = DIVIDE(rows, THREADS_Y);
            kernel<<<gridDim, blockDim>>>(temp_a, temp_b, rows, intermediate_size, temp_logits + i, num_experts);

            // print_global_mem(temp_a+14336-32, 1, 32, 14336);

            gemm_half_q_half_cuda(cublas_handle, temp_a, w2[i], x, rows, columns, intermediate_size, false, temp_dq, true, temp_logits + i, num_experts, true);
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
