#include "q_mlp.cuh"
#include "q_gemm.cuh"
#include "rms_norm.cuh"
#include "util.cuh"
#include "matrix_view.cuh"

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
    const int
);

template <bool use_half2>
__global__ void silu_mul_kernel
(
    half* __restrict__ x,
    const half* __restrict__ y,
    const int height,
    const int width
)
{
    MatrixView_half_rw x_(x, height, width);
    MatrixView_half y_(y, height, width);

    int column = (THREADS_X * blockIdx.x + threadIdx.x); if constexpr (use_half2) column *= 2;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= height) return;

    // silu(x) * y

    if constexpr (use_half2)
    {
        half2 one = __half2half2(__float2half(1.0f));

        half2 x_item = x_.item_half2(row, column);
        half2 y_item = y_.item_half2(row, column);

        x_item = silu(x_item);
        x_item = __hmul2(x_item, y_item);

        x_.set_half2(row, column, x_item);
    }
    else
    {
        half one = __float2half(1.0f);

        half x_item = x_.item(row, column);
        half y_item = y_.item(row, column);

        x_item = silu(x_item);
        x_item = __hmul(x_item, y_item);

        x_.set(row, column, x_item);
    }
}

fp_silu_mul_kernel pick_silu_mul_kernel(bool use_half2)
{
    if (use_half2) return silu_mul_kernel<true>;
    else           return silu_mul_kernel<false>;
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

void QMLP::forward_
(
    cublasHandle_t cublas_handle,
    half* x,
    int rows,
    int columns
)
{
    bool use_half2 = true;
    int intermediate_size = gate->width;

    rms_norm_cuda(x, layernorm, temp_state, norm_epsilon, rows, columns);
    gemm_half_q_half_cuda(cublas_handle, temp_state, gate, temp_a, rows, intermediate_size, columns, true, temp_dq);
    gemm_half_q_half_cuda(cublas_handle, temp_state, up,   temp_b, rows, intermediate_size, columns, true, temp_dq);

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = THREADS_Y;
    gridDim.x = DIVIDE(up->width, THREADS_X) / (use_half2 ? 2 : 1);
    gridDim.y = DIVIDE(rows, THREADS_Y);

    fp_silu_mul_kernel kernel = pick_silu_mul_kernel(use_half2);
    kernel<<<gridDim, blockDim>>>(temp_a, temp_b, rows, intermediate_size);

    gemm_half_q_half_cuda(cublas_handle, temp_a, down, x, rows, columns, intermediate_size, false, temp_dq);
}
