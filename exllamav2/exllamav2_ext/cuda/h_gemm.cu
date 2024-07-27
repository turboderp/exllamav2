#include "h_gemm.cuh"
#include "util.cuh"
#include "../config.h"
#include "matrix_view.cuh"

// union half2_uint32
// {
//     uint32_t as_uint32;
//     half2 as_half2;
//     __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
//     __device__ half2_uint32(half2 val) : as_half2(val) {}
// };

// TODO: Improve tall kernel, maybe special cases for size_n = 1, 2, 4, 8, 16

const int T_THREADS_M = 1;
const int T_THREADS_N = 8;
const int T_BLOCKSIZE_K = 32;
const int T_MAX_M = 16;
const int T_MAX_N = 64;
const int T_MAX_K = 1024 / T_THREADS_N * T_BLOCKSIZE_K;
const int T_MAX_BLOCKS_K = T_MAX_K / T_BLOCKSIZE_K;

__global__ void h_gemm_tall_kernel
(
    const int size_m,
    const int size_n,
    const int size_k,
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    bool clear
)
{
    __shared__ half accum[T_MAX_BLOCKS_K][T_THREADS_N];

    int m = blockIdx.y * T_THREADS_M + threadIdx.z;
    int n = blockIdx.x * T_THREADS_N + threadIdx.x;
    int k = threadIdx.y * T_BLOCKSIZE_K;

    if (n >= size_n) return;
    if (m >= size_m) return;

    MatrixView_half a_(a, size_m, size_k);
    MatrixView_half b_(b, size_k, size_n);
    MatrixView_half_rw c_(c, size_m, size_n);

    int k_end = min(k + T_BLOCKSIZE_K, size_k);

    const half* a_ptr = a_.item_ptr(m, k);
    const half* a_ptr_end = a_.item_ptr(m, k_end);
    const half* b_ptr = b_.item_ptr(k, n);
    half* c_ptr = c_.item_ptr(m, n);

    half2 r2 = {};

    while(a_ptr <= a_ptr_end - 8)
    {
        int4 a_int4 = *((int4*) a_ptr);
        half2 a_01 = ((half2_uint32) a_int4.x).as_half2;
        half2 a_23 = ((half2_uint32) a_int4.y).as_half2;
        half2 a_45 = ((half2_uint32) a_int4.z).as_half2;
        half2 a_67 = ((half2_uint32) a_int4.w).as_half2;
        a_ptr += 8;

        half b_0 = *b_ptr; b_ptr += size_n;
        half b_1 = *b_ptr; b_ptr += size_n;
        half b_2 = *b_ptr; b_ptr += size_n;
        half b_3 = *b_ptr; b_ptr += size_n;
        half b_4 = *b_ptr; b_ptr += size_n;
        half b_5 = *b_ptr; b_ptr += size_n;
        half b_6 = *b_ptr; b_ptr += size_n;
        half b_7 = *b_ptr; b_ptr += size_n;
        half2 b_01 = __halves2half2(b_0, b_1);
        half2 b_23 = __halves2half2(b_2, b_3);
        half2 b_45 = __halves2half2(b_4, b_5);
        half2 b_67 = __halves2half2(b_6, b_7);

        r2 = __hfma2(a_01, b_01, r2);
        r2 = __hfma2(a_23, b_23, r2);
        r2 = __hfma2(a_45, b_45, r2);
        r2 = __hfma2(a_67, b_67, r2);
    }

    while(a_ptr <= a_ptr_end - 4)
    {
        int2 a_int2 = *((int2*) a_ptr);
        half2 a_01 = ((half2_uint32) a_int2.x).as_half2;
        half2 a_23 = ((half2_uint32) a_int2.y).as_half2;
        a_ptr += 4;

        half b_0 = *b_ptr; b_ptr += size_n;
        half b_1 = *b_ptr; b_ptr += size_n;
        half b_2 = *b_ptr; b_ptr += size_n;
        half b_3 = *b_ptr; b_ptr += size_n;
        half2 b_01 = __halves2half2(b_0, b_1);
        half2 b_23 = __halves2half2(b_2, b_3);

        r2 = __hfma2(a_01, b_01, r2);
        r2 = __hfma2(a_23, b_23, r2);
    }

    half r = __hadd(__low2half(r2), __high2half(r2));

    while(a_ptr < a_ptr_end)
    {
        half a_item = *a_ptr++;
        half b_item = *b_ptr; b_ptr += size_n;
        r = __hfma(a_item, b_item, r);
    }

    accum[threadIdx.y][threadIdx.x] = r;
    __syncthreads();

    if (threadIdx.y == 0)
    {
        half acc = accum[0][threadIdx.x];
        for (int i = 1; i < blockDim.y; ++i) acc = __hadd(accum[i][threadIdx.x], acc);
        if (!clear) acc = __hadd(acc, *c_ptr);
        *c_ptr = acc;
    }
}


const int W_MAX_M = 16;
const int W_MAX_N = 65536;
const int W_MAX_K = 32;
const int W_THREADS_M = 1;
const int W_THREADS_N = 32;

__global__ void h_gemm_wide_kernel
(
    const int size_m,
    const int size_n,
    const int size_k,
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    bool clear
)
{
    int m = blockIdx.y * W_THREADS_M + threadIdx.y;
    int n = blockIdx.x * W_THREADS_N + threadIdx.x;

    if (m >= size_m) return;

    MatrixView_half a_(a, size_m, size_k);
    MatrixView_half b_(b, size_k, size_n);
    MatrixView_half_rw c_(c, size_m, size_n);

    half* c_ptr = c_.item_ptr(m, n);

    __shared__ half read_a[W_MAX_K];
    int t = threadIdx.x;

    if (t < size_k)
    {
        read_a[t] = a_.item(m, t);
    }
    __syncthreads();

    if (n >= size_n) return;

    half r = {};

    for (int k = 0; k < size_k; ++k)
    {
        half item_a = read_a[k];
        half item_b = b_.item(k, n);
        r = __hfma(item_a, item_b, r);
    }

    if (threadIdx.y == 0)
    {
        if (!clear) r = __hadd(r, *c_ptr);
        *c_ptr = r;
    }
}


// cuBLAS

void h_gemm_cublas
(
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    const int size_m,
    const int size_n,
    const int size_k,
    const half* a,
    const half* b,
    half* c,
    const float alpha,
    const float beta
)
{
    half alpha_ = __float2half(alpha);
    half beta_ = __float2half(beta);
    cublasSetStream(cublas_handle, stream);
    cublasHgemm(cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                size_n, size_m, size_k,
                &alpha_, b, size_n,
                         a, size_k,
                &beta_,  c, size_n);
}


// alpha * ( a[m,k] @ b[k,n] ) + beta * c[m,n] -> c[m,n]

void h_gemm_cuda
(
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    const int size_m,
    const int size_n,
    const int size_k,
    const half* a,
    const half* b,
    half* c,
    const float alpha,
    const float beta
)
{
    if ((beta == 1.0f || beta == 0.0f) && (alpha == 1.0f))
    {
        bool clear = (beta == 0.0f);

        //DBGI3(size_m, size_n, size_k);

        if (size_m <= T_MAX_M && size_n <= T_MAX_N && size_k <= T_MAX_K)
        {
            // Tall

            dim3 blockDim, gridDim;
            blockDim.x = T_THREADS_N;
            blockDim.y = DIVIDE(size_k, T_BLOCKSIZE_K);
            blockDim.z = T_THREADS_M;
            gridDim.x = DIVIDE(size_n, T_THREADS_N);
            gridDim.y = DIVIDE(size_m, T_THREADS_M);
            gridDim.z = 1;

//             DBGI3(blockDim.x, blockDim.y, blockDim.z);
//             DBGI3(gridDim.x, gridDim.y, gridDim.z);

            h_gemm_tall_kernel<<<gridDim, blockDim, 0, stream>>>(size_m, size_n, size_k, a, b, c, clear);
            cuda_check( cudaPeekAtLastError() );
            return;
        }

        if (size_m <= W_MAX_M && size_n <= W_MAX_N && size_k <= W_MAX_K)
        {
            // Wide

            dim3 blockDim, gridDim;
            blockDim.x = W_THREADS_N;
            blockDim.y = W_THREADS_M;
            blockDim.z = 1;
            gridDim.x = DIVIDE(size_n, W_THREADS_N);
            gridDim.y = DIVIDE(size_m, W_THREADS_M);
            gridDim.z = 1;

//             DBGI3(blockDim.x, blockDim.y, blockDim.z);
//             DBGI3(gridDim.x, gridDim.y, gridDim.z);

            h_gemm_wide_kernel<<<gridDim, blockDim, 0, stream>>>(size_m, size_n, size_k, a, b, c, clear);
            cuda_check( cudaPeekAtLastError() );
            return;
        }
    }

    h_gemm_cublas(stream, cublas_handle, size_m, size_n, size_k, a, b, c, alpha, beta);
//     DBGI3(size_m, size_n, size_k);
    cuda_check( cudaPeekAtLastError() );

}