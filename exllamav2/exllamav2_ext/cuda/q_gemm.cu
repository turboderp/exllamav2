#include "q_gemm.cuh"
#include "util.cuh"
#include "../config.h"

#define CLEAR_N_SIZE 256

#include "comp_units/kernel_select.cuh"
#include "q_gemm_autotune.cuh"
#include "h_add.cuh"

void gemm_half_q_half_cuda_part
(
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    int m_count,
    bool clear,
    const half* r_weights,
    int r_weights_stride,
    bool mul_r_weights
)
{
    if (!b->is_gptq)
    {
        int block_kn_size;
        bool measure;
        AT_Result* atr;
        cudaEvent_t start, stop;

        bool use_autotune = true;

        if (!use_autotune)
        {
            block_kn_size = at_get_fallback_blocksize(b->device, size_m, size_n, size_k);
        }
        else
        {
            // Only autotune up to EXL2_BLOCK_M_SIZE_MAX

            if (size_m > EXL2_BLOCK_M_SIZE_MAX)
            {
                atr = at_get_top(b->device, size_k, size_n);
                if (atr && atr->best) block_kn_size = atr->best;
                else block_kn_size = at_get_fallback_blocksize(b->device, size_m, size_n, size_k);
                measure = false;
            }

            // Use autotuned size or prepare measurement

            else
            {
                atr = at_get(b->device, size_m, size_k, size_n);
                if (atr->best)
                {
                    block_kn_size = atr->best;
                    measure = false;
                }
                else
                {
                    measure = true;
                    int c32 = atr->timings_32.size();
                    int c64 = atr->timings_64.size();
                    if (c32 + c64 == AT_NUM_MEASURE)
                    {
                        at_select(atr);
                        block_kn_size = atr->best;
                        measure = false;
                    }
                    else
                    {
                        block_kn_size = c32 < c64 ? 32 : 64;
                        measure = true;
                    }
                }
            }
        }

        // Prepare kernel

        dim3 blockDim, gridDim;
        blockDim.x = block_kn_size;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = DIVIDE(size_n, block_kn_size * 4);
        gridDim.y = DIVIDE(size_m, m_count);
        gridDim.z = DIVIDE(size_k, block_kn_size);

        int max_m = min(EXL2_BLOCK_M_SIZE_MAX, size_m);

        fp_gemm_half_q_half_kernel kernel = pick_gemm_half_q_half_kernel(max_m, b->kernel_p, r_weights != NULL, mul_r_weights, block_kn_size);
        if (!kernel) return;

        // Measurement events

        if (measure)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        }

        // Launch kernel

        kernel<<<gridDim, blockDim>>>
        (
            a,
            b->cuda_q_weight,
            b->cuda_q_scale,
            b->cuda_q_scale_max,
            c,
            size_m,
            size_n,
            size_k,
            b->groups,
            b->cuda_q_group_map,
            b->cuda_q_perm,
            b->rows_8,
            b->rows_6,
            b->rows_5,
            b->rows_4,
            b->rows_3,
            b->rows_2,
            clear,
            r_weights,
            r_weights_stride
        );

        // Finish measurement

        if (measure)
        {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float timing = 0.0f;
            cudaEventElapsedTime(&timing, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            if (block_kn_size == 32) atr->timings_32.push_back(timing);
            if (block_kn_size == 64) atr->timings_64.push_back(timing);
        }
    }

    // GPTQ kernel

    else
    {
        dim3 blockDim, gridDim;
        blockDim.x = GPTQ_BLOCK_KN_SIZE;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = DIVIDE(size_n, GPTQ_BLOCK_KN_SIZE * 4);
        gridDim.y = DIVIDE(size_m, GPTQ_BLOCK_M_SIZE_MAX);
        gridDim.z = DIVIDE(size_k, GPTQ_BLOCK_KN_SIZE);

        int max_m = min(GPTQ_BLOCK_M_SIZE_MAX, size_m);

        fp_gemm_half_q_half_gptq_kernel kernel = pick_gemm_half_q_half_gptq_kernel(GPTQ_BLOCK_M_SIZE_MAX, r_weights != NULL, mul_r_weights);
        if (!kernel) return;

//         DBGX((uint64_t) r_weights);
//         if (r_weights)
//             print_global_mem(r_weights, 1, 1, 1);
//         DBGI(r_weights_stride);

        kernel<<<gridDim, blockDim>>>
        (
            a,
            b->cuda_q_weight,
            b->cuda_gptq_qzeros,
            b->cuda_gptq_scales,
            c,
            size_m,
            size_n,
            size_k,
            b->groups,
            b->gptq_groupsize,
            b->cuda_q_perm,
            b->rows_4,
            clear,
            r_weights,
            r_weights_stride
        );
    }
}

void gemm_half_q_half_cuda
(
    cublasHandle_t cublas_handle,
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    bool clear,
    half* temp_dq,
    bool force_cuda,
    const half* r_weights,
    const int r_weights_stride,
    bool mul_r_weights
)
{
    // Here we force CUDA matmul for matrices that are too big to dequantize. This is necessary for the
    // extremely large output layers of some models. Splitting along K and dequantizing/multiplying in
    // chunks would work also except the remapping of EXL2 matrices complicates it.
    //
    // TODO: Finish the chunking stuff

    int row_step = (b->max_dq_rows / 128) * 128;

    if (size_m > MAX_Q_GEMM_ROWS && !force_cuda && size_k <= row_step)
    {
        int row_b = 0;
        if (row_step == 0) row_step = size_k;

        while (row_b < size_k)
        {
            int row_a = row_b;
            row_b += row_step;
            row_b = min(row_b, size_k);
            int chunk_k = row_b - row_a;

            // Reconstruct FP16 matrix, then cuBLAS

            if (!temp_dq) temp_dq = b->temp_dq;
            b->reconstruct(temp_dq, row_a, row_b);

            const half alpha = __float2half(1.0f);
            const half beta = (clear && row_a == 0) ? __float2half(0.0f) : __float2half(1.0f);
            cublasHgemm(cublas_handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        size_n, size_m, chunk_k,
                        &alpha, temp_dq,   size_n,
                                a + row_a, size_k,
                        &beta,  c,         size_n);

//            cublasHgemm(cublas_handle,
//                        CUBLAS_OP_N,
//                        CUBLAS_OP_N,
//                        size_n, size_m, size_k,
//                        &alpha, temp_dq, size_n,
//                                a,       size_k,
//                        &beta,  c,       size_n);
        }

        //const float alpha = 1.0f;
        //const float beta = clear ? 0.0f : 1.0f;
        //cublasSgemmEx(cublas_handle,
        //             CUBLAS_OP_N,
        //             CUBLAS_OP_N,
        //             size_n, size_m, size_k,
        //             &alpha, temp_dq, CUDA_R_16F, size_n,
        //                     a,       CUDA_R_16F, size_k,
        //             &beta,  c,       CUDA_R_16F, size_n);

        //const float alpha = 1.0f;
        //const float beta = clear ? 0.0f : 1.0f;
        //cublasGemmEx(cublas_handle,
        //             CUBLAS_OP_N, CUBLAS_OP_N,
        //             size_n, size_m, size_k,
        //             &alpha, temp_dq, CUDA_R_16F, size_n,
        //                     a,       CUDA_R_16F, size_k,
        //             &beta,  c,       CUDA_R_16F, size_n,
        //             CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP);
    }
    else
    {
        // Quantized matmul

        int block_m_size_max = b->is_gptq ? GPTQ_BLOCK_M_SIZE_MAX : EXL2_BLOCK_M_SIZE_MAX;
        int block_m = min(size_m, block_m_size_max);
        gemm_half_q_half_cuda_part(a, b, c, size_m, size_n, size_k, block_m, clear, r_weights, r_weights_stride, mul_r_weights);
    }

    if (b->cuda_bias) cuda_vector_add_(c, b->cuda_bias, size_m, size_n);
}

__global__ void clear_kernel
(
    half* __restrict__ c,
    const int size_m,
    const int size_n
)
{
    int m = blockIdx.y;
    int n = (blockIdx.x * CLEAR_N_SIZE + threadIdx.x) * 8;
    if (n >= size_n) return;
    int4* c_ptr = (int4*)(c + m * size_n + n);
    *c_ptr = {};
}

void clear_tensor_cuda
(
    half* c,
    int size_m,
    int size_n
)
{
//     dim3 blockDim, gridDim;
//     blockDim.x = CLEAR_N_SIZE;
//     blockDim.y = 1;
//     gridDim.x = DIVIDE(size_n / 8, CLEAR_N_SIZE);
//     gridDim.y = size_m;
//     clear_kernel<<<gridDim, blockDim>>>(c, size_m, size_n);
}
