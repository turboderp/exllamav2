#ifndef _kernel_select_cuh
#define _kernel_select_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "../q_gemm_kernel.cuh"
#include "../q_gemm_kernel_gptq.cuh"

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel_1(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel_2(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel_3(const int m_count, bool r_weights, bool mul_r_weights);

fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_1(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_2(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_3(const int m_count, bool r_weights, bool mul_r_weights);

template <bool use_r_weights, bool mul_r_weights>
struct map_m_count_gptq {
    static constexpr fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(int m_count)
    {
        #if GPTQ_BLOCK_M_SIZE_MAX >= 1
        if (m_count == 1) return gemm_half_q_half_gptq_kernel<1, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 2
        if (m_count == 2) return gemm_half_q_half_gptq_kernel<2, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 3
        if (m_count == 3) return gemm_half_q_half_gptq_kernel<3, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 4
        if (m_count == 4) return gemm_half_q_half_gptq_kernel<4, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 5
        if (m_count == 5) return gemm_half_q_half_gptq_kernel<5, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 6
        if (m_count == 6) return gemm_half_q_half_gptq_kernel<6, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 7
        if (m_count == 7) return gemm_half_q_half_gptq_kernel<7, use_r_weights, mul_r_weights>;
        #endif
        #if GPTQ_BLOCK_M_SIZE_MAX >= 8
        if (m_count == 8) return gemm_half_q_half_gptq_kernel<8, use_r_weights, mul_r_weights>;
        #endif
        return NULL;
    }
};

template <bool use_r_weights, bool mul_r_weights>
struct map_m_count_exl2 {
    static constexpr fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel(const int m_count)
    {
        #if EXL2_BLOCK_M_SIZE_MAX >= 1
        if (m_count == 1) return gemm_half_q_half_kernel<1, use_r_weights, mul_r_weights>;
        #endif
        #if EXL2_BLOCK_M_SIZE_MAX >= 2
        if (m_count == 2) return gemm_half_q_half_kernel<2, use_r_weights, mul_r_weights>;
        #endif
        #if EXL2_BLOCK_M_SIZE_MAX >= 3
        if (m_count == 3) return gemm_half_q_half_kernel<3, use_r_weights, mul_r_weights>;
        #endif
        #if EXL2_BLOCK_M_SIZE_MAX >= 4
        if (m_count == 4) return gemm_half_q_half_kernel<4, use_r_weights, mul_r_weights>;
        #endif
        #if EXL2_BLOCK_M_SIZE_MAX >= 5
        if (m_count == 5) return gemm_half_q_half_kernel<5, use_r_weights, mul_r_weights>;
        #endif
        #if EXL2_BLOCK_M_SIZE_MAX >= 6
        if (m_count == 6) return gemm_half_q_half_kernel<6, use_r_weights, mul_r_weights>;
        #endif
        #if EXL2_BLOCK_M_SIZE_MAX >= 7
        if (m_count == 7) return gemm_half_q_half_kernel<7, use_r_weights, mul_r_weights>;
        #endif
        #if EXL2_BLOCK_M_SIZE_MAX >= 8
        if (m_count == 8) return gemm_half_q_half_kernel<8, use_r_weights, mul_r_weights>;
        #endif
        return NULL;
    }
};

#endif