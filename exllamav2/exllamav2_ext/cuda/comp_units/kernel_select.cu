#include "kernel_select.cuh"

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(const int m_count, bool r_weights, bool mul_r_weights)
{
    fp_gemm_half_q_half_gptq_kernel k;
    k = pick_gemm_half_q_half_gptq_kernel_1(m_count, r_weights, mul_r_weights); if (k) return k;
    k = pick_gemm_half_q_half_gptq_kernel_2(m_count, r_weights, mul_r_weights); if (k) return k;
    k = pick_gemm_half_q_half_gptq_kernel_3(m_count, r_weights, mul_r_weights);        return k;
}

fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel(const int m_count, bool r_weights, bool mul_r_weights)
{
    fp_gemm_half_q_half_kernel k;
    k = pick_gemm_half_q_half_kernel_1(m_count, r_weights, mul_r_weights); if (k) return k;
    k = pick_gemm_half_q_half_kernel_2(m_count, r_weights, mul_r_weights); if (k) return k;
    k = pick_gemm_half_q_half_kernel_3(m_count, r_weights, mul_r_weights);        return k;
}

