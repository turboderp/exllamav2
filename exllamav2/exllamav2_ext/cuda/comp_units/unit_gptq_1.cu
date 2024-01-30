#include "kernel_select.cuh"

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel_1(const int max_m, bool r_weights, bool mul_r_weights)
{
    if (!r_weights && !mul_r_weights)
    {
        if (max_m == 1) return map_m_count_gptq<1, false, false>::pick_gemm_half_q_half_gptq_kernel();
        if (max_m == 2) return map_m_count_gptq<2, false, false>::pick_gemm_half_q_half_gptq_kernel();
        if (max_m == 3) return map_m_count_gptq<3, false, false>::pick_gemm_half_q_half_gptq_kernel();
        if (max_m == 4) return map_m_count_gptq<4, false, false>::pick_gemm_half_q_half_gptq_kernel();
    }
    return NULL;
}
