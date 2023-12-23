#include "kernel_select.cuh"

fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_2a(const int m_count, bool r_weights, bool mul_r_weights)
{
    // if (!r_weights && !mul_r_weights) return map_m_count_exl2_a<false, false>::pick_gemm_half_q_half_kernel(m_count);
    // if (!r_weights &&  mul_r_weights) return map_m_count_exl2_a<false,  true>::pick_gemm_half_q_half_kernel(m_count);
    if ( r_weights && !mul_r_weights) return map_m_count_exl2_a< true, false>::pick_gemm_half_q_half_kernel(m_count);
    // if ( r_weights &&  mul_r_weights) return map_m_count_exl2_a< true,  true>::pick_gemm_half_q_half_kernel(m_count);
    return NULL;
}
