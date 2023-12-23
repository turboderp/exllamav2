#include "kernel_select.cuh"

fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_1b(const int perm, bool r_weights, bool mul_r_weights)
{
    if (!r_weights && !mul_r_weights) return map_m_count_exl2_b<false, false>::pick_gemm_half_q_half_kernel(perm);
    // if (!r_weights &&  mul_r_weights) return map_m_count_exl2_b<false,  true>::pick_gemm_half_q_half_kernel(perm);
    // if ( r_weights && !mul_r_weights) return map_m_count_exl2_b< true, false>::pick_gemm_half_q_half_kernel(perm);
    // if ( r_weights &&  mul_r_weights) return map_m_count_exl2_b< true,  true>::pick_gemm_half_q_half_kernel(perm);
    return NULL;
}
