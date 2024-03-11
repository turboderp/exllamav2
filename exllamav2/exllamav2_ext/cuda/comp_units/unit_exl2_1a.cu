#include "kernel_select.cuh"

fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_1a(const int max_m, const int perm, bool r_weights, bool mul_r_weights, int kn_size)
{
    if (!r_weights && !mul_r_weights)
    {
        if (max_m == 1) return map_m_count_exl2_a<1, false, false>::pick_gemm_half_q_half_kernel(perm, kn_size);
        if (max_m == 2) return map_m_count_exl2_a<2, false, false>::pick_gemm_half_q_half_kernel(perm, kn_size);
        if (max_m == 3) return map_m_count_exl2_a<3, false, false>::pick_gemm_half_q_half_kernel(perm, kn_size);
        if (max_m == 4) return map_m_count_exl2_a<4, false, false>::pick_gemm_half_q_half_kernel(perm, kn_size);
    }
    return NULL;
}
