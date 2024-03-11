#include "kernel_select.cuh"
#include "../util.cuh"

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(const int max_m, bool r_weights, bool mul_r_weights)
{
    fp_gemm_half_q_half_gptq_kernel k;
    k = pick_gemm_half_q_half_gptq_kernel_1(max_m, r_weights, mul_r_weights); if (k) return k;
    k = pick_gemm_half_q_half_gptq_kernel_2(max_m, r_weights, mul_r_weights); if (k) return k;
    k = pick_gemm_half_q_half_gptq_kernel_3(max_m, r_weights, mul_r_weights);        return k;
}

fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel(const int max_m, const int perm, bool r_weights, bool mul_r_weights, int kn_size)
{
    fp_gemm_half_q_half_kernel k;
    k = pick_gemm_half_q_half_kernel_1a(max_m, perm, r_weights, mul_r_weights, kn_size); if (k) return k;
    k = pick_gemm_half_q_half_kernel_1b(max_m, perm, r_weights, mul_r_weights, kn_size); if (k) return k;
    k = pick_gemm_half_q_half_kernel_2a(max_m, perm, r_weights, mul_r_weights, kn_size); if (k) return k;
    k = pick_gemm_half_q_half_kernel_2b(max_m, perm, r_weights, mul_r_weights, kn_size); if (k) return k;
    k = pick_gemm_half_q_half_kernel_3a(max_m, perm, r_weights, mul_r_weights, kn_size); if (k) return k;
    k = pick_gemm_half_q_half_kernel_3b(max_m, perm, r_weights, mul_r_weights, kn_size);        return k;
}
