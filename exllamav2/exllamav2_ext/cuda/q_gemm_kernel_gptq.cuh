
__forceinline__ __device__ half2 dot22_8(half2(&dq)[4], const half* a_ptr, const half2 g_result)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    return __hadd2(result, g_result);
}

typedef void (*fp_gemm_half_q_half_gptq_kernel)
(
    const half*,
    const uint32_t*,
    const uint32_t*,
    const half*,
    half*,
    const int,
    const int,
    const int,
    const int,
    const int,
    const uint16_t*,
    const int,
    const bool
    );

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_kernel
(
    const half* __restrict__ a,
    const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales,
    half* __restrict__ c,
    const int size_m,
    const int size_n,
    const int size_k,
    const int groups,
    const int groupsize,
    const uint16_t* __restrict__ b_q_perm,
    const int rows_4,
    const bool clear
)
{
    MatrixView_half a_(a, size_m, size_k);
    MatrixView_half_rw c_(c, size_m, size_n);
    MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
    MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

    int t = threadIdx.x;

    // Block

    int offset_n = blockIdx.x * BLOCK_KN_SIZE;
    int offset_m = blockIdx.y * m_count;
    int offset_k = blockIdx.z * BLOCK_KN_SIZE;

    int end_n = min(offset_n + BLOCK_KN_SIZE, size_n);
    int end_m = min(offset_m + m_count, size_m);
    int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
    int n = offset_n + t;

    // Preload block_a

    __shared__ half block_a[m_count][BLOCK_KN_SIZE];

    if (offset_k + t < end_k)
    {
        for (int m = 0; m < m_count; ++m)
        {
            const half* a_ptr = a_.item_ptr(offset_m + m, 0);
            half* block_a_ptr = block_a[m];
            
            half a0;
            if (b_q_perm) a0 = a_ptr[b_q_perm[offset_k + t]];
            else a0 = a_ptr[offset_k + t];
            block_a_ptr[t] = a0;
        }
    }

    // Zero output

    if (n >= size_n) return;

    if (clear && blockIdx.z == 0) // && (threadIdx.x & 1) == 0)
    {
        for (int m = 0; m < m_count; m++)
            *((uint16_t*)c_.item_ptr(offset_m + m, n)) = 0;
    }

    __syncthreads();

    // Find initial group

    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;
    
    // a, b offset

    int qk = offset_k / 32 * 4;

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
    const half* a_ptr = &block_a[0][0];
    int a_stride = BLOCK_KN_SIZE;

    // Initial group

    half2 z1z16[2];
    half2 y1y16[2];
    dequant_4bit_8_prep_zero_scale(b_gptq_qzeros_.item(group, n) + 1, b_gptq_scales_.item(group, n), z1z16, y1y16);

    __syncthreads();

    // Column result

    half2 block_c[m_count] = {};

    // Dequantize and multiply

    int k = offset_k;
    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            dequant_4bit_8_prep_zero_scale(b_gptq_qzeros_.item(group, n) + 1, b_gptq_scales_.item(group, n), z1z16, y1y16);
            nextgroup += groupsize;
        }

        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            half2 dq[4];
            dequant_4bit_8_gptq(b_ptr, dq, z1z16, y1y16, size_n);
            b_ptr += size_n;
            for (int m = 0; m < m_count; m++) block_c[m] = dot22_8(dq, a_ptr + m * a_stride, block_c[m]);
            a_ptr += 8;
        }
        k += 32;
    }

    for (int m = 0; m < m_count; m++) atomicAdd(c_.item_ptr(offset_m + m, n), __hadd(block_c[m].x, block_c[m].y));
}

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(bool first_block, const int m_count)
{
    #if BLOCK_M_SIZE_MAX >= 1
    if (m_count == 1) return gemm_half_q_half_gptq_kernel<true, 1>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 2
    if (m_count == 2) return gemm_half_q_half_gptq_kernel<true, 2>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 3
    if (m_count == 3) return gemm_half_q_half_gptq_kernel<true, 3>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 4
    if (m_count == 4) return gemm_half_q_half_gptq_kernel<true, 4>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 5
    if (m_count == 5) return gemm_half_q_half_gptq_kernel<true, 5>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 6
    if (m_count == 6) return gemm_half_q_half_gptq_kernel<true, 6>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 7
    if (m_count == 7) return gemm_half_q_half_gptq_kernel<true, 7>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 8
    if (m_count == 8) return gemm_half_q_half_gptq_kernel<true, 8>;
    #endif
    return NULL;
}
