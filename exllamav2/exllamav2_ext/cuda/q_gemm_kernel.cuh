
__forceinline__ __device__ half2 dot22_8(half2(&dq)[4], const half* a_ptr, const half2 g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_16(half2(&dq)[8], const half* a_ptr, const half2 g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_32(half2(&dq)[16], const half* a_ptr, const half2 g_result, const half qs_h)
{
    half2 result = {};
    const half2* a2_ptr = (const half2*)a_ptr;
    #pragma unroll
    for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

typedef void (*fp_gemm_half_q_half_kernel)
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
    const int,
    const int,
    const int,
    const int,
    const int,
    const bool
);

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_kernel
(
    const half*      __restrict__ a,
    const uint32_t*  __restrict__ b_q_weight,
    const uint32_t*  __restrict__ b_q_scale,
    const half*      __restrict__ b_q_scale_max,
    half*            __restrict__ c,
    const int size_m,
    const int size_n,
    const int size_k,
    const int groups,
    const int groupsize,
    const uint16_t* __restrict__ b_q_perm,
    const int rows_8,
    const int rows_6,
    const int rows_5,
    const int rows_4,
    const int rows_3,
    const int rows_2,
    const bool clear
)
{
    MatrixView_half a_(a, size_m, size_k);
    MatrixView_half_rw c_(c, size_m, size_n);
    MatrixView_q4_row b_q_scale_(b_q_scale, groups, size_n);

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
            half a0 = a_ptr[b_q_perm[offset_k + t]];
            block_a_ptr[t] = a0;
        }
    }

    // Clear

    if (n >= size_n) return;

    if (clear && blockIdx.z == 0) // && (threadIdx.x & 1) == 0)
    {
        for (int m = 0; m < m_count; m++)
            *((uint16_t*) c_.item_ptr(offset_m + m, n)) = 0;
    }

    __syncthreads();

    // Find initial group

    int group = offset_k / groupsize;

    // Preload scales

    half scales[MAX_GROUPS_IN_BLOCK];

    int groups_in_block = DIVIDE((end_k - offset_k), groupsize);
    for (int g = 0; g < groups_in_block; g++)
    {
        half s = dq_scale(b_q_scale_.item(group + g, n), b_q_scale_max[group + g]);
        scales[g] = s;
    }

    // a, b offset

    int pre_rows_8 = min(rows_8, offset_k);
    int pre_rows_6 = offset_k > rows_8 ? min(rows_6, offset_k) - rows_8 : 0;
    int pre_rows_5 = offset_k > rows_6 ? min(rows_5, offset_k) - rows_6 : 0;
    int pre_rows_4 = offset_k > rows_5 ? min(rows_4, offset_k) - rows_5 : 0;
    int pre_rows_3 = offset_k > rows_4 ? min(rows_3, offset_k) - rows_4 : 0;
    int pre_rows_2 = offset_k > rows_3 ? min(rows_2, offset_k) - rows_3 : 0;
    int qk = 0;
    qk += pre_rows_8 / 32 * 8;
    qk += pre_rows_6 / 32 * 6;
    qk += pre_rows_5 / 32 * 5;
    qk += pre_rows_4 / 32 * 4;
    qk += pre_rows_3 / 32 * 3;
    qk += pre_rows_2 / 32 * 2;

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
    const half* a_ptr = &block_a[0][0];
    int a_stride = BLOCK_KN_SIZE;

    // Initial group

    half qs_h = scales[0];
    int scales_idx = 0;
    int nextgroup = offset_k + groupsize;

    // Column result

    half2 block_c[m_count] = {};

    // Dequantize groups

    int k = offset_k;

    while (k < rows_8 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h = scales[scales_idx];
            nextgroup += groupsize;
        }

        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            half2 dq[4];
            dequant_8bit_8(b_ptr, dq, size_n); b_ptr += 2 * size_n;
            for (int m = 0; m < m_count; m++) block_c[m] = dot22_8(dq, a_ptr + m * a_stride, block_c[m], qs_h);
            a_ptr += 8;
        }
        k += 32;
    }

    while (k < rows_6 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h = scales[scales_idx];
            nextgroup += groupsize;
        }

        #pragma unroll
        for (int j = 0; j < 2; j++)
        {
            half2 dq[8];
            dequant_6bit_16(b_ptr, dq, size_n); b_ptr += size_n * 3;
            for (int m = 0; m < m_count; m++) block_c[m] = dot22_16(dq, a_ptr + m * a_stride, block_c[m], qs_h);
            a_ptr += 16;
        }
        k += 32;
    }

    while (k < rows_5 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h = scales[scales_idx];
            nextgroup += groupsize;
        }

        half2 dq[16];
        dequant_5bit_32(b_ptr, dq, size_n); b_ptr += 5 * size_n;
        for (int m = 0; m < m_count; m++) block_c[m] = dot22_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);
        a_ptr += 32;
        k += 32;
    }

    while (k < rows_4 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h = scales[scales_idx];
            nextgroup += groupsize;
        }

        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            half2 dq[4];
            dequant_4bit_8(b_ptr, dq, size_n); b_ptr += size_n;
            for (int m = 0; m < m_count; m++) block_c[m] = dot22_8(dq, a_ptr + m * a_stride, block_c[m], qs_h);
            a_ptr += 8;
        }
        k += 32;
    }

    while (k < rows_3 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h = scales[scales_idx];
            nextgroup += groupsize;
        }

        half2 dq[16];
        dequant_3bit_32(b_ptr, dq, size_n); b_ptr += 3 * size_n;
        for (int m = 0; m < m_count; m++) block_c[m] = dot22_32(dq, a_ptr + m * a_stride, block_c[m], qs_h);
        a_ptr += 32;
        k += 32;
    }

    while (k < rows_2 && k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            scales_idx++;
            qs_h = scales[scales_idx];
            nextgroup += groupsize;
        }

        #pragma unroll
        for (int j = 0; j < 2; j++)
        {
            half2 dq[8];
            dequant_2bit_16(b_ptr, dq, size_n); b_ptr += size_n;
            for (int m = 0; m < m_count; m++) block_c[m] = dot22_16(dq, a_ptr + m * a_stride, block_c[m], qs_h);
            a_ptr += 16;
        }
        k += 32;
    }

    // Accumulate column sums in c

    for (int m = 0; m < m_count; m++) atomicAdd(c_.item_ptr(offset_m + m, n), __hadd(block_c[m].x, block_c[m].y));
//     for (int m = 0; m < m_count; m++) c_.set(offset_m + m, n, __hadd(block_c[m].x, block_c[m].y));
}

fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel(bool first_block, const int m_count)
{
    #if BLOCK_M_SIZE_MAX >= 1
    if (m_count == 1) return gemm_half_q_half_kernel<true, 1>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 2
    if (m_count == 2) return gemm_half_q_half_kernel<true, 2>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 3
    if (m_count == 3) return gemm_half_q_half_kernel<true, 3>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 4
    if (m_count == 4) return gemm_half_q_half_kernel<true, 4>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 5
    if (m_count == 5) return gemm_half_q_half_kernel<true, 5>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 6
    if (m_count == 6) return gemm_half_q_half_kernel<true, 6>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 7
    if (m_count == 7) return gemm_half_q_half_kernel<true, 7>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 8
    if (m_count == 8) return gemm_half_q_half_kernel<true, 8>;
    #endif
    return NULL;
}
