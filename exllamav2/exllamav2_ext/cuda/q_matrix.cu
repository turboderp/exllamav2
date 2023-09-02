#include "q_matrix.cuh"
#include "matrix_view.cuh"
#include "util.cuh"
#include "q_gemm_dq.cuh"

#include "quant/qdq_2.cuh"
#include "quant/qdq_3.cuh"
#include "quant/qdq_4.cuh"

#define BLOCK_KN_SIZE 256

#define THREADS_X 32
#define THREADS_Y 32

// Shuffle quantized data on load

__global__ void shuffle_kernel
(
    uint32_t* __restrict__ b_q_weight,
    const int size_k,
    const int size_n,
    const int rows_8,
    const int rows_6,
    const int rows_5,
    const int rows_4,
    const int rows_3,
    const int rows_2
)
{
    int n = blockIdx.x * THREADS_X + threadIdx.x;
    if (n >= size_n) return;
    int k = 0;
    uint32_t* b_ptr = b_q_weight + n;
    while (k < rows_8) { b_ptr += size_n; k += 4; }
    while (k < rows_6) { b_ptr += 6 * size_n; k += 32; }
    while (k < rows_5) { b_ptr += 5 * size_n; k += 32; }
//     while (k < rows_4) {                                 b_ptr += 1 * size_n; k +=  8; }
    while (k < rows_4) { shuffle_4bit_8 (b_ptr, size_n); b_ptr += 1 * size_n; k +=  8; }
    while (k < rows_3) { shuffle_3bit_32(b_ptr, size_n); b_ptr += 3 * size_n; k += 32; }
    while (k < rows_2) { shuffle_2bit_16(b_ptr, size_n); b_ptr += 1 * size_n; k += 16; }
}


// QMatrix constructor

QMatrix::QMatrix
(
    const int _device,
    const int _height,
    const int _width,
    const int _groups,

    uint32_t* _q_weight,
    uint16_t* _q_perm,
    uint16_t* _q_invperm,
    uint32_t* _q_scale,
    half* _q_scale_max,
    uint16_t* _q_groups,

    half* _temp_dq
) :
    device(_device),
    height(_height),
    width(_width),
    groups(_groups),
    temp_dq(_temp_dq)
{
    cudaSetDevice(device);

    cuda_q_weight = _q_weight;
    cuda_q_perm = _q_perm;
    cuda_q_invperm = _q_invperm;
    cuda_q_scale = _q_scale;
    cuda_q_scale_max = _q_scale_max;
    cuda_q_groups = _q_groups;

    groupsize = 1;
    while (groupsize * groups < height) groupsize *= 2;

    // Create group map

    rows_8 = 0;
    rows_6 = 0;
    rows_5 = 0;
    rows_4 = 0;
    rows_3 = 0;
    rows_2 = 0;

    uint16_t* cpu_q_groups = (uint16_t*) calloc(groups * 2, sizeof(uint16_t));
    cudaMemcpy(cpu_q_groups, cuda_q_groups, groups * 2 * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < groups; i++)
    {
        int bits = cpu_q_groups[i * 2];
        if (bits == 8) rows_8 += groupsize;
        if (bits == 6) rows_6 += groupsize;
        if (bits == 5) rows_5 += groupsize;
        if (bits == 4) rows_4 += groupsize;
        if (bits == 3) rows_3 += groupsize;
        if (bits == 2) rows_2 += groupsize;
    }

    free(cpu_q_groups);

    rows_6 += rows_8;
    rows_5 += rows_6;
    rows_4 += rows_5;
    rows_3 += rows_4;
    rows_2 += rows_3;

    // printf("-------------\n");
    // DBGI(rows_8);
    // DBGI(rows_6);
    // DBGI(rows_5);
    // DBGI(rows_4);
    // DBGI(rows_3);
    // DBGI(rows_2);

    // Shuffle quantized data

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, THREADS_X);
    gridDim.y = 1;

    shuffle_kernel<<<gridDim, blockDim>>>(cuda_q_weight, height, width, rows_8, rows_6, rows_5, rows_4, rows_3, rows_2);
}


// Reconstruct b[k,n]

__global__ void reconstruct_kernel
(
    const uint32_t* __restrict__ b_q_weight,
    const uint16_t* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_q_scale,
    const half* __restrict__ b_q_scale_max,
    const uint16_t* __restrict__ b_q_groups,
    const int size_k,
    const int size_n,
    const int groupsize,
    const int groups,
    half* __restrict__ b
)
{
    MatrixView_half_rw b_(b, size_k, size_n);
    MatrixView_q4_row b_q_scale_(b_q_scale, groups, size_n);

    int offset_k = BLOCK_KN_SIZE * blockIdx.y;

    // Preload remapping table

    __shared__ uint16_t perm[BLOCK_KN_SIZE];
    if (offset_k + threadIdx.x < size_k)
        perm[threadIdx.x] = b_q_perm[offset_k + threadIdx.x];
        //perm[threadIdx.x] = offset_k + threadIdx.x;

    // Column

    int n = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;
    if (n >= size_n) return;

    // Find initial group

    int group = offset_k / groupsize;
    int bits = b_q_groups[group * 2];
    int qk = b_q_groups[group * 2 + 1];
    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

    half qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]);
    int nextgroup = groupsize;

    int end_k = min(BLOCK_KN_SIZE, size_k - offset_k);

    __syncthreads();

    int k = 0;
    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            bits = b_q_groups[group * 2];
            qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]);
            nextgroup += groupsize;
//             if (group == 10 && n < 7)
//             {
//                 DBGI3(group, n, b_q_scale_.item(group, n));
//             }
        }

        switch(bits)
        {
            case 2: for (int p = 0; p < 2; p++) { half dq[16]; dq_2bit_16(b_ptr, dq, size_n); b_ptr += size_n    ; for (int j = 0; j < 16; j++) b_.set(perm[k++], n, __hmul(dq[j], qs_h)); } break;
            case 3:                             { half dq[32]; dq_3bit_32(b_ptr, dq, size_n); b_ptr += size_n * 3; for (int j = 0; j < 32; j++) b_.set(perm[k++], n, __hmul(dq[j], qs_h)); } break;
            case 4: for (int p = 0; p < 4; p++) { half dq[8];  dq_4bit_8 (b_ptr, dq, size_n); b_ptr += size_n    ; for (int j = 0; j <  8; j++) b_.set(perm[k++], n, __hmul(dq[j], qs_h)); } break;
            case 5:                             { half dq[32]; dq_5bit_32(b_ptr, dq, size_n); b_ptr += size_n * 5; for (int j = 0; j < 32; j++) b_.set(perm[k++], n, __hmul(dq[j], qs_h)); } break;
            case 6: for (int p = 0; p < 2; p++) { half dq[16]; dq_6bit_16(b_ptr, dq, size_n); b_ptr += size_n * 3; for (int j = 0; j < 16; j++) b_.set(perm[k++], n, __hmul(dq[j], qs_h)); } break;
            case 8: for (int p = 0; p < 8; p++) { half dq[4];  dq_8bit_4 (b_ptr, dq, size_n); b_ptr += size_n    ; for (int j = 0; j <  4; j++) b_.set(perm[k++], n, __hmul(dq[j], qs_h)); } break;
//             case 2: for (int p = 0; p < 2; p++) { half dq[16]; dq_2bit_16(b_ptr, dq, size_n); b_ptr += size_n    ; for (int j = 0; j < 16; j++) b_.set(b_q_perm[offset_k + k++], n, __hmul(dq[j], qs_h)); } break;
//             case 3:                             { half dq[32]; dq_3bit_32(b_ptr, dq, size_n); b_ptr += size_n * 3; for (int j = 0; j < 32; j++) b_.set(b_q_perm[offset_k + k++], n, __hmul(dq[j], qs_h)); } break;
//             case 4: for (int p = 0; p < 4; p++) { half dq[8];  dq_4bit_8 (b_ptr, dq, size_n); b_ptr += size_n    ; for (int j = 0; j <  8; j++) b_.set(b_q_perm[offset_k + k++], n, __hmul(dq[j], qs_h)); } break;
//             case 5:                             { half dq[32]; dq_5bit_32(b_ptr, dq, size_n); b_ptr += size_n * 5; for (int j = 0; j < 32; j++) b_.set(b_q_perm[offset_k + k++], n, __hmul(dq[j], qs_h)); } break;
//             case 6: for (int p = 0; p < 2; p++) { half dq[16]; dq_6bit_16(b_ptr, dq, size_n); b_ptr += size_n * 3; for (int j = 0; j < 16; j++) b_.set(b_q_perm[offset_k + k++], n, __hmul(dq[j], qs_h)); } break;
//             case 8: for (int p = 0; p < 8; p++) { half dq[4];  dq_8bit_4 (b_ptr, dq, size_n); b_ptr += size_n    ; for (int j = 0; j <  4; j++) b_.set(b_q_perm[offset_k + k++], n, __hmul(dq[j], qs_h)); } break;
//             case 2: for (int p = 0; p < 2; p++) { half dq[16]; dq_2bit_16(b_ptr, dq, size_n); b_ptr += size_n    ; for (int j = 0; j < 16; j++) b_.set(offset_k + k++, n, __hmul(dq[j], qs_h)); } break;
//             case 3:                             { half dq[32]; dq_3bit_32(b_ptr, dq, size_n); b_ptr += size_n * 3; for (int j = 0; j < 32; j++) b_.set(offset_k + k++, n, __hmul(dq[j], qs_h)); } break;
//             case 4: for (int p = 0; p < 4; p++) { half dq[8];  dq_4bit_8 (b_ptr, dq, size_n); b_ptr += size_n    ; for (int j = 0; j <  8; j++) b_.set(offset_k + k++, n, __hmul(dq[j], qs_h)); } break;
//             case 5:                             { half dq[32]; dq_5bit_32(b_ptr, dq, size_n); b_ptr += size_n * 5; for (int j = 0; j < 32; j++) b_.set(offset_k + k++, n, __hmul(dq[j], qs_h)); } break;
//             case 6: for (int p = 0; p < 2; p++) { half dq[16]; dq_6bit_16(b_ptr, dq, size_n); b_ptr += size_n * 3; for (int j = 0; j < 16; j++) b_.set(offset_k + k++, n, __hmul(dq[j], qs_h)); } break;
//             case 8: for (int p = 0; p < 8; p++) { half dq[4];  dq_8bit_4 (b_ptr, dq, size_n); b_ptr += size_n    ; for (int j = 0; j <  4; j++) b_.set(offset_k + k++, n, __hmul(dq[j], qs_h)); } break;
        }
    }
}

void QMatrix::reconstruct(half* out)
{
    dim3 blockDim, gridDim;
    blockDim.x = BLOCK_KN_SIZE;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);
    gridDim.y = DIVIDE(height, BLOCK_KN_SIZE);

    reconstruct_kernel<<<gridDim, blockDim>>>
    (
        cuda_q_weight,
        cuda_q_perm,
        cuda_q_scale,
        cuda_q_scale_max,
        cuda_q_groups,
        height,
        width,
        groupsize,
        groups,
        out
    );
}
