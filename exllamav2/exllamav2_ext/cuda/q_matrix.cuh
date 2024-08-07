#ifndef _q_matrix_cuh
#define _q_matrix_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#define MAX_SUPERGROUPS 16

class QMatrix
{
public:

    int device;
    bool is_gptq;

    int height;
    int width;
    int groups;
    int gptq_groupsize;

    int rows_8;
    int rows_6;
    int rows_5;
    int rows_4;
    int rows_3;
    int rows_2;
    int kernel_p;

    uint32_t* cuda_q_weight = NULL;
    uint16_t* cuda_q_perm = NULL;
    uint16_t* cuda_q_invperm = NULL;
    uint32_t* cuda_q_scale = NULL;
    half* cuda_q_scale_max = NULL;
    uint16_t* cuda_q_groups = NULL;
    uint16_t* cuda_q_group_map = NULL;
    uint32_t* cuda_gptq_qzeros = NULL;
    half* cuda_gptq_scales = NULL;
    half* cuda_bias = NULL;

    half* temp_dq;
    int max_dq_rows;

    bool failed;

    QMatrix
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
        uint16_t* _q_group_map,

        uint32_t* _gptq_qzeros,
        half* _gptq_scales,
        uint32_t* _gptq_g_idx,

        half* bias,

        half* _temp_dq,
        const int _max_dq_rows,

        bool no_map = false
    );

    ~QMatrix();

    void reconstruct(cudaStream_t stream, half* out, int row_a = 0, int row_b = 0);
    bool make_sequential(const uint32_t* cpu_g_idx);

private:

};

void matrix_q4_to_fp16_cuda
(
    cudaStream_t stream,
    const uint8_t* in_ptr,
    const half* scales_ptr,
    half* out_ptr,
    int numel
);

void matrix_fp16_to_q4_cuda
(
    cudaStream_t stream,
    const half* in_ptr,
    uint8_t* out_ptr,
    half* scales_ptr,
    int numel
);

void matrix_fp8_to_fp16_cuda
(
    cudaStream_t stream,
    const uint8_t* in_ptr,
    half* out_ptr,
    int numel
);

void matrix_fp16_to_fp8_cuda
(
    cudaStream_t stream,
    const half* in_ptr,
    uint8_t* out_ptr,
    int numel
);

#endif
