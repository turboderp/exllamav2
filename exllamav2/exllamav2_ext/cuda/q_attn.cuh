#ifndef _q_attn_cuh
#define _q_attn_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "q_matrix.cuh"

class QAttn
{
public:

    half* layernorm;
    float norm_epsilon;

    QMatrix* q_proj;
    QMatrix* k_proj;
    QMatrix* v_proj;
    QMatrix* o_proj;

    half* temp_state;
//     half* temp_q;
//     half* temp_k;
//     half* temp_v;
    half* temp_dq;

    int device;
    int max_rows;
    int hidden_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;

    QAttn
    (
        half* _layernorm,
        float _norm_epsilon,
        QMatrix* _q_proj,
        QMatrix* _k_proj,
        QMatrix* _v_proj,
        QMatrix* _o_proj,
        half* _temp_state,
//         half* _temp_q,
//         half* _temp_k,
//         half* _temp_v,
        half* _temp_dq,
        int _max_rows,
        int _hidden_size,
        int _num_heads,
        int _num_kv_heads,
        int _head_dim,
        int _max_seq_len
    );

    ~QAttn();

    void forward_cuda_1
    (
        cublasHandle_t cublas_handle,
        half* x,
        int batch_size,
        int q_len,
        int past_len,
        const uint32_t* past_lens,
        half* temp_q,
        half* temp_k,
        half* temp_v,
        const half* sin,
        const half* cos
    );

    void forward_cuda_2
    (
        cublasHandle_t cublas_handle,
        const half* attn_output,
        half* hidden_state,
        int q_len,
        int batch_size
    );

private:

};

#endif