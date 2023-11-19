#ifndef _q_mlp_cuh
#define _q_mlp_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "q_matrix.cuh"

class QMLP
{
public:

    half* layernorm;
    float norm_epsilon;

    QMatrix* gate;
    QMatrix* up;
    QMatrix* down;

    half* temp_state;
    half* temp_a;
    half* temp_b;
    half* temp_dq;

    int device;
    int max_rows;

    std::unordered_map<uintptr_t, std::tuple<half*, half*, int>> gate_proj_lora;
    std::unordered_map<uintptr_t, std::tuple<half*, half*, int>> up_proj_lora;
    std::unordered_map<uintptr_t, std::tuple<half*, half*, int>> down_proj_lora;

    QMLP
    (
        half* _layernorm,
        float _norm_epsilon,
        QMatrix* _gate,
        QMatrix* _up,
        QMatrix* _down,
        half* _temp_state,
        half* _temp_a,
        half* _temp_b,
        half* _temp_dq,
        int _max_rows
    );

    ~QMLP();

    void forward_
    (
        cublasHandle_t cublas_handle,
        half* x,
        int rows,
        int columns,
        const std::vector<uintptr_t>& loras,
        half* lora_temp
    );

private:

};

#endif