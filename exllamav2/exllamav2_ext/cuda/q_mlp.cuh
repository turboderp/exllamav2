#ifndef _q_mlp_cuh
#define _q_mlp_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "q_matrix.cuh"
#include "graph.cuh"

struct QMLP_params_const
{
    int rows;
    int columns;

    bool operator==(const QMLP_params_const& other) const
    {
        return rows == other.rows &&
               columns == other.columns;
    }
};

struct QMLP_params_const_hash
{
    std::size_t operator()(const QMLP_params_const& key) const
    {
        return (std::hash<int>()(key.rows)) ^
               (std::hash<int>()(key.columns) << 1);
    }
};

class QMLP
{
public:

    half* layernorm;
    half* layernorm_bias;
    half* post_layernorm;
    half* post_layernorm_bias;
    bool layernorm_is_rms;
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

    bool act_gelu;
    bool has_residual;
    bool residual_fp32;

    bool use_graphs;
    std::unordered_map<QMLP_params_const, Graph*, QMLP_params_const_hash> graph_map;

    QMLP
    (
        half* _layernorm,
        half* _layernorm_bias,
        bool _layernorm_is_rms,
        float _norm_epsilon,
        QMatrix* _gate,
        QMatrix* _up,
        QMatrix* _down,
        half* _temp_state,
        half* _temp_a,
        half* _temp_b,
        half* _temp_dq,
        int _max_rows,
        bool _act_gelu,
        bool _has_residual,
        half* _post_layernorm,
        half* _post_layernorm_bias,
        bool _residual_fp32,
        bool _use_graphs
    );

    ~QMLP();

    void forward_
    (
        cudaStream_t stream,
        cublasHandle_t cublas_handle,
        void* x,
        int rows,
        int columns,
        const std::vector<uintptr_t>& loras,
        half* lora_temp
    );

    void forward_run_
    (
        cudaStream_t stream,
        cublasHandle_t cublas_handle,
        void* x,
        int rows,
        int columns,
        const std::vector<uintptr_t>& loras,
        half* lora_temp,
        Graph* graph = NULL
    );

private:

};

// ---------------------------------------------------------------------------------

class QMoEMLP
{
public:

    half* layernorm;
    half* layernorm_bias;
    bool layernorm_is_rms;
    float norm_epsilon;

    half* gate;
    int num_experts;
    int num_experts_per_token;

    std::vector<QMatrix*> w1;
    std::vector<QMatrix*> w2;
    std::vector<QMatrix*> w3;

    half* temp_state;
    half* temp_gathered_state;
    half* temp_a;
    half* temp_b;
    half* temp_logits;
    half* temp_dq;

    int device;
    int max_rows;
    int hidden_dim;

    bool act_gelu;

//    std::vector<std::unordered_map<uintptr_t, std::tuple<half*, half*, int>>> w1_lora;
//    std::vector<std::unordered_map<uintptr_t, std::tuple<half*, half*, int>>> w2_lora;
//    std::vector<std::unordered_map<uintptr_t, std::tuple<half*, half*, int>>> w3_lora;

    QMoEMLP
    (
        half* _layernorm,
        half* _layernorm_bias,
        bool _layernorm_is_rms,
        float _norm_epsilon,
        half* _gate,
        int _num_experts,
        int _num_experts_per_token,
        std::vector<QMatrix*>& w1,
        std::vector<QMatrix*>& w2,
        std::vector<QMatrix*>& w3,
        half* _temp_state,
        half* _temp_gathered_state,
        half* _temp_a,
        half* _temp_b,
        half* _temp_logits,
        half* _temp_dq,
        int _max_rows,
        int _hidden_dim,
        bool _act_gelu
    );

    ~QMoEMLP();

    void forward_
    (
        cudaStream_t stream,
        cublasHandle_t cublas_handle,
        half* x,
        int rows,
        int columns
//        const std::vector<uintptr_t>& loras,
//        half* lora_temp
    );

private:

};

// ---------------------------------------------------------------------------------

void act_mul_cuda
(
    cudaStream_t stream,
    half* gate,
    half* up,
    int rows,
    int dim,
    bool act_gelu
);

#endif