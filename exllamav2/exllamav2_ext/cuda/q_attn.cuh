#ifndef _q_attn_cuh
#define _q_attn_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "q_matrix.cuh"
#include "graph.cuh"

#define ROPE_STYLE_NONE 0
#define ROPE_STYLE_GPTJ 1
#define ROPE_STYLE_NEOX 2

struct QAttn_params_const
{
    int batch_size;
    int q_len;

    bool operator==(const QAttn_params_const& other) const
    {
        return batch_size == other.batch_size &&
               q_len == other.q_len;
    }
};

struct QAttn_params_const_hash
{
    std::size_t operator()(const QAttn_params_const& key) const
    {
        return (std::hash<int>()(key.batch_size)) ^
               (std::hash<int>()(key.q_len) << 1);
    }
};

class QAttn
{
public:

    half* layernorm;
    half* layernorm_bias;
    half* post_layernorm;
    half* post_layernorm_bias;
    bool layernorm_is_rms;
    float norm_epsilon;

    half* q_norm;
    half* k_norm;

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

    std::unordered_map<uintptr_t, std::tuple<half*, half*, int>> q_proj_lora;
    std::unordered_map<uintptr_t, std::tuple<half*, half*, int>> k_proj_lora;
    std::unordered_map<uintptr_t, std::tuple<half*, half*, int>> v_proj_lora;
    std::unordered_map<uintptr_t, std::tuple<half*, half*, int>> o_proj_lora;

    bool has_residual;
    bool residual_fp32;
    int rope_style;

    bool use_graphs;
    std::unordered_map<QAttn_params_const, Graph*, QAttn_params_const_hash> graph_map;

    QAttn
    (
        half* _layernorm,
        half* _layernorm_bias,
        bool _layernorm_is_rms,
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
        int _max_seq_len,
        bool _has_residual,
        int _rope_style,
        half* _q_norm,
        half* _k_norm,
        half* _post_layernorm,
        half* _post_layernorm_bias,
        bool _residual_fp32,
        bool _use_graphs
    );

    ~QAttn();

    void forward_cuda_1
    (
        cudaStream_t stream,
        cublasHandle_t cublas_handle,
        void* x,
        int batch_size,
        int q_len,
        int past_len,
        int32_t* past_lens,
        half* temp_q,
        half* temp_k,
        half* temp_v,
        const half* sin,
        const half* cos,
        const std::vector<uintptr_t>& loras,
        half* lora_temp
    );

    void forward_cuda_1_run
    (
        cudaStream_t stream,
        cublasHandle_t cublas_handle,
        void* x,
        int batch_size,
        int q_len,
        int past_len,
        int32_t* past_lens,
        half* temp_q,
        half* temp_k,
        half* temp_v,
        const half* sin,
        const half* cos,
        const std::vector<uintptr_t>& loras,
        half* lora_temp,
        Graph* graph = NULL
    );

    void forward_cuda_2
    (
        cudaStream_t stream,
        cublasHandle_t cublas_handle,
        const half* attn_output,
        void* hidden_state,
        int q_len,
        int batch_size,
        const std::vector<uintptr_t>& loras,
        half* lora_temp
    );

private:

};

#endif