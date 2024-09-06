#pragma once

#include <vector>
#include <tuple>

#include <torch/all.h>

#include "cpp/quantize_func.h"
#include "cpp/safetensors.h"
#include "cpp/generator.h"
#include "cpp/threadpool.h"
#include "cuda/tp.cuh"

// cache ops
void fp16_to_fp8(
    torch::Tensor in_tensor,
    torch::Tensor out_tensor,
    int64_t batch_size,
    int64_t offset,
    int64_t width);
void fp8_to_fp16(
    torch::Tensor in_tensor,
    torch::Tensor out_tensor,
    int64_t batch_size,
    int64_t offset,
    int64_t width);

void fp16_to_q_kv
(
    torch::Tensor k_in,
    torch::Tensor k_out,
    torch::Tensor k_scales,
    torch::Tensor v_in,
    torch::Tensor v_out,
    torch::Tensor v_scales,
    int64_t batch_size,
    int64_t offset,
    int64_t width,
    int64_t page_size,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    torch::Tensor cal_k,
    torch::Tensor cal_v,
    int64_t wbits
);

void q_to_fp16_kv
(
    torch::Tensor k_in,
    torch::Tensor k_out,
    torch::Tensor k_scales,
    torch::Tensor v_in,
    torch::Tensor v_out,
    torch::Tensor v_scales,
    int64_t batch_size,
    int64_t offset,
    int64_t width,
    int64_t page_size,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    torch::Tensor cal_k,
    torch::Tensor cal_v,
    int64_t wbits
);

int64_t count_match
(
    torch::Tensor a,
    torch::Tensor b,
    int64_t max_a
);

// element ops
void softcap_
(
    torch::Tensor x,
    double scale
);

// gemm ops
void gemm_half_half_half
(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    const double alpha,
    const double beta,
    bool force_cublas
);

// hadamard ops
void had_paley
(
    torch::Tensor h
);

void had_paley2
(
    torch::Tensor h
);

// layernorm ops
void rms_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    double epsilon
);

void rms_norm_tp
(
    std::vector<torch::Tensor> x,
    std::vector<torch::Tensor> w,
    std::vector<torch::Tensor> y,
    double epsilon,
    uintptr_t tp_context
);

void rms_norm_
(
    torch::Tensor x,
    torch::Tensor w,
    double epsilon
);

void layer_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor y,
    double epsilon
);

void layer_norm_
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    double epsilon
);

void head_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor y,
    double epsilon
);

void head_norm_
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    double epsilon
);

// qattn ops
uintptr_t make_q_attn
(
    torch::Tensor layernorm,
    torch::Tensor layernorm_bias,
    bool layernorm_is_rms,
    double norm_epsilon,
    uintptr_t q_q_proj,
    uintptr_t q_k_proj,
    uintptr_t q_v_proj,
    uintptr_t q_o_proj,
    torch::Tensor temp_state,
//    torch::Tensor temp_q,
//    torch::Tensor temp_k,
//    torch::Tensor temp_v,
    torch::Tensor temp_dq,
    int64_t max_rows,
    int64_t hidden_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t max_seq_len,
    bool has_residual,
    int64_t rope_style,
    torch::Tensor q_norm,
    torch::Tensor k_norm,
    torch::Tensor post_layernorm,
    torch::Tensor post_layernorm_bias,
    bool residual_fp32,
    bool use_graphs
);

void free_q_attn
(
    uintptr_t handle
);

void q_attn_forward_1
(
    uintptr_t q_attn,
    torch::Tensor x,
    int64_t batch_size,
    int64_t q_len,
    int64_t past_len,
    torch::Tensor past_lens,
    torch::Tensor q_temp,
    torch::Tensor k_temp,
    torch::Tensor v_temp,
    torch::Tensor sin,
    torch::Tensor cos,
    const std::vector<uintptr_t>& loras,
    torch::Tensor loras_temp
);

void q_attn_forward_2
(
    uintptr_t q_attn,
    torch::Tensor x,
    torch::Tensor attn_output,
    int64_t batch_size,
    int64_t q_len,
    const std::vector<uintptr_t>& loras,
    torch::Tensor loras_temp
);

int64_t q_attn_set_loras
(
    uintptr_t q_attn,
    std::unordered_map<uintptr_t, torch::Tensor>& q_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& q_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& k_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& k_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& v_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& v_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& o_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& o_proj_lora_b
);

// TODO: Find a way to call this function directly without going through pybind

typedef std::vector<at::Tensor> (*MHAFwdKVCacheFunc)
(
    at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    c10::optional<const at::Tensor> &,
    c10::optional<const at::Tensor> &,
    c10::optional<const at::Tensor> &,
    c10::optional<const at::Tensor> &,
    c10::optional<const at::Tensor> &,
    c10::optional<const at::Tensor> &,
    c10::optional<const at::Tensor> &,
    c10::optional<at::Tensor> &,
    c10::optional<at::Tensor> &,
    c10::optional<at::Tensor> &,
    const double,
    bool,
    int64_t,
    int64_t,
    const double,
    bool,
    int64_t,
);

//void set_flash_attn_func(MHAFwdKVCacheFunc f);
void set_flash_attn_func();

void tp_attn_forward_paged_
(
    uintptr_t tp_context,
    torch::Tensor hidden_states,
    const std::vector<torch::Tensor> &temp_bc0,
    const std::vector<torch::Tensor> &temp_bc1,
    const std::vector<torch::Tensor> &temp_bc2,
    const std::vector<torch::Tensor> &temp_q,
    const std::vector<torch::Tensor> &temp_k,
    const std::vector<torch::Tensor> &temp_v,
    const std::vector<torch::Tensor> &temp_o,
    const std::vector<torch::Tensor> &k_cache,
    const std::vector<torch::Tensor> &v_cache,
    const std::vector<torch::Tensor> &pre_layernorm,
    double norm_epsilon,
    const std::vector<uintptr_t> &q_proj,
    const std::vector<uintptr_t> &k_proj,
    const std::vector<uintptr_t> &v_proj,
    const std::vector<uintptr_t> &o_proj,
    int64_t head_dim,
    int64_t rope_style,
    int64_t batch_size,
    int64_t q_len,
    const std::vector<torch::Tensor> &sin,
    const std::vector<torch::Tensor> &cos,
    const std::vector<torch::Tensor> &past_lens,
    const std::vector<torch::Tensor> &block_index,
    double scaling
);

void tp_attn_forward_
(
    uintptr_t tp_context,
    torch::Tensor hidden_states,
    const std::vector<torch::Tensor> &temp_bc0,
    const std::vector<torch::Tensor> &temp_bc1,
    const std::vector<torch::Tensor> &temp_bc2,
    const std::vector<torch::Tensor> &temp_q,
    const std::vector<torch::Tensor> &temp_k,
    const std::vector<torch::Tensor> &temp_v,
    const std::vector<torch::Tensor> &temp_o,
    const std::vector<torch::Tensor> &k_cache,
    const std::vector<torch::Tensor> &v_cache,
    const std::vector<torch::Tensor> &pre_layernorm,
    double norm_epsilon,
    const std::vector<uintptr_t> &q_proj,
    const std::vector<uintptr_t> &k_proj,
    const std::vector<uintptr_t> &v_proj,
    const std::vector<uintptr_t> &o_proj,
    int64_t head_dim,
    int64_t rope_style,
    int64_t batch_size,
    int64_t q_len,
    const std::vector<torch::Tensor> &sin,
    const std::vector<torch::Tensor> &cos,
    const std::vector<torch::Tensor> &past_len_tp,
    double scaling
);

// qmatrix ops
uintptr_t make_q_matrix
(
    torch::Tensor q_weight,
    torch::Tensor q_perm,
    torch::Tensor q_invperm,
    torch::Tensor q_scale,
    torch::Tensor q_scale_max,
    torch::Tensor q_groups,
    torch::Tensor q_group_map,
    torch::Tensor gptq_qzeros,
    torch::Tensor gptq_scales,
    torch::Tensor gptq_g_idx,
    torch::Tensor bias,
    torch::Tensor temp_dq,
    int64_t max_dq_rows
);

uintptr_t make_q_matrix_split
(
    torch::Tensor q_weight,
    torch::Tensor q_perm,
    torch::Tensor q_invperm,
    torch::Tensor q_scale,
    torch::Tensor q_scale_max,
    torch::Tensor q_groups,
    torch::Tensor q_group_map,
    torch::Tensor gptq_qzeros,
    torch::Tensor gptq_scales,
    torch::Tensor gptq_g_idx,
    torch::Tensor bias,
    torch::Tensor temp_dq,
    int64_t max_dq_rows
);

void free_q_matrix
(
    uintptr_t tp_context
);

void reconstruct
(
    uintptr_t q_handle,
    torch::Tensor output
);

void gemm_half_q_half
(
    torch::Tensor a,
    uintptr_t b,
    torch::Tensor c,
    bool force_cuda
);

void gemm_half_q_half_tp
(
    const std::vector<torch::Tensor> &a,
    const std::vector<uintptr_t> &b,
    const std::vector<torch::Tensor> &c,
    bool force_cuda,
    uintptr_t tp_context,
    int64_t t_device = -1
);

void matrix_q4_to_fp16
(
    torch::Tensor in,
    torch::Tensor scales,
    torch::Tensor out
);

void matrix_fp16_to_q4
(
    torch::Tensor in,
    torch::Tensor out,
    torch::Tensor scales
);

// qmlp ops
uintptr_t make_q_mlp
(
    torch::Tensor layernorm,
    torch::Tensor layernorm_bias,
    bool layernorm_is_rms,
    double norm_epsilon,
    uintptr_t q_gate,
    uintptr_t q_up,
    uintptr_t q_down,
    torch::Tensor temp_state,
    torch::Tensor temp_a,
    torch::Tensor temp_b,
    torch::Tensor temp_dq,
    int64_t max_rows,
    bool act_gelu,
    bool has_residual,
    torch::Tensor post_layernorm,
    torch::Tensor post_layernorm_bias,
    bool residual_fp32,
    bool use_graphs
);

void free_q_mlp
(
   uintptr_t handle
);

void q_mlp_forward_
(
    uintptr_t q_mlp,
    torch::Tensor x,
    const std::vector<uintptr_t>& loras,
    torch::Tensor loras_temp
);

int64_t q_mlp_set_loras
(
    uintptr_t q_mlp,
    std::unordered_map<uintptr_t, torch::Tensor>& gate_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& gate_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& up_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& up_proj_lora_b,
    std::unordered_map<uintptr_t, torch::Tensor>& down_proj_lora_a,
    std::unordered_map<uintptr_t, torch::Tensor>& down_proj_lora_b
);

uintptr_t make_q_moe_mlp
(
    torch::Tensor layernorm,
    torch::Tensor layernorm_bias,
    bool layernorm_is_rms,
    double norm_epsilon,
    torch::Tensor gate,
    int64_t num_experts,
    int64_t num_experts_per_token,
    const std::vector<uintptr_t>& w1,
    const std::vector<uintptr_t>& w2,
    const std::vector<uintptr_t>& w3,
    torch::Tensor temp_state,
    torch::Tensor temp_gathered_state,
    torch::Tensor temp_a,
    torch::Tensor temp_b,
    torch::Tensor temp_logits,
    torch::Tensor temp_dq,
    int64_t max_rows,
    bool act_gelu
);

void free_q_moe_mlp
(
   uintptr_t handle
);

void q_moe_mlp_forward_
(
    uintptr_t q_moe_mlp,
    torch::Tensor x
//    const std::vector<uintptr_t>& loras,
//    torch::Tensor loras_temp
);

//int64_t q_moe_mlp_set_loras
//(
//    uintptr_t q_moe_mlp,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w1_lora_a,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w1_lora_b,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w2_lora_a,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w2_lora_b,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w3_lora_a,
//    std::vector<std::unordered_map<uintptr_t, torch::Tensor>>& w3_lora_b
//);

void tp_mlp_forward_
(
    uintptr_t tp_context,
    torch::Tensor hidden_states,
    const std::vector<torch::Tensor> &temp_bc0,
    const std::vector<torch::Tensor> &temp_bc1,
    const std::vector<torch::Tensor> &temp_bc2,
    const std::vector<torch::Tensor> &temp_gate,
    const std::vector<torch::Tensor> &temp_up,
    const std::vector<torch::Tensor> &temp_down,
    const std::vector<torch::Tensor> &pre_layernorm,
    double norm_epsilon,
    const std::vector<uintptr_t> &gate,
    const std::vector<uintptr_t> &up,
    const std::vector<uintptr_t> &down,
    bool act_gelu
);

// quant ops
void pack_columns
(
    torch::Tensor input,
    torch::Tensor output,
    int64_t bits
);

void pack_rows_4
(
    torch::Tensor input,
    torch::Tensor output
);

void quantize_err
(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    double qzero,
    double maxq,
    double err_norm,
    double min_p,
    double max_p,
    int64_t p_grid
);

void quantize
(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    torch::Tensor out_q,
    double qzero,
    double maxq
);

std::tuple<std::vector<std::tuple<uint64_t, double>>, std::vector<int64_t>, double, uint64_t, double> sim_anneal
(
    const std::vector<std::vector<std::tuple<uint64_t, double>>>& slots,
    uint64_t max_cost,
    double initial_temp,
    double cooling_factor,
    double min_temp,
    int64_t iterations,
    double norm
);

// rope ops
void rope_
(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos,
    int64_t past_len,
    int64_t num_heads,
    int64_t head_dim,
    torch::Tensor offsets,
    bool neox_style
);

// sampling ops
void apply_rep_penalty
(
    torch::Tensor sequence,
    double penalty_max,
    int64_t sustain,
    int64_t decay,
    double alpha_frequency,
    double alpha_presence,
    torch::Tensor logits
);

std::vector<double> sample_basic
(
    torch::Tensor logits,           // shape [bsz, vocab_size]
    double temperature,
    int64_t top_k,
    float top_p,
    double top_a,
    double min_p,
    double tfs,
    double typical,
    double random,
    torch::Tensor output_tokens,    // shape [bsz, 1]
    torch::Tensor output_probs,     // shape [bsz, 1]
    torch::Tensor output_kprobs,    // None or [bsz, 1, num_probs]
    torch::Tensor output_ktokens,   // None or [bsz, 1, num_probs]
    torch::Tensor logit_filter,     // shape [bsz, vocab_size]
    bool mirostat,
    std::vector<double>& mirostat_mu,
    double mirostat_tau,
    double mirostat_eta,
    double post_temperature,
    double min_temp,
    double max_temp,
    double temp_exponent,
    double smoothing_factor,
    double skew
);

void logit_filter_exclusive
(
    torch::Tensor filter,                                       // shape [bsz, vocab_size]
    const py::list& exclusive_lists
);

void fast_fill_cpu_ones_bool(torch::Tensor tensor);

void fast_fadd_cpu(torch::Tensor a, torch::Tensor b);

void fast_copy_cpu(torch::Tensor a, torch::Tensor b);

void dump_profile_results();

// tensor parallel ops
#ifndef _ext_tp_h
#define _ext_tp_h

#define BROADCAST_KV 0
#define BROADCAST_ID 1
#define BROADCAST_VC 2
#define BROADCAST_RS 3
#define BROADCAST_Q 4

class ExtTPContext
{
public:
    std::vector<std::tuple<int64_t, int64_t, int64_t>> kv_split;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> id_split;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> vc_split;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> rs_split;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> q_split;
    std::vector<void*> pinned_temp;
    size_t pinned_size;
    std::vector<cudaStream_t> streams;

    std::vector<int64_t> all_devices;

    ThreadPool* thread_pool;
    ExtTPData* tp_data;

    void* mapped_globals;

    std::vector<cudaEvent_t> sync_events;
//    std::vector<ncclComm_t> comms;
//    std::vector<int64_t> comms_index;

    ExtTPContext
    (
        std::vector<std::tuple<int64_t, int64_t, int64_t>> _kv_split,
        std::vector<std::tuple<int64_t, int64_t, int64_t>> _id_split,
        std::vector<std::tuple<int64_t, int64_t, int64_t>> _vc_split,
        std::vector<std::tuple<int64_t, int64_t, int64_t>> _rs_split,
        std::vector<std::tuple<int64_t, int64_t, int64_t>> _q_split,
        std::vector<torch::Tensor> _pinned_temp,
        std::vector<cudaStream_t> _streams
    );
    ~ExtTPContext();
};

uintptr_t make_tp_context
(
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> kv_split,
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> id_split,
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> vc_split,
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> rs_split,
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> q_split,
    std::vector<torch::Tensor> pinned_temp,
    std::vector<uintptr_t> streams
);

void free_tp_context(uintptr_t ctx);

void tp_broadcast
(
    uintptr_t tp_context,
    int64_t buffer,
    torch::Tensor source,
    int64_t broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int64_t dim,
    int64_t t_device = -1
);

void tp_gather
(
    uintptr_t tp_context,
    int64_t buffer,
    const std::vector<torch::Tensor> &inputs,
    int64_t broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int64_t broadcast_type_target,
    int64_t dim,
    int64_t t_device = -1
);

void tp_gather_barrier
(
    uintptr_t tp_context,
    int64_t buffer,
    const std::vector<torch::Tensor> &inputs,
    int64_t broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int64_t broadcast_type_target,
    int64_t dim,
    int64_t t_device = -1,
    Barrier* barrier = nullptr
);

void tp_cross_device_barrier
(
    uintptr_t tp_context,
    int64_t broadcast_type,
    int64_t t_device = -1,
    int64_t stage = -1,
    int64_t next_stage = -1
);

//void tp_all_reduce
//(
//    uintptr_t tp_context,
//    const std::vector<torch::Tensor> &tensors
//);

void tp_all_reduce
(
    uintptr_t tp_context,
    int64_t buffer,
    const std::vector<torch::Tensor> &tensors,
    const std::vector<torch::Tensor> &residuals
);

#endif