
uintptr_t make_q_mlp
(
    torch::Tensor layernorm,
    torch::Tensor layernorm_bias,
    bool layernorm_is_rms,
    float norm_epsilon,
    uintptr_t q_gate,
    uintptr_t q_up,
    uintptr_t q_down,
    torch::Tensor temp_state,
    torch::Tensor temp_a,
    torch::Tensor temp_b,
    torch::Tensor temp_dq,
    int max_rows,
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

int q_mlp_set_loras
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
    float norm_epsilon,
    torch::Tensor gate,
    int num_experts,
    int num_experts_per_token,
    const std::vector<uintptr_t>& w1,
    const std::vector<uintptr_t>& w2,
    const std::vector<uintptr_t>& w3,
    torch::Tensor temp_state,
    torch::Tensor temp_gathered_state,
    torch::Tensor temp_a,
    torch::Tensor temp_b,
    torch::Tensor temp_logits,
    torch::Tensor temp_dq,
    int max_rows,
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

//int q_moe_mlp_set_loras
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
    float norm_epsilon,
    const std::vector<uintptr_t> &gate,
    const std::vector<uintptr_t> &up,
    const std::vector<uintptr_t> &down,
    bool act_gelu
);