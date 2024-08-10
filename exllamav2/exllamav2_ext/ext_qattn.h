
uintptr_t make_q_attn
(
    torch::Tensor layernorm,
    torch::Tensor layernorm_bias,
    bool layernorm_is_rms,
    float norm_epsilon,
    uintptr_t q_q_proj,
    uintptr_t q_k_proj,
    uintptr_t q_v_proj,
    uintptr_t q_o_proj,
    torch::Tensor temp_state,
//    torch::Tensor temp_q,
//    torch::Tensor temp_k,
//    torch::Tensor temp_v,
    torch::Tensor temp_dq,
    int max_rows,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    bool has_residual,
    int rope_style,
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
    int batch_size,
    int q_len,
    int past_len,
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
    int batch_size,
    int q_len,
    const std::vector<uintptr_t>& loras,
    torch::Tensor loras_temp
);

int q_attn_set_loras
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
    const float,
    bool,
    int,
    int,
    const float,
    bool,
    int
);

//void set_flash_attn_func(MHAFwdKVCacheFunc f);
void set_flash_attn_func();

void tp_attn_forward_
(
    uintptr_t tp_context,
    torch::Tensor hidden_states,
    const py::list &temp_bc0,
    const py::list &temp_bc1,
    const py::list &temp_q,
    const py::list &temp_k,
    const py::list &temp_v,
    const py::list &temp_o,
    const py::list &k_cache,
    const py::list &v_cache,
    const py::list &pre_layernorm,
    float norm_epsilon,
    const py::list &q_proj,
    const py::list &k_proj,
    const py::list &v_proj,
    const py::list &o_proj,
    int head_dim,
    int rope_style,
    int batch_size,
    int q_len,
    const py::list &sin,
    const py::list &cos,
    const py::list &past_lens,
    const py::list &block_index,
    float scaling
);