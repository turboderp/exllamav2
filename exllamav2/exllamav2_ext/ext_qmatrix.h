
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
    int max_dq_rows
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
    int max_dq_rows
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
    int t_device = -1
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


