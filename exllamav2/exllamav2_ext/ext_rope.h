
void rope_
(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos,
    int past_len,
    int num_heads,
    int head_dim,
    torch::Tensor offsets,
    bool neox_style
);

long gen_mrope_pos_ids
(
    torch::Tensor mrope_pos_ids,
    torch::Tensor ids,
    int merge_size,
    const std::vector<std::tuple<long, long>> &spans,
    const std::vector<std::tuple<long, long, long>> &grids
);