
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
