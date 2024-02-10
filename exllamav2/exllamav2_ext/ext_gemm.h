
void gemm_half_half_half
(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    const float alpha,
    const float beta,
    bool force_cublas
);
