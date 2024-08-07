
void rms_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    float epsilon
);

void rms_norm_tp
(
    std::vector<torch::Tensor> x,
    std::vector<torch::Tensor> w,
    std::vector<torch::Tensor> y,
    float epsilon,
    uintptr_t tp_context
);

void rms_norm_
(
    torch::Tensor x,
    torch::Tensor w,
    float epsilon
);

void layer_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor y,
    float epsilon
);

void layer_norm_
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    float epsilon
);

void head_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor y,
    float epsilon
);

void head_norm_
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    float epsilon
);



