
void fp16_to_fp8(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width);
void fp8_to_fp16(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width);

void fp16_to_q_kv
(
    torch::Tensor k_in,
    torch::Tensor k_out,
    torch::Tensor k_scales,
    torch::Tensor v_in,
    torch::Tensor v_out,
    torch::Tensor v_scales,
    int batch_size,
    int offset,
    int width,
    int page_size,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    torch::Tensor cal_k,
    torch::Tensor cal_v,
    int wbits
);

void q_to_fp16_kv
(
    torch::Tensor k_in,
    torch::Tensor k_out,
    torch::Tensor k_scales,
    torch::Tensor v_in,
    torch::Tensor v_out,
    torch::Tensor v_scales,
    int batch_size,
    int offset,
    int width,
    int page_size,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    torch::Tensor cal_k,
    torch::Tensor cal_v,
    int wbits
);

int count_match
(
    torch::Tensor a,
    torch::Tensor b,
    int max_a
);

//void array_fp16_to_fp8_ref(torch::Tensor in_tensor, torch::Tensor out_tensor, int size);
//void array_fp8_to_fp16_ref(torch::Tensor in_tensor, torch::Tensor out_tensor, int size);



