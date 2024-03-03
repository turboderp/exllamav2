
void fp16_to_fp8(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width);
void fp8_to_fp16(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width);

void fp16_to_q4(torch::Tensor in_tensor, torch::Tensor out_tensor, torch::Tensor scales, int batch_size, int offset, int width);
void q4_to_fp16(torch::Tensor in_tensor, torch::Tensor scales, torch::Tensor out_tensor, int batch_size, int offset, int width);

//void array_fp16_to_fp8_ref(torch::Tensor in_tensor, torch::Tensor out_tensor, int size);
//void array_fp8_to_fp16_ref(torch::Tensor in_tensor, torch::Tensor out_tensor, int size);



