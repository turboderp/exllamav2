#include "lora.cuh"
#include "util.cuh"
#include "h_gemm.cuh"

void apply_loras_cuda
(
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    const std::unordered_map<uintptr_t, std::tuple<half*, half*, int>>& adapters,
    const std::vector<uintptr_t>& ids,
    QMatrix* base,
    const half* input,
    half* output,
    half* temp,
    int rows
)
{
    for (uintptr_t lora_id : ids)
    {
        auto it = adapters.find(lora_id);
        if (it == adapters.end()) continue;

        const std::tuple<half*, half*, int>& lora = it->second;
        half* lora_a = std::get<0>(lora);
        half* lora_b = std::get<1>(lora);
        int rank = std::get<2>(lora);

//         DBGI3(rows, rank, base->height);
//         DBGI3(rows, base->width, rank);

        h_gemm_cuda(stream, cublas_handle, rows, rank, base->height, input, lora_a, temp, 1.0f, 0.0f);
        h_gemm_cuda(stream, cublas_handle, rows, base->width, rank, temp, lora_b, output, 1.0f, 1.0f);
    }
}
