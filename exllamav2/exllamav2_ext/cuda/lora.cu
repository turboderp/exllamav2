#include "lora.cuh"
#include "util.cuh"

#include "compat_gemm.cuh"

// alpha * ( a[m,k] @ b[k,n] ) + beta * c[m,n] -> c[m,n]

void gemm_
(
    cublasHandle_t cublas_handle,
    int size_m,
    int size_n,
    int size_k,
    const half* a,
    const half* b,
    half* c,
    float alpha,
    float beta
)
{
    //DBGI3(size_m, size_n, size_k);

    half alpha_ = __float2half(alpha);
    half beta_ = __float2half(beta);
    cublasHgemm(cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                size_n, size_m, size_k,
                &alpha_, b, size_n,
                         a, size_k,
                &beta_,  c, size_n);
}

void apply_loras
(
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

        gemm_(cublas_handle, rows, rank, base->height, input, lora_a, temp, 1.0f, 0.0f);
        gemm_(cublas_handle, rows, base->width, rank, temp, lora_b, output, 1.0f, 1.0f);
    }
}
