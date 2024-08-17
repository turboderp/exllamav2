#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"
#include "ext_qmatrix.h"

#include "cuda/q_matrix.cuh"
#include "cuda/q_gemm.cuh"

#include "cpp/util.h"
#include "cpp/threadpool.h"
#include "ext_tp.h"

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
)
{
    TORCH_CHECK_DTYPE(q_weight, kInt);
    TORCH_CHECK_DTYPE_OPT(q_perm, kShort);
    TORCH_CHECK_DTYPE_OPT(q_invperm, kShort);
    TORCH_CHECK_DTYPE_OPT(q_scale, kInt);
    TORCH_CHECK_DTYPE_OPT(q_scale_max, kHalf);
    TORCH_CHECK_DTYPE_OPT(q_groups, kShort);
    TORCH_CHECK_DTYPE_OPT(q_group_map, kShort);
    TORCH_CHECK_DTYPE_OPT(gptq_qzeros, kInt);
    TORCH_CHECK_DTYPE_OPT(gptq_scales, kHalf);
    TORCH_CHECK_DTYPE_OPT(gptq_g_idx, kInt);
    TORCH_CHECK_DTYPE_OPT(bias, kHalf);

    TORCH_CHECK_SHAPES(q_perm, 0, q_invperm, 0, 1);

    int device = q_weight.device().index();
    int width = q_weight.size(1);
    int groups;
    int height;

    if (!q_scale.device().is_meta())
    {
        TORCH_CHECK_SHAPES(q_weight, 1, q_scale, 1, 8);
        TORCH_CHECK_SHAPES(q_scale_max, 0, q_scale, 0, 1);
        groups = q_scale.size(0);
        height = q_group_map.size(0) / 2;
    }
    else
    {
        TORCH_CHECK_SHAPES(q_weight, 1, gptq_qzeros, 1, 8);
        TORCH_CHECK_SHAPES(q_weight, 1, gptq_scales, 1, 1);
        groups = gptq_qzeros.size(0);
        height = q_weight.size(0) * 8;
    }

    if (!bias.device().is_meta())
    {
        TORCH_CHECK_SHAPES(q_weight, 1, bias, 0, 1);
    }

    if (!temp_dq.device().is_meta())
    {
        uint64_t dq_req = (uint64_t)width * std::min((uint64_t)max_dq_rows, (uint64_t)height);
        TORCH_CHECK(temp_dq.size(0) >= dq_req, "Insufficient size of temp_dq buffer")
    }

    QMatrix* m = new QMatrix
    (
        device,
        height,
        width,
        groups,
        (uint32_t*) q_weight.data_ptr(),
        q_perm.device().is_meta() ? NULL : (uint16_t*) q_perm.data_ptr(),
        q_invperm.device().is_meta() ? NULL : (uint16_t*) q_invperm.data_ptr(),
        q_scale.device().is_meta() ? NULL : (uint32_t*) q_scale.data_ptr(),
        q_scale_max.device().is_meta() ? NULL : (half*) q_scale_max.data_ptr(),
        q_groups.device().is_meta() ? NULL : (uint16_t*) q_groups.data_ptr(),
        q_group_map.device().is_meta() ? NULL : (uint16_t*) q_group_map.data_ptr(),
        gptq_qzeros.device().is_meta() ? NULL : (uint32_t*) gptq_qzeros.data_ptr(),
        gptq_scales.device().is_meta() ? NULL : (half*) gptq_scales.data_ptr(),
        gptq_g_idx.device().is_meta() ? NULL : (uint32_t*) gptq_g_idx.data_ptr(),
        bias.device().is_meta() ? NULL : (half*) bias.data_ptr(),
        (half*) temp_dq.data_ptr(),
        max_dq_rows
    );

    if (m->failed) throw std::runtime_error("CUDA out of memory");

    return reinterpret_cast<uintptr_t> (m);
}

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
)
{
    TORCH_CHECK(
        gptq_qzeros.device().is_meta() &&
        gptq_scales.device().is_meta() &&
        gptq_g_idx.device().is_meta(),
        "Tensor split not implemented for GPTQ matrices"
    );

    int device = q_weight.device().index();
    int width = q_weight.size(1);
    int groups;
    int height;

    if (!q_scale.device().is_meta())
    {
        TORCH_CHECK_SHAPES(q_weight, 1, q_scale, 1, 8);
        TORCH_CHECK_SHAPES(q_scale_max, 0, q_scale, 0, 1);
        groups = q_scale.size(0);
        height = q_group_map.size(0) / 2;
//        DBGI(groups);
//        DBGI(height);
    }
    else
    {
        TORCH_CHECK(false, "Tensor split not implemented for GPTQ matrices");
    }

    QMatrix* m = new QMatrix
    (
        device,
        height,
        width,
        groups,
        (uint32_t*) q_weight.data_ptr(),
        q_perm.device().is_meta() ? NULL : (uint16_t*) q_perm.data_ptr(),
        q_invperm.device().is_meta() ? NULL : (uint16_t*) q_invperm.data_ptr(),
        q_scale.device().is_meta() ? NULL : (uint32_t*) q_scale.data_ptr(),
        q_scale_max.device().is_meta() ? NULL : (half*) q_scale_max.data_ptr(),
        q_groups.device().is_meta() ? NULL : (uint16_t*) q_groups.data_ptr(),
        q_group_map.device().is_meta() ? NULL : (uint16_t*) q_group_map.data_ptr(),
        gptq_qzeros.device().is_meta() ? NULL : (uint32_t*) gptq_qzeros.data_ptr(),
        gptq_scales.device().is_meta() ? NULL : (half*) gptq_scales.data_ptr(),
        gptq_g_idx.device().is_meta() ? NULL : (uint32_t*) gptq_g_idx.data_ptr(),
        bias.device().is_meta() ? NULL : (half*) bias.data_ptr(),
        (half*) temp_dq.data_ptr(),
        max_dq_rows,
        true
    );

    if (m->failed) throw std::runtime_error("CUDA out of memory");

    return reinterpret_cast<uintptr_t> (m);
}

void free_q_matrix
(
    uintptr_t handle
)
{
    QMatrix* m = reinterpret_cast<QMatrix*> (handle);
    delete m;
}

void reconstruct
(
    uintptr_t q_handle,
    torch::Tensor output
)
{
    QMatrix* qm = reinterpret_cast<QMatrix*> (q_handle);
    TORCH_CHECK(qm->height == output.size(0) && qm->width == output.size(1), "Output tensor doesn't match shape of QMatrix")
    TORCH_CHECK_DTYPE(output, kHalf);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    qm->reconstruct(stream, (half*) output.data_ptr());
}


void gemm_half_q_half
(
    torch::Tensor a,
    uintptr_t b,
    torch::Tensor c,
    bool force_cuda
)
{
    QMatrix* qm = reinterpret_cast<QMatrix*> (b);

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(c, kHalf);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
//    DBGI2(qm->height, qm->width);
    TORCH_CHECK(qm->height == a.size(1), "a and b have incompatible shapes")
    TORCH_CHECK(qm->width == c.size(1), "b and c have incompatible shapes")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gemm_half_q_half_cuda
    (
        stream,
        at::cuda::getCurrentCUDABlasHandle(),
        (const half*) a.data_ptr(),
        qm,
        (half*) c.data_ptr(),
        c.size(0), // m
        c.size(1), // n
        a.size(1), // k
        true,
        NULL,
        force_cuda
    );
}

void gemm_half_q_half_tp
(
    const std::vector<torch::Tensor> &a,
    const std::vector<uintptr_t> &b,
    const std::vector<torch::Tensor> &c,
    bool force_cuda,
    uintptr_t tp_context,
    int t_device
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    for (size_t i = 0; i < a.size(); ++i)
    {
        QMatrix* qm = reinterpret_cast<QMatrix*> (b[i]);
        int dev = qm->device;
        if (t_device != -1 && t_device != dev) continue;
//        TORCH_CHECK_DTYPE(a[i], kHalf);
//        TORCH_CHECK_DTYPE(c[i], kHalf);
//        TORCH_CHECK_SHAPES(a[i], 0, c[i], 0, 1);
//        TORCH_CHECK(qm->height == a[i].size(1), "a and b have incompatible shapes")
//        TORCH_CHECK(qm->width == c[i].size(1), "b and c have incompatible shapes")

        cudaSetDevice(dev);
        cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
        gemm_half_q_half_cuda
        (
            ctx->streams[dev],
            cublas_handle,
            (const half*) a[i].data_ptr(),
            qm,
            (half*) c[i].data_ptr(),
            c[i].size(0), // m
            c[i].size(1), // n
            a[i].size(1), // k
            true,
            NULL,
            force_cuda
        );
    }
}

// Convert tensors

void matrix_q4_to_fp16
(
    torch::Tensor in,
    torch::Tensor scales,
    torch::Tensor out
)
{
    TORCH_CHECK(in.numel() * 2 == out.numel(), "matrix_q4_to_fp16: tensor size mismatch");
    TORCH_CHECK_DTYPE(in, kByte);
    TORCH_CHECK_DTYPE(out, kHalf);
    int numel = out.numel();
    const at::cuda::OptionalCUDAGuard device_guard(device_of(in));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    matrix_q4_to_fp16_cuda
    (
        stream,
        (const uint8_t*) in.data_ptr(),
        (const half*) scales.data_ptr(),
        (half*) out.data_ptr(),
        numel
    );
}

void matrix_fp16_to_q4
(
    torch::Tensor in,
    torch::Tensor out,
    torch::Tensor scales
)
{
    TORCH_CHECK(in.numel() == out.numel() * 2, "matrix_fp16_to_q4: tensor size mismatch");
    TORCH_CHECK_DTYPE(in, kHalf);
    TORCH_CHECK_DTYPE(out, kByte);
    int numel = in.numel();
    const at::cuda::OptionalCUDAGuard device_guard(device_of(in));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    matrix_fp16_to_q4_cuda
    (
        stream,
        (const half*) in.data_ptr(),
        (uint8_t*) out.data_ptr(),
        (half*) scales.data_ptr(),
        numel
    );
}