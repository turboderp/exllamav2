#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ext_quant.h"
#include "ext_sampling.h"
#include "ext_safetensors.h"
#include "ext_qmatrix.h"
#include "ext_qattn.h"
#include "ext_qmlp.h"
#include "ext_cache.h"
#include "ext_gemm.h"
#include "ext_norm.h"
#include "ext_rope.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // quant

    m.def("pack_rows_4", &pack_rows_4, "pack_rows_4");
    m.def("pack_columns", &pack_columns, "pack_columns");
    m.def("quantize", &quantize, "quantize");
    m.def("quantize_err", &quantize_err, "quantize_err");
    m.def("quantize_range", &quantize_range, "quantize_range");
    m.def("quantize_range_inplace", &quantize_range_inplace, "quantize_range_inplace");

    // sampling

    m.def("apply_rep_penalty", &apply_rep_penalty, "apply_rep_penalty");
    m.def("sample_basic", &sample_basic, "sample_basic");
    m.def("logit_filter_exclusive", &logit_filter_exclusive, "logit_filter_exclusive");
    m.def("fast_fill_cpu_ones_bool", &fast_fill_cpu_ones_bool, "fast_fill_cpu_ones_bool");
    m.def("fast_fadd_cpu", &fast_fadd_cpu, "fast_fadd_cpu");
    m.def("fast_copy_cpu", &fast_copy_cpu, "fast_copy_cpu");
    m.def("dump_profile_results", &dump_profile_results, "dump_profile_results");

    // safetensors

    m.def("safetensors_open", &safetensors_open, "safetensors_open");
    m.def("safetensors_close", &safetensors_close, "safetensors_close");
    m.def("safetensors_load", &safetensors_load, "safetensors_load");
    m.def("safetensors_pinned_buffer", &safetensors_pinned_buffer, "safetensors_pinned_buffer");
    m.def("safetensors_free_pinned_buffer", &safetensors_free_pinned_buffer, "safetensors_free_pinned_buffer");

    // qmatrix

    m.def("make_q_matrix", &make_q_matrix, "make_q_matrix");
    m.def("free_q_matrix", &free_q_matrix, "free_q_matrix");
    m.def("reconstruct", &reconstruct, "reconstruct");
    m.def("gemm_half_q_half", &gemm_half_q_half, "gemm_half_q_half");
    m.def("matrix_fp16_to_q4", &matrix_fp16_to_q4, "matrix_fp16_to_q4");
    m.def("matrix_q4_to_fp16", &matrix_q4_to_fp16, "matrix_q4_to_fp16");

    // qattn

    m.def("make_q_attn", &make_q_attn, "make_q_attn");
    m.def("free_q_attn", &free_q_attn, "free_q_attn");
    m.def("q_attn_forward_1", &q_attn_forward_1, "q_attn_forward_1");
    m.def("q_attn_forward_2", &q_attn_forward_2, "q_attn_forward_2");
    m.def("q_attn_set_loras", &q_attn_set_loras, "q_attn_set_loras");

    // qmlp

    m.def("make_q_mlp", &make_q_mlp, "make_q_mlp");
    m.def("free_q_mlp", &free_q_mlp, "free_q_mlp");
    m.def("make_q_moe_mlp", &make_q_moe_mlp, "make_q_moe_mlp");
    m.def("free_q_moe_mlp", &free_q_moe_mlp, "free_q_moe_mlp");
    m.def("q_mlp_forward_", &q_mlp_forward_, "q_mlp_forward_");
    m.def("q_mlp_set_loras", &q_mlp_set_loras, "q_mlp_set_loras");
    m.def("q_moe_mlp_forward_", &q_moe_mlp_forward_, "q_moe_mlp_forward_");
//    m.def("q_moe_mlp_set_loras", &q_moe_mlp_set_loras, "q_moe_mlp_set_loras");

    // cache

    m.def("fp16_to_fp8", &fp16_to_fp8, "fp16_to_fp8");
    m.def("fp8_to_fp16", &fp8_to_fp16, "fp8_to_fp16");
    m.def("fp16_to_q4_kv", &fp16_to_q4_kv, "fp16_to_q4_kv");
    m.def("q4_to_fp16_kv", &q4_to_fp16_kv, "q4_to_fp16_kv");
//    m.def("array_fp16_to_fp8_ref", &array_fp16_to_fp8_ref, "array_fp16_to_fp8_ref");
//    m.def("array_fp8_to_fp16_ref", &array_fp8_to_fp16_ref, "array_fp8_to_fp16_ref");

    // gemm

    m.def("gemm_half_half_half", &gemm_half_half_half, "gemm_half_half_half");

    // norm

    m.def("rms_norm", &rms_norm, "rms_norm");
    m.def("rms_norm_", &rms_norm_, "rms_norm_");
    m.def("layer_norm", &layer_norm, "layer_norm");
    m.def("layer_norm_", &layer_norm_, "layer_norm_");
    m.def("head_norm", &head_norm, "head_norm");
    m.def("head_norm_", &head_norm_, "head_norm_");

    // rope

    m.def("rope_", &rope_, "rope_");
}