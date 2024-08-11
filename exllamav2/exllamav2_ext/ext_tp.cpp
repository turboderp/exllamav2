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
#include "ext_tp.h"
//#include "cuda/util.cuh"

#define cuda_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

ExtTPContext::ExtTPContext
(
    std::vector<std::tuple<int, int, int>> _kv_split,
    std::vector<std::tuple<int, int, int>> _id_split,
    std::vector<std::tuple<int, int, int>> _vc_split,
    std::vector<std::tuple<int, int, int>> _rs_split,
    std::vector<std::tuple<int, int, int>> _q_split,
    torch::Tensor _pinned_temp,
    std::vector<torch::Tensor> _device_temp,
    std::vector<cudaStream_t> _streams
) :
    kv_split(_kv_split),
    id_split(_id_split),
    vc_split(_vc_split),
    rs_split(_rs_split),
    q_split(_q_split),
    streams(_streams)
{
    pinned_temp = (void*) _pinned_temp.data_ptr();
    pinned_size = _pinned_temp.numel() * _pinned_temp.element_size();

    for (int i = 0; i < _device_temp.size(); ++i)
        device_temp.push_back((void*) _device_temp[i].data_ptr());
}

ExtTPContext::~ExtTPContext()
{
    cudaEventDestroy(sync_event);
    for (int i = 0; i < streams.size(); ++i)
    {
        cuda_check(cudaEventDestroy(sync_events[i]));
        cuda_check(cudaEventDestroy(sync_events2[i]));
        cuda_check(cudaEventDestroy(sync_events3[i]));
    }
}

void ExtTPContext::create_events()
{
    if (sync_events.size()) return;
//    DBGI(sync_events.size());

//    DBGX(pinned_temp);

    cuda_check(cudaEventCreate(&sync_event));

    sync_events.resize(streams.size());
    sync_events2.resize(streams.size());
    sync_events3.resize(streams.size());
    for (int i = 0; i < streams.size(); ++i)
    {
        cudaSetDevice(i);
        cuda_check(cudaEventCreate(&sync_events[i]));
        cuda_check(cudaEventCreate(&sync_events2[i]));
        cuda_check(cudaEventCreate(&sync_events3[i]));
//        DBGIX(i, sync_events[i]);
//        DBGIX(i, sync_events2[i]);
//        DBGIX(i, sync_events3[i]);
    }
}

uintptr_t make_tp_context
(
    std::vector<std::tuple<int, int, int>> kv_split,
    std::vector<std::tuple<int, int, int>> id_split,
    std::vector<std::tuple<int, int, int>> vc_split,
    std::vector<std::tuple<int, int, int>> rs_split,
    std::vector<std::tuple<int, int, int>> q_split,
    torch::Tensor pinned_temp,
    std::vector<torch::Tensor> device_temp,
    std::vector<uintptr_t> streams
)
{
    std::vector<cudaStream_t> streams_;
    for (int i = 0; i < streams.size(); ++i)
        streams_.push_back(reinterpret_cast<cudaStream_t> (streams[i]));

    ExtTPContext* ctx = new ExtTPContext
    (
        kv_split,
        id_split,
        vc_split,
        rs_split,
        q_split,
        pinned_temp,
        device_temp,
        streams_
    );

    return reinterpret_cast<uintptr_t> (ctx);
}

void free_tp_context
(
    uintptr_t tp_context
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    delete ctx;
}

void tp_broadcast
(
    uintptr_t tp_context,
    torch::Tensor source,
    int broadcast_type,
    const py::list &targets,
    int dim
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    ctx->create_events();

    size_t size = source.numel() * 2;
//    DBGI(size);
    TORCH_CHECK(size <= ctx->pinned_size, "Temporary tensor is too small")

    void* source_g = NULL;

    int src_dev = source.device().index();
//    DBGI(src_dev);

    if (src_dev >= 0)
    {
        cudaSetDevice(src_dev);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        source_g = (void*) source.data_ptr();
        cuda_check(cudaMemcpyAsync(ctx->pinned_temp, source_g, size, cudaMemcpyDeviceToHost, stream));
//        DBGIX(src_dev, ctx->sync_events[src_dev]);
        cuda_check(cudaEventRecord(ctx->sync_events[src_dev], stream));
    }

    std::vector<std::tuple<int, int, int>> split;
    switch(broadcast_type)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    for (int i = 0; i < split.size(); ++i)
    {
        int dev = std::get<0>(split[i]);
        void* target = (void*) targets[i].cast<torch::Tensor>().data_ptr();
        if (target == source_g) continue;

        cudaSetDevice(dev);
        cudaStream_t stream = ctx->streams[dev];
        if (src_dev >= 0)
            cuda_check(cudaStreamWaitEvent(stream,ctx->sync_events[src_dev], 0));
        cuda_check(cudaMemcpyAsync(target, ctx->pinned_temp, size, cudaMemcpyHostToDevice, stream));

//        cuda_check(cudaEventRecord(ctx->sync_events2[i], ctx->streams[dev]));
    }

//    for (int i = 0; i < split.size(); ++i)
//    {
//        int dev = std::get<0>(split[i]);
//        cudaSetDevice(dev);
//
//        for (int j = 0; j < split.size(); ++j)
//        {
//            int dev2 = std::get<0>(split[j]);
//            if (dev == dev2) continue;
//            cuda_check(cudaStreamWaitEvent(ctx->streams[dev], ctx->sync_events2[dev2], 0));
//        }
//    }
}

void tp_gather
(
    uintptr_t tp_context,
    const py::list &inputs,
    int broadcast_type,
    const py::list &targets,
    int broadcast_type_target,
    int dim
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    ctx->create_events();

    std::vector<std::tuple<int, int, int>> split;
    switch(broadcast_type)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    int out_rows = inputs[0].cast<torch::Tensor>().size(0);
    int out_cols = std::get<2>(split[split.size() - 1]) * dim;
    int esize = inputs[0].cast<torch::Tensor>().element_size();

    for (int i = 0; i < split.size(); ++i)
    {
        int dev = std::get<0>(split[i]);
        uint8_t* src = (uint8_t*) inputs[i].cast<torch::Tensor>().data_ptr();
        int src_cols = inputs[i].cast<torch::Tensor>().size(1);
        uint8_t* dst = ((uint8_t*) ctx->pinned_temp) + std::get<1>(split[i]) * esize * dim;

        cudaSetDevice(dev);
        cuda_check(cudaMemcpy2DAsync
        (
            dst,
            out_cols * esize,
            src,
            src_cols * esize,
            src_cols * esize,
            out_rows,
            cudaMemcpyDeviceToHost,
            ctx->streams[dev]
        ));

//        DBGIX(i, ctx->sync_events3[i]);
        cuda_check(cudaEventRecord(ctx->sync_events3[i], ctx->streams[dev]));
    }

    for (int i = 0; i < split.size(); ++i)
    {
        int dev = std::get<0>(split[i]);
        cudaSetDevice(dev);

        for (int j = 0; j < split.size(); ++j)
        {
            int dev2 = std::get<0>(split[j]);
            if (dev == dev2) continue;
            cuda_check(cudaStreamWaitEvent(ctx->streams[dev], ctx->sync_events3[dev2], 0));
        }
    }

    if (!targets.size()) return;

    size_t size = targets[0].cast<torch::Tensor>().numel() * 2;

    switch(broadcast_type_target)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    for (int i = 0; i < split.size(); ++i)
    {
        int dev = std::get<0>(split[i]);
        void* target = (void*) targets[i].cast<torch::Tensor>().data_ptr();

        cudaSetDevice(dev);
        cudaStream_t stream = ctx->streams[dev];
        cuda_check(cudaMemcpyAsync(target, ctx->pinned_temp, size, cudaMemcpyHostToDevice, stream));
    }
}