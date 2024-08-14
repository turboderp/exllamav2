#ifndef _tp_cuh
#define _tp_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#define MAX_SYNC_DEVICES 8
#define MAX_SYNC_STAGES 3

struct ExtTPData
{
    uint32_t sync[MAX_SYNC_STAGES][MAX_SYNC_DEVICES];
    uint32_t next_stage;
};

void init_tp_data(ExtTPData* tp_data);

void cross_device_barrier_cuda
(
    cudaStream_t stream,
    uint32_t* sync,
    uint32_t* sync_next,
    uint32_t num_devices,
    uint32_t device
);

#endif