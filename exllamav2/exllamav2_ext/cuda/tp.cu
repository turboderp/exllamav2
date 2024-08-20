#include "rope.cuh"
#include "tp.cuh"
#include "util.cuh"

void init_tp_data(ExtTPData* tp_data)
{
    for (int i = 0; i < MAX_SYNC_STAGES; ++i)
        for (int j = 0; j < MAX_SYNC_DEVICES; ++j)
            tp_data->sync[i][j] = 0;
    tp_data->next_stage = 0;
}

// Not used atm, disabled for compatibility

//__global__ void cross_device_barrier_cuda_kernel
//(
//    uint32_t* sync,
//    uint32_t* sync_next,
//    uint32_t num_devices,
//    uint32_t device
//)
//{
//
//    // Arrive
//    sync[device] = 1;
//
//    // Clear flags for next barrier
//    if (device == 0)
//        for (int i = 0; i < MAX_SYNC_DEVICES; ++i)
//            sync_next[i] = 0;
//
//    // Wait for other devices to arrive
//    int delay = 5;
//    while (true)
//    {
//        int i = 0;
//        for (; i < num_devices; ++i) if (i != device && sync[i] == 0) break;
//        if (i == num_devices) break;
//        __nanosleep(delay);
//        delay = min(delay * 2, 500);
//        __threadfence_system();
//    }
//}

void cross_device_barrier_cuda
(
    cudaStream_t stream,
    uint32_t* sync,
    uint32_t* sync_next,
    uint32_t num_devices,
    uint32_t device
)
{
//    cross_device_barrier_cuda_kernel<<<1, 1, 0, stream>>>
//    (
//        sync,
//        sync_next,
//        num_devices,
//        device
//    );
}
