#include "rms_norm.cuh"
#include "util.cuh"

#if defined(USE_ROCM)
#define __shfl_xor_sync(mask, var, laneMask) __shfl_xor(var, laneMask)
#endif

// y = x * w / sqrt(row_mean(x * x) + epsilon)

#define NUM_WARPS 32
#define WARP_SIZE 32
#define BLOCK_SIZE WARP_SIZE
#define NUM_THREADS (NUM_WARPS * WARP_SIZE)

typedef void (*fp_rms_norm_kernel)
(
    const half*,
    const half*,
    half*,
    const float,
    const float,
    const int,
    const int
);

template <int blocks_per_warp>
__global__ void rms_norm_kernel
(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ y,
    const float epsilon,
    const float r_dim,
    const int rows,
    const int dim
)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x;
    const half* x_row = x + row * dim;
    half* y_row = y + row * dim;

    //int blocks_per_warp = DIVIDE(dim, NUM_THREADS);

    // Compute sum of squares for each block

    float sum = 0.0f;
    float itemf[blocks_per_warp];

    #pragma unroll
    for (int i = 0; i < blocks_per_warp; i++)
    {
        int column = warp_id * WARP_SIZE + lane_id + NUM_THREADS * i;
        if (column >= dim) break;

        float f = __half2float(x_row[column]);
        f = fmaxf(-65504.0f, fminf(f, 65504.0f));
        itemf[i] = f;
        sum = fma(f, f, sum);
    }

    // Shuffle to sum across lanes

    __shared__ float sums[NUM_WARPS];

    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane_id == 0) sums[warp_id] = sum;
    __syncthreads();

    // Load partial sums from across warps, shuffle again across lanes

    sum = sums[lane_id];
    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);

    // Get norm

    float rmf = rsqrtf(sum * r_dim + epsilon);

    // Normalize x, scaling by w

    #pragma unroll 4
    for (int i = 0; i < blocks_per_warp; i++)
    {
        int column = warp_id * WARP_SIZE + lane_id + NUM_THREADS * i;
        if (column >= dim) return;

        float x_itemf = itemf[i];
        float w_itemf = __half2float(w[column]);
        float n = x_itemf * w_itemf * rmf;
        y_row[column] = __float2half_rn(n);
    }
}

fp_rms_norm_kernel pick_rms_norm_kernel(const int blocks_per_warp)
{
    if (blocks_per_warp == 1) return rms_norm_kernel<1>;
    if (blocks_per_warp == 2) return rms_norm_kernel<2>;
    if (blocks_per_warp == 3) return rms_norm_kernel<3>;
    if (blocks_per_warp == 4) return rms_norm_kernel<4>;
    if (blocks_per_warp == 5) return rms_norm_kernel<5>;
    if (blocks_per_warp == 6) return rms_norm_kernel<6>;
    if (blocks_per_warp == 7) return rms_norm_kernel<7>;
    if (blocks_per_warp == 8) return rms_norm_kernel<8>;
	return NULL;
}


void rms_norm_cuda
(
    const half* x,
    const half* w,
    half* y,
    const float epsilon,
    const int rows,
    const int dim
)
{
    dim3 blockDim, gridDim;
    blockDim.x = NUM_THREADS;
    blockDim.y = 1;
    gridDim.x = rows;
    gridDim.y = 1;

    float r_dim = 1.0f / (float) dim;

    int blocks_per_warp = DIVIDE(dim, NUM_THREADS);
    fp_rms_norm_kernel kernel = pick_rms_norm_kernel(blocks_per_warp);
    kernel<<<gridDim, blockDim>>>(x, w, y, epsilon, r_dim, rows, dim);
}
