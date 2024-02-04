#include "layer_norm.cuh"
#include "util.cuh"

#if defined(USE_ROCM)
#define __shfl_xor_sync(mask, var, laneMask) __shfl_xor(var, laneMask)
#define NUM_WARPS (1024 / warpSize)
#define WARP_SIZE (warpSize)
#else
#define NUM_WARPS 32
#define WARP_SIZE 32
#endif

// y = x * w / sqrt(row_mean(x * x) + epsilon)

#define BLOCK_SIZE WARP_SIZE
#define NUM_THREADS (NUM_WARPS * WARP_SIZE)

typedef void (*fp_layer_norm_kernel)
(
    const half*,
    const half*,
    const half*,
    half*,
    const float,
    const float,
    const int,
    const int
);

template <int blocks_per_warp>
__global__ void layer_norm_kernel
(
    const half* __restrict__ x,
    const half* __restrict__ w,
    const half* __restrict__ b,
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

    float itemf[blocks_per_warp];

    // Compute sum for each block

    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < blocks_per_warp; i++)
    {
        int column = warp_id * WARP_SIZE + lane_id + NUM_THREADS * i;
        if (column >= dim) break;

        float f = __half2float(x_row[column]);
        f = fmaxf(-65504.0f, fminf(f, 65504.0f));
        itemf[i] = f;
        sum += f;
    }

    // Shuffle to sum across lanes

    __shared__ float sums[NUM_WARPS];

    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane_id == 0) sums[warp_id] = sum;
    __syncthreads();

    // Load partial sums from across warps, shuffle again across lanes

    sum = sums[lane_id];
    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);

    // Compute mean

    float mean = sum * r_dim;

    // Compute square of distance to mean

    sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < blocks_per_warp; i++)
    {
        int column = warp_id * WARP_SIZE + lane_id + NUM_THREADS * i;
        if (column >= dim) break;

        float f = itemf[i];
        f = fmaxf(-65504.0f, fminf(f, 65504.0f));
        f -= mean;
        itemf[i] = f;
        sum = fma(f, f, sum);
    }

    // Shuffle to sum across lanes

    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane_id == 0) sums[warp_id] = sum;
    __syncthreads();

    // Load partial sums from across warps, shuffle again across lanes

    sum = sums[lane_id];
    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);

    // Get 1/sqrt(variance)

    float rsvar = rsqrtf(sum * r_dim + epsilon);

    // Normalize x, scaling by w

    #pragma unroll 4
    for (int i = 0; i < blocks_per_warp; i++)
    {
        int column = warp_id * WARP_SIZE + lane_id + NUM_THREADS * i;
        if (column >= dim) return;

        float x_itemf = itemf[i];
        float w_itemf = __half2float(w[column]);
        float n = x_itemf * w_itemf * rsvar;

        half nh = __float2half_rn(n);
        if (b) nh = __hadd(nh, b[column]);  // Optional bias

        y_row[column] = nh;
    }
}

fp_layer_norm_kernel pick_layer_norm_kernel(const int blocks_per_warp)
{
    if (blocks_per_warp == 1) return layer_norm_kernel<1>;
    if (blocks_per_warp == 2) return layer_norm_kernel<2>;
    if (blocks_per_warp == 3) return layer_norm_kernel<3>;
    if (blocks_per_warp == 4) return layer_norm_kernel<4>;
    if (blocks_per_warp == 5) return layer_norm_kernel<5>;
    if (blocks_per_warp == 6) return layer_norm_kernel<6>;
    if (blocks_per_warp == 7) return layer_norm_kernel<7>;
    if (blocks_per_warp == 8) return layer_norm_kernel<8>;
	return NULL;
}


void layer_norm_cuda
(
    const half* x,
    const half* w,
    const half* b,
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
    fp_layer_norm_kernel kernel = pick_layer_norm_kernel(blocks_per_warp);
    kernel<<<gridDim, blockDim>>>(x, w, b, y, epsilon, r_dim, rows, dim);
}
