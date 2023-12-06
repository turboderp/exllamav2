#include <iostream>
#include <cassert>
#include <vector>
#include <utility>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/types.h>
#include <torch/extension.h>

using namespace torch::indexing;
using namespace nvcuda;

#define FULL_MASK 0xffffffff
#define HALF_MASK 0x0000ffff

#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) 	        do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(false)
#define gpuErrchk(ans)          do { gpuAssert((ans), __FILE__, __LINE__); } while (false)


__host__ static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}


__global__ void cuda_lookupmatmul_d4_k8_kernel(
    const c10::Half* __restrict__ X,      // k x n
    const uint8_t* __restrict__ YIs,      // m x (n/4)
    const c10::Half* __restrict__ CB,     // 256 x 4
    c10::Half* __restrict__ Z,            // k x m
    size_t K,
    size_t M,
    size_t N) {

  long m1 = blockIdx.x;
  long k1 = blockIdx.y;

  __shared__ c10::Half Y_cache[32*16];

  wmma::fragment<wmma::matrix_a, 8, 32, 16, __half, wmma::row_major> a;  // 8 x 16
  wmma::fragment<wmma::matrix_b, 8, 32, 16, __half, wmma::col_major> b;  // 32 x 16
  wmma::fragment<wmma::accumulator, 8, 32, 16, __half> c;                // 8 x 32
  fill_fragment(c, __float2half(0.0));

  for (long jn = 0; jn < N / 16; jn++) {
# pragma unroll 4
    for (long r = 0; r < 4; r++) {
      uint8_t yidxs = *(uint8_t*)(YIs + jn*(4*M) + m1*4*32 + threadIdx.x*4 + r);
      ((uint64_t*)Y_cache)[threadIdx.x*4 + r] = ((uint64_t*)CB)[(yidxs & 255)];
    }
    load_matrix_sync(a, (const __half*)(X + 8*N*k1 + 16*jn), N);
    load_matrix_sync(b, (const __half*)Y_cache, 16);
    mma_sync(c, a, b, c);
  }
  
  store_matrix_sync((__half*)(&Z[8*M*k1 + 32*m1]), c, M, wmma::mem_row_major);
}


void lookupmatmul_d4_k8(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
) {
  auto k = X.sizes()[0];
  auto m = YIs.sizes()[0];
  auto n = X.sizes()[1];

  assert(X.dtype() == torch::kFloat16);
  assert(YIs.dtype() == torch::kUInt8);
  assert(CB.dtype() == torch::kFloat16);
  assert(Z.dtype() == torch::kFloat16);

  assert(Z.sizes()[0] == k);
  assert(YIs.sizes()[1] * 4 == n);
  assert(Z.sizes()[1] == m);

  assert(k % 8 == 0); // if you want larger k, use k = 16
  assert(m % 32 == 0);
  assert(n % 16 == 0);

  const dim3 threads(32);
  const dim3 blocks(m/32,k/8);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_lookupmatmul_d4_k8_kernel<<<blocks, threads, 0, stream>>>(
    X.data_ptr<c10::Half>(),
    YIs.data_ptr<uint8_t>(),
    CB.data_ptr<c10::Half>(),
    Z.data_ptr<c10::Half>(),
    k,m,n
  );
}



__global__ void cuda_lookupmatmul_d4_k16_kernel(
    const c10::Half* __restrict__ X,      // k x n
    const uint8_t* __restrict__ YIs,      // m x (n/4)
    const c10::Half* __restrict__ CB,     // 256 x 4
    c10::Half* __restrict__ Z,            // k x m
    size_t K,
    size_t M,
    size_t N) {

  long m1 = blockIdx.x;
  long k1 = blockIdx.y;

  __shared__ c10::Half Y_cache[32*16];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;  
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;   
  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c0;               
  fill_fragment(c0, __float2half(0.0));

  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c1;    
  fill_fragment(c1, __float2half(0.0));

  for (long jn = 0; jn < N / 16; jn++) {
    for (long r = 0; r < 4; r++) {
      uint8_t yidxs = *(uint8_t*)(YIs + jn*(4*M) + m1*4*32 + threadIdx.x*4 + r);
      ((uint64_t*)Y_cache)[threadIdx.x*4 + r] = ((uint64_t*)CB)[(yidxs & 255)];
    }

    load_matrix_sync(a, (const __half*)(X + 16*N*k1 + 16*jn), N);

    load_matrix_sync(b, (const __half*)Y_cache, 16);
    mma_sync(c0, a, b, c0);
    
    load_matrix_sync(b, (const __half*)Y_cache + 16*16, 16);
    mma_sync(c1, a, b, c1);
  }
  
  store_matrix_sync((__half*)(&Z[16*M*k1 + 32*m1 +  0]), c0, M, wmma::mem_row_major);
  store_matrix_sync((__half*)(&Z[16*M*k1 + 32*m1 + 16]), c1, M, wmma::mem_row_major);
}


void lookupmatmul_d4_k16(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
) {
  auto k = X.sizes()[0];
  auto m = YIs.sizes()[0];
  auto n = X.sizes()[1];

  assert(X.dtype() == torch::kFloat16);
  assert(YIs.dtype() == torch::kUInt8);
  assert(CB.dtype() == torch::kFloat16);
  assert(Z.dtype() == torch::kFloat16);

  assert(Z.sizes()[0] == k);
  assert(YIs.sizes()[1] * 4 == n);
  assert(Z.sizes()[1] == m);

  assert(k % 16 == 0);
  assert(m % 32 == 0);
  assert(n % 16 == 0);

  const dim3 threads(32);
  const dim3 blocks(m/32,k/16);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_lookupmatmul_d4_k16_kernel<<<blocks, threads, 0, stream>>>(
    X.data_ptr<c10::Half>(),
    YIs.data_ptr<uint8_t>(),
    CB.data_ptr<c10::Half>(),
    Z.data_ptr<c10::Half>(),
    k,m,n
  );
}


__global__ void cuda_lookupmatmul_d4_k32_kernel(
    const c10::Half* __restrict__ X,      // k x n
    const uint8_t* __restrict__ YIs,      // m x (n/4)
    const c10::Half* __restrict__ CB,     // 256 x 4
    c10::Half* __restrict__ Z,            // k x m
    size_t K,
    size_t M,
    size_t N) {

  long m1 = blockIdx.x;
  long k1 = blockIdx.y;

  __shared__ c10::Half Y_cache[32*16];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;  
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;   
  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c0;               
  fill_fragment(c0, __float2half(0.0));

  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c1;    
  fill_fragment(c1, __float2half(0.0));

  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c2;    
  fill_fragment(c2, __float2half(0.0));

  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c3;    
  fill_fragment(c3, __float2half(0.0));

  for (long jn = 0; jn < N / 16; jn++) {
    for (long r = 0; r < 4; r++) {
      uint8_t yidxs = *(uint8_t*)(YIs + jn*(4*M) + m1*4*32 + threadIdx.x*4 + r);
      ((uint64_t*)Y_cache)[threadIdx.x*4 + r] = ((uint64_t*)CB)[(yidxs & 255)];
    }

    load_matrix_sync(a, (const __half*)(X + 16*N*(2*k1+0) + 16*jn), N);

    load_matrix_sync(b, (const __half*)Y_cache, 16);
    mma_sync(c0, a, b, c0);
    
    load_matrix_sync(b, (const __half*)Y_cache + 16*16, 16);
    mma_sync(c1, a, b, c1);

    load_matrix_sync(a, (const __half*)(X + 16*N*(2*k1+1) + 16*jn), N);
    mma_sync(c3, a, b, c3);

    load_matrix_sync(b, (const __half*)Y_cache, 16);
    mma_sync(c2, a, b, c2);
  }
  
  store_matrix_sync((__half*)(&Z[16*M*(2*k1+0) + 32*m1 +  0]), c0, M, wmma::mem_row_major);
  store_matrix_sync((__half*)(&Z[16*M*(2*k1+0) + 32*m1 + 16]), c1, M, wmma::mem_row_major);
  store_matrix_sync((__half*)(&Z[16*M*(2*k1+1) + 32*m1 +  0]), c2, M, wmma::mem_row_major);
  store_matrix_sync((__half*)(&Z[16*M*(2*k1+1) + 32*m1 + 16]), c3, M, wmma::mem_row_major);
}


void lookupmatmul_d4_k32(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
) {
  auto k = X.sizes()[0];
  auto m = YIs.sizes()[0];
  auto n = X.sizes()[1];

  assert(X.dtype() == torch::kFloat16);
  assert(YIs.dtype() == torch::kUInt8);
  assert(CB.dtype() == torch::kFloat16);
  assert(Z.dtype() == torch::kFloat16);

  assert(Z.sizes()[0] == k);
  assert(YIs.sizes()[1] * 4 == n);
  assert(Z.sizes()[1] == m);

  assert(k % 16 == 0);
  assert(m % 32 == 0);
  assert(n % 16 == 0);

  const dim3 threads(32);
  const dim3 blocks(m/32,k/32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_lookupmatmul_d4_k32_kernel<<<blocks, threads, 0, stream>>>(
    X.data_ptr<c10::Half>(),
    YIs.data_ptr<uint8_t>(),
    CB.data_ptr<c10::Half>(),
    Z.data_ptr<c10::Half>(),
    k,m,n
  );
}

#define DECOMPRESS_D4_BLOCK_SIZE 256

__global__ void cuda_decompress_d4_origorder_kernel(
    const uint8_t* __restrict__ YIs,      // m x (n/4)
    const c10::Half* __restrict__ CB,           // 256 x 4
    c10::Half* __restrict__ Y             // m x n
) {
  const long i = threadIdx.x + DECOMPRESS_D4_BLOCK_SIZE * blockIdx.x;

  for(long r = 0; r < 4; r++) {
    uint8_t yidx = ((uint8_t*)YIs)[i*4 + r];
    ((uint64_t*)Y)[i*4 + r] = ((uint64_t*)CB)[yidx & 255];
  }
}


void decompress_d4_origorder(
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Y         // m x n
) {
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(CB.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 4 == n);
  assert(CB.sizes()[0] == 256);
  assert(CB.sizes()[1] == 4);

  const dim3 threads(DECOMPRESS_D4_BLOCK_SIZE);
  const dim3 blocks(m*n/(16*DECOMPRESS_D4_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_d4_origorder_kernel<<<blocks, threads, 0, stream>>>(
    YIs.data_ptr<uint8_t>(),
    CB.data_ptr<c10::Half>(),
    Y.data_ptr<c10::Half>()
  );
}


__global__ void cuda_decompress_d4_kernel(
    const uint8_t* __restrict__ YIs,      // m x (n/4)
    const c10::Half* __restrict__ CB,     // 256 x 4
    c10::Half* __restrict__ Y,            // m x n
    size_t M,
    size_t N
) {
  const long i = threadIdx.x + DECOMPRESS_D4_BLOCK_SIZE * blockIdx.x;

  const long j = (i % (N/16))*M + (i / (N/16));

  for(long r = 0; r < 4; r++) {
    uint8_t yidx = ((uint8_t*)YIs)[j*4 + r];
    ((uint64_t*)Y)[i*4 + r] = ((uint64_t*)CB)[yidx & 255];
  }
}


void decompress_d4(
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Y         // m x n
) {
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(CB.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 4 == n);
  assert(CB.sizes()[0] == 256);
  assert(CB.sizes()[1] == 4);

  const dim3 threads(DECOMPRESS_D4_BLOCK_SIZE);
  const dim3 blocks(m*n/(16*DECOMPRESS_D4_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_d4_kernel<<<blocks, threads, 0, stream>>>(
    YIs.data_ptr<uint8_t>(),
    CB.data_ptr<c10::Half>(),
    Y.data_ptr<c10::Half>(),
    m,n
  );
}


#define DECOMPRESS_E8P_BLOCK_SIZE 256
#define FLIP_MASK 9223512776490647552LLU // (1 << 63) + (1 << 47) + (1 << 31) + (1 << 15)

__global__ void cuda_decompress_e8p_origorder_kernel(
    const int16_t* __restrict__ YIs,      // m x (n/8)
    const c10::Half* __restrict__ CB, // 256 x 8
    const bool* __restrict__ CB_even_flips, 
    c10::Half* __restrict__ Y             // m x n
) {
  const long i = threadIdx.x + DECOMPRESS_E8P_BLOCK_SIZE * blockIdx.x;

  uint16_t yidx = ((uint16_t*)YIs)[i] - 32768;
  uint16_t abs_idx = (yidx & 65280) >> 8;
  uint16_t flips = (yidx & 254) >> 1;
  flips |= (((__popc(flips) & 1) == CB_even_flips[abs_idx]) << 7);
  
  ((uint64_t*)Y)[i*2] = ((uint64_t*)CB)[abs_idx*2];
  uint64_t l4flips = (uint64_t)(flips >> 4);
  l4flips |= (l4flips << 34);
  l4flips |= (l4flips << 17);
  l4flips = (l4flips << 12);
  l4flips &= FLIP_MASK;
  ((uint64_t*)Y)[i*2] |= l4flips;
  
  ((uint64_t*)Y)[i*2 + 1] = ((uint64_t*)CB)[abs_idx*2 + 1];
  uint64_t r4flips = (uint64_t)(flips & 15);
  r4flips |= (r4flips << 34);
  r4flips |= (r4flips << 17);
  r4flips = (r4flips << 12);
  r4flips &= FLIP_MASK;
  ((uint64_t*)Y)[i*2 + 1] |= r4flips;
  
  __half2 const shift = (yidx & 1 ? __half2half2((c10::Half)0.25) : __half2half2((c10::Half)-0.25));
# pragma unroll 4
  for(long k = 0; k < 4; k++){
    ((__half2*)Y)[i*4 + k] = __hadd2(((__half2*)Y)[i*4 + k], shift);
  }
}


void decompress_e8p_origorder(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,       // 256 x 8
    torch::Tensor CB_even_flips, // 256
    torch::Tensor &Y         // m x n
) {
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(CB.is_contiguous());
  assert(CB_even_flips.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 8 == n);
  assert(CB.sizes()[0] == 256);
  assert(CB.sizes()[1] == 8);
  assert(CB_even_flips.sizes()[0] == 256);
  
  const dim3 threads(DECOMPRESS_E8P_BLOCK_SIZE);
  const dim3 blocks(m*n/(8*DECOMPRESS_E8P_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_e8p_origorder_kernel<<<blocks, threads, 0, stream>>>(
    YIs.data_ptr<int16_t>(),
    CB.data_ptr<c10::Half>(),
    CB_even_flips.data_ptr<bool>(),
    Y.data_ptr<c10::Half>()
  );
}


#define BLOCK_SIZE 512
#define WARP_SIZE 32


__device__ static inline uint64_t decode8weights(
    uint16_t weight_compressed,
    const int64_t *__restrict__ codebook_abs
) {

    bool bit_shift = !(weight_compressed & 1);
    uint8_t bits_sign = (weight_compressed >> 1) & ((1 << 7) - 1);
    uint8_t bits_abs = (weight_compressed >> 8) & ((1 << 9) - 1);

    int64_t packed = codebook_abs[bits_abs];

    // TODO: optimize this by redefining the bit pattern
    bool parity = __popcll(packed & 0x0404040404040404) % 2 == 0;
    uint64_t decoded_sign = __brev(bits_sign | (((__popc(bits_sign) & 1) == parity) << 7)) >> 24;
    decoded_sign |= (decoded_sign << (32-4));
    decoded_sign |= (decoded_sign << (16-2));
    decoded_sign |= (decoded_sign << (8-1));
    decoded_sign &= 0x0101010101010101;
    decoded_sign *= 255 - 3;
    packed ^= decoded_sign;

    packed -= bit_shift * 0x0202020202020202;
    packed |= 0x0101010101010101;

    return packed;
}


/*
llama 2 70B:
M N K
1 8192 8192
1 57344 8192
1 8192 28672
1 10240 8192
*/
template <typename scalar_t>
__global__ static void
__launch_bounds__(BLOCK_SIZE)
decode_matmul_e8p_kernel(
    scalar_t *__restrict__ output,
    const scalar_t *__restrict__ x,
    const int16_t *__restrict__ weights_compressed,
    const int64_t *__restrict__ codebook_abs,
    int64_t M,
    int64_t N,
    int64_t K
) {
    __shared__ int64_t codebook_local[256];
    if (threadIdx.x < 256) {
    codebook_local[threadIdx.x] = codebook_abs[threadIdx.x];
    }
    __syncthreads();

    int64_t warpId = threadIdx.x / WARP_SIZE;
    int64_t laneId = threadIdx.x % WARP_SIZE;

    // each thread adds 8 activation-weight products
    int64_t unroll_k = 2;
    int64_t pack = 8;
    int64_t elem_per_thread = pack * unroll_k;
    int64_t warps_per_elem = K / WARP_SIZE / elem_per_thread;
    int64_t unroll_n = 16;
    int64_t local_k = 1; // in terms of warp size. 32 threads of elem_per_thread fma each, dont set below 1 because of __shfl_down_sync
    int64_t local_n = BLOCK_SIZE / WARP_SIZE / local_k;
    int64_t grid_N = N / unroll_n;

    __shared__ scalar_t accum_scratch[BLOCK_SIZE / WARP_SIZE];
    bool SHARED_REDUCE = false;

    for (int64_t warpPos = blockIdx.x * BLOCK_SIZE/WARP_SIZE + warpId;
            warpPos < M * grid_N * warps_per_elem;
            warpPos += gridDim.x * BLOCK_SIZE/WARP_SIZE) {

        int64_t local_n_i = (warpPos% (BLOCK_SIZE / WARP_SIZE)) / local_k;
        int64_t local_k_i = (warpPos% (BLOCK_SIZE / WARP_SIZE)) % local_k;
        int64_t m = (warpPos / warps_per_elem) / (grid_N);
        int64_t k_ = warpPos % (warps_per_elem * local_n);
        int64_t k = k_ / (local_k * local_n) * local_k + k_ % local_k;

#pragma unroll
        for (int64_t unroll_n_i = 0; unroll_n_i < unroll_n; unroll_n_i++) {
            scalar_t accumulator = 0;
            int64_t n = ((warpPos/local_k) % local_n) + ((warpPos / warps_per_elem) % grid_N) / local_n * local_n;
            __syncwarp();
#pragma unroll
            for (int64_t unroll_k_i = 0; unroll_k_i < unroll_k; unroll_k_i++) {
                // TODO: optimize access pattern by reordering weights
                const scalar_t *activations = x + m * K + (k * WARP_SIZE + laneId) * elem_per_thread + unroll_k_i * pack;
                uint16_t encoded = weights_compressed[(n*unroll_n + unroll_n_i) * K/pack + (k * WARP_SIZE + laneId) * unroll_k + unroll_k_i];
                uint64_t decoded = decode8weights(encoded, codebook_local);

                if constexpr (std::is_same<scalar_t, float>::value) {
                    const float4 *first_half = reinterpret_cast<const float4 *>(activations);
                    accumulator += first_half->x * static_cast<int8_t>(decoded >> 0);
                    accumulator += first_half->y * static_cast<int8_t>(decoded >> 8);
                    accumulator += first_half->z * static_cast<int8_t>(decoded >> 16);
                    accumulator += first_half->w * static_cast<int8_t>(decoded >> 24);
                    const float4 *second_half = reinterpret_cast<const float4 *>(activations + 4);
                    accumulator += second_half->x * static_cast<int8_t>(decoded >> 32);
                    accumulator += second_half->y * static_cast<int8_t>(decoded >> 40);
                    accumulator += second_half->z * static_cast<int8_t>(decoded >> 48);
                    accumulator += second_half->w * static_cast<int8_t>(decoded >> 56);
                } else {
#pragma unroll
                    for (int64_t i = 0; i < 8; i += 1) {
                        int8_t weight = decoded >> (i * 8);
                        accumulator += activations[i] * weight;
                    }
                }
            }
            accumulator *= 0.25;

            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                // apparently c10::Half does arithmetic operations in float32?
                // https://github.com/pytorch/pytorch/blob/0bd4d1f4ab38d3088de8aa5fbba35427b42d118e/c10/util/Half.h#L4C58-L6C80
                if constexpr (std::is_same<scalar_t, c10::Half>::value) {
                    accumulator += __shfl_down_sync(0xFFFFFFFF, __float2half(accumulator), offset);
                } else {
                    accumulator += __shfl_down_sync(0xFFFFFFFF, accumulator, offset);
                }
            }

            if (SHARED_REDUCE) {
                if (laneId == 0) {
                    accum_scratch[warpId] = accumulator;
                    __syncthreads();
                    if (warpId % local_k == 0) {
                        scalar_t local_accum = 0;
                        for (int64_t accum_i = 0; accum_i < local_k; accum_i++) {
                            local_accum += accum_scratch[warpId / local_k * local_k + accum_i];
                        }
                        atomicAdd(output + m * N + n * unroll_n + unroll_n_i, local_accum);
                    }
                } else {
                    __syncthreads();
                }
            } else {
                if (laneId == 0) {
                    atomicAdd(output + m * N + n * unroll_n + unroll_n_i, accumulator);
                }
            }
        }
    }
}


__host__ extern torch::Tensor decode_matmul_e8p(
    torch::Tensor x,
    torch::Tensor weights_compressed,
    torch::Tensor codebook_abs
) {

    CHECK_INPUT(x);
    CHECK_INPUT(weights_compressed);
    CHECK_INPUT(codebook_abs);

    TORCH_CHECK(weights_compressed.scalar_type() == torch::kInt16);
    TORCH_CHECK(codebook_abs.scalar_type() == torch::kInt64);
    TORCH_CHECK(x.size(-1) == weights_compressed.size(-1) << 3);
    TORCH_CHECK(codebook_abs.size(-1) == 256);

    int64_t M = x.size(-2);
    int64_t N = weights_compressed.size(-2);
    int64_t K = x.size(-1);
    //printf("%lld %lld %lld\n", M, N, K);

    TORCH_CHECK(K % WARP_SIZE == 0, "K is not divisible by WARP_SIZE");

    at::DeviceGuard guard(x.device());
    torch::TensorOptions options = torch::TensorOptions()
        .dtype(x.scalar_type())
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    torch::Tensor output = torch::zeros(std::vector<int64_t>{M, N}, options);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, x.get_device());
    int64_t grid_size = static_cast<int64_t>(6 * deviceProp.multiProcessorCount);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            x.scalar_type(),
            "decode_matmul_e8p",
            [&] {
        decode_matmul_e8p_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                output.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                weights_compressed.data_ptr<int16_t>(),
                codebook_abs.data_ptr<int64_t>(),
                M,
                N,
                K);
        gpuErrchk(cudaPeekAtLastError());
    });

    return output;
}
