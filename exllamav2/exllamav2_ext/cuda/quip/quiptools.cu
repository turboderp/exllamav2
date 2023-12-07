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


// This is a terrible kernel, only use this to not call the pytorch version

#define DECOMPRESS_HI4B1C_BLOCK_SIZE 128

__global__ void cuda_decompress_hi4b1c_packed_kernel(
    const int32_t* __restrict__ YIs,     // m x (n/8)
    const c10::Half* __restrict__ CB,     // 16 x 1
    c10::Half* __restrict__ Y             // m x n
) {
  const long i = threadIdx.x + DECOMPRESS_HI4B1C_BLOCK_SIZE * blockIdx.x;

  // 0 2 4 6 1 3 5 7
  uint32_t packed = YIs[i];
  Y[i*8 + 7] = CB[packed & 15];
  Y[i*8 + 5] = CB[(packed >> 4) & 15];
  Y[i*8 + 3] = CB[(packed >> 8) & 15];
  Y[i*8 + 1] = CB[(packed >> 12) & 15];
  Y[i*8 + 6] = CB[(packed >> 16) & 15];
  Y[i*8 + 4] = CB[(packed >> 20) & 15];
  Y[i*8 + 2] = CB[(packed >> 24) & 15];
  Y[i*8 + 0] = CB[(packed >> 28) & 15];
}


void decompress_hi4b1c_packed(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,
    torch::Tensor &Y         // m x n
) {
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 8 == n);

  assert(CB.sizes()[0] == 16);
  assert(CB.sizes()[1] == 1);

  
  const dim3 threads(DECOMPRESS_HI4B1C_BLOCK_SIZE);
  const dim3 blocks(m*n/(8*DECOMPRESS_HI4B1C_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_hi4b1c_packed_kernel<<<blocks, threads, 0, stream>>>(
    YIs.data_ptr<int32_t>(),
    CB.data_ptr<c10::Half>(),
    Y.data_ptr<c10::Half>()
  );
}
