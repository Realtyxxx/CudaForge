#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float shared[32]; // one entry per warp
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  float sum = warp_reduce_sum(val);
  if (lane == 0)
    shared[warp] = sum;
  __syncthreads();

  float result = 0.f;
  if (warp == 0) {
    float partial = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.f;
    result = warp_reduce_sum(partial);
  }
  result = __shfl_sync(0xffffffff, result, 0);
  return result;
}

// Block-per-(batch,inner) RMSNorm kernel.
__global__ void rmsnorm_kernel(const float *X, float *Y, int64_t batch,
                               int64_t features, int64_t inner, float eps) {
  int64_t row = static_cast<int64_t>(blockIdx.x);
  int64_t total_rows = batch * inner;
  if (row >= total_rows)
    return;

  int64_t batch_idx = row / inner;
  int64_t inner_idx = row - batch_idx * inner;
  int64_t base = batch_idx * features * inner + inner_idx;

  float thread_sum = 0.f;
  for (int64_t f = threadIdx.x; f < features; f += blockDim.x) {
    float v = X[base + f * inner];
    thread_sum += v * v;
  }
  float sum_sq = block_reduce_sum(thread_sum);

  __shared__ float shared_scale;
  if (threadIdx.x == 0) {
    float mean = sum_sq / static_cast<float>(features);
    shared_scale = rsqrtf(mean + eps);
  }
  __syncthreads();
  float scale = shared_scale;

  for (int64_t f = threadIdx.x; f < features; f += blockDim.x) {
    int64_t idx = base + f * inner;
    Y[idx] = X[idx] * scale;
  }
}

extern "C" void launch_kernel(const float *X, float *Y, int64_t batch,
                              int64_t features, int64_t inner, float eps) {
  if (batch <= 0 || features <= 0 || inner <= 0)
    return;
  const int threads = 256;
  int64_t rows = batch * inner;
  dim3 block(threads);
  dim3 grid(static_cast<unsigned>((rows + 1 - 1) / 1)); // one row per block
  rmsnorm_kernel<<<grid, block>>>(X, Y, batch, features, inner, eps);
}
