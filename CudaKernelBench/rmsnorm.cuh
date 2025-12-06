#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <stdexcept>
#include <stdint.h>

// Naive RMSNorm across the feature dimension.
__global__ void rmsnorm_kernel(const float *X, float *Y, int64_t batch,
                               int64_t features, int64_t inner, float eps) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * features * inner;
  if (idx < total) {
    int64_t inner_idx = idx % inner;
    int64_t tmp = idx / inner;
    int64_t feature_idx = tmp % features;
    int64_t batch_idx = tmp / features;

    int64_t base = (batch_idx * features * inner) + inner_idx;
    float sum_sq = 0.f;
    for (int64_t f = 0; f < features; ++f) {
      float v = X[base + f * inner];
      sum_sq += v * v;
    }
    float scale = rsqrtf(sum_sq / static_cast<float>(features) + eps);
    Y[idx] = X[idx] * scale;
  }
}

extern "C" void launch_kernel(const float *X, float *Y, int64_t batch,
                              int64_t features, int64_t inner, float eps) {
  const int threads = 256;
  int64_t total = batch * features * inner;
  dim3 block(threads);
  dim3 grid(static_cast<unsigned>((total + threads - 1) / threads));
  rmsnorm_kernel<<<grid, block>>>(X, Y, batch, features, inner, eps);
}
