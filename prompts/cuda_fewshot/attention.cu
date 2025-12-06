#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <stdexcept>
#include <stdint.h>

// Naive attention kernel.
// Shapes: Q, K, V are [batch, heads, seq, dim]; O matches V.
__global__ void attention_kernel(const float *Q, const float *K, const float *V,
                                 float *O, int64_t batch, int64_t heads,
                                 int64_t seq, int64_t dim, float scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * heads * seq * dim;
  if (idx >= total)
    return;

  int64_t d = idx % dim;
  int64_t tmp = idx / dim;
  int64_t q_idx = tmp % seq;
  tmp /= seq;
  int64_t h = tmp % heads;
  int64_t b = tmp / heads;

  int64_t bh_offset = (b * heads + h) * seq * dim;
  int64_t q_base = bh_offset + q_idx * dim;

  // Softmax denominator across sequence dimension for this (b, h, q)
  float denom = 0.f;
  for (int64_t k = 0; k < seq; ++k) {
    const float *q_ptr = Q + q_base;
    const float *k_ptr = K + bh_offset + k * dim;
    float dot = 0.f;
    for (int64_t c = 0; c < dim; ++c) {
      dot += q_ptr[c] * k_ptr[c];
    }
    denom += expf(dot * scale);
  }
  denom = denom > 0.f ? denom : 1.f; // avoid division by zero

  // Weighted sum for this output element
  float out = 0.f;
  for (int64_t k = 0; k < seq; ++k) {
    const float *q_ptr = Q + q_base;
    const float *k_ptr = K + bh_offset + k * dim;
    float dot = 0.f;
    for (int64_t c = 0; c < dim; ++c) {
      dot += q_ptr[c] * k_ptr[c];
    }
    float weight = expf(dot * scale) / denom;
    const float *v_ptr = V + bh_offset + k * dim;
    out += weight * v_ptr[d];
  }

  O[idx] = out;
}

extern "C" void launch_kernel(const float *Q, const float *K, const float *V,
                              float *O, int64_t batch, int64_t heads,
                              int64_t seq, int64_t dim, float scale) {
  const int threads = 256;
  int64_t total = batch * heads * seq * dim;
  dim3 block(threads);
  dim3 grid(static_cast<unsigned>((total + threads - 1) / threads));
  attention_kernel<<<grid, block>>>(Q, K, V, O, batch, heads, seq, dim, scale);
}

