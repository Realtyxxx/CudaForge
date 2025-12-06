#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <stdexcept>
#include <stdint.h>


// Naive matrix multiplication: C[M, N] = A[M, K] @ B[K, N]
__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              int64_t M, int64_t N, int64_t K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.f;
    int64_t a_base = row * K;
    int64_t b_base = col;
    for (int64_t k = 0; k < K; ++k) {
      acc += A[a_base + k] * B[k * N + b_base];
    }
    C[row * N + col] = acc;
  }
}

extern "C" void launch_kernel(const float *A, const float *B, float *C,
                              int64_t M, int64_t N, int64_t K) {
  const dim3 block(16, 16);
  const dim3 grid((static_cast<unsigned>(N) + block.x - 1) / block.x,
                  (static_cast<unsigned>(M) + block.y - 1) / block.y);
  matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
