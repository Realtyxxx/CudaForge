import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import argparse

# Add workspace root to sys.path
sys.path.append("/home/tanyanxi/workspace/CudaForge")

from main import _run_single_task

def mock_llm_caller(prompt, sys_prompt=None, log_path=None, call_type="unknown", round_idx=-1):
    # Return the naive implementation as "generated" code
    return """
```cuda
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <stdexcept>
#include <stdint.h>

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
```
"""

def test_run_single_task():
    args = argparse.Namespace(
        gpu="Quadro RTX 6000",
        server_type="local",
        model_name="mock-model",
        max_tokens=1024,
        temperature=0.2,
        top_p=1.0,
        server_address="localhost",
        server_port=8000,
        round=2, # Run 2 rounds
        device=0,
        warmup=1,
        repeat=5,
        tol=1e-3,
        subproc_id=0
    )
    
    batch_dir = Path("run/test_batch").resolve()
    # Ensure batch dir exists
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    task_path = Path("CudaKernelBench/matmul.cuh").resolve()
    
    print(f"Testing _run_single_task with {task_path}...")
    
    # Patch _make_llm_caller to return our mock
    with patch("main._make_llm_caller", return_value=mock_llm_caller):
        res = _run_single_task(task_path, args, batch_dir)
        
    print("Result:", res)
    
    # Basic assertions
    if res["best_runnable"]:
        print("SUCCESS: Best kernel is runnable.")
    else:
        print("FAILURE: Best kernel is NOT runnable.")
        
    if res["best_score"] > 0:
        print(f"SUCCESS: Best score is positive ({res['best_score']}).")
    else:
        print(f"FAILURE: Best score is not positive ({res['best_score']}).")

if __name__ == "__main__":
    test_run_single_task()
