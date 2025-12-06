# CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization

A training-free multi-agent workflow for CUDA kernel generation and optimization, which is inspired by the iterative workflow of human experts, which contains steps such as developing initial kernels, testing correctness, analyzing hardware feedback, and iterative improvement.

<img src="./pic/human_agents_v2.png">

## 🔧 Build Environment

```
conda env create -f environment.yml
```

Note: Some packages may fail to install automatically when creating the environment.
If that happens, please install the missing packages manually using conda install or pip install.

```
pip install torch
pip install openai
pip install pandas
pip install matplotlib
```

## ⚙️ CUDA Toolkit and Build Tools

Please make sure that CUDA Toolkit and Ninja are correctly installed on your system.
Both nvcc (the CUDA compiler) and Nsight Compute (NCU) should be accessible and have matching versions with your installed CUDA Toolkit.

You can verify their availability and version consistency using:

```
nvcc --version
ncu --version
```

If either command is not found or versions are mismatched, please reinstall or update your CUDA Toolkit and Nsight Compute accordingly.

## 🔒 Permission Setup for Nsight Compute

Nsight Compute (ncu) requires elevated privileges to access GPU performance counters.
Please ensure that you have sudo or root privileges on the system.

If you need to enable password-less sudo for profiling commands, you can configure it as follows:

```
sudo visudo
```

Then add the following line (replace <username> with your actual user name):

```
<username> ALL=(ALL) NOPASSWD: ALL
```

After this setup, you can run profiling commands such as sudo ncu without being prompted for a password.

## 🚀 Run

Testing multiple tasks：

```bash
python3 main.py KernelBench/level1  --first_n 100  --gpu "Quadro RTX 6000"   --server_type openai   --model_name o3   --device 0   --round 10   --subproc_id 0

```

Testing single task：

```bash
python3 main.py KernelBench/level1/1_Square_matrix_multiplication_.py  --gpu "Quadro RTX 6000"   --server_type openai   --model_name o3   --device 0   --round 10   --subproc_id 0

```

----

## Begin

### Quick smoke tests (test_kernels)

Quick TVM FFI smoke tests for early validation, using registered input generators to check multiple `.cu` kernels for correctness.

- Prerequisites: `conda activate CudaForge`, ensure `apache-tvm` is installed and a GPU is available.
- Run: `python CudaKernelBench/test_kernels.py`
- Output: each kernel prints `correct: True/False, max error: <val>`; shows `All tests PASSED!` when everything succeeds.

### Benchmarking CUDA kernels (TVM FFI path)

Use `utils/compile_and_run.py` to compare a baseline `.cu` against an optimized one with registered input generators:

```bash
conda activate CudaForge
python utils/compile_and_run.py CudaKernelBench/attention.cu CudaKernelBench/new_flash_attention.cu --device 0 --warmup 3 --repeat 10 --tol 1e-4
python utils/compile_and_run.py CudaKernelBench/matmul.cu CudaKernelBench/new_matmul.cu --device 0 --warmup 3 --repeat 10 --tol 1e-4
```

Example result (attention vs new_flash_attention, GPU env in repo):

- max_abs_err: 2.384e-07, mean_abs_err: 3.37e-08 (outputs align)
- latency ms: ref avg 0.0272 (min 0.0256, max 0.0307); test avg 0.2036 (min 0.1956, max 0.2181)
