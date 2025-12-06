from __future__ import annotations

import tvm_ffi

from tvm_ffi import Module, load_module
"""
compare_and_bench.py – benchmark reference PyTorch vs. standalone CUDA `.cu`.

Pipeline
--------
* Dynamically imports the **reference** PyTorch model (for inputs + gold outputs).
* Compiles the candidate `.cu` file with `nvcc` to a shared object.
* Loads `launch_kernel` via TVM FFI, feeds tensors via DLPack, benchmarks, and
  compares outputs against the reference.
"""

import argparse
import contextlib
import ctypes
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn

# TVM FFI imports
try:
    import tvm
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    tvm = None

# Optional TVM-FFI (stable C ABI) support for zero-copy DLPack interop.
try:
    import tvm_ffi  # type: ignore
    TVM_FFI_AVAILABLE = True
except ImportError:
    TVM_FFI_AVAILABLE = False
    tvm_ffi = None

from utils.kernel_signature import (
    KernelSignature, ArgType, ArgRole,
)
from utils.input_generators import build_registered_inputs
from utils.codegen_utils import generate_tvm_wrapper

# ---------------------------------------------------------------------------

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class _ModuleCacheEntry:
    mtime_ns: int
    size: int
    module: Module
    so_path: Path


_MODULE_CACHE: Dict[str, _ModuleCacheEntry] = {}
_MODULE_CACHE_LOCK = Lock()

# ---------------------------------------------------------------------------


def _has_tvm_wrapper(src: str) -> bool:
    """粗略检查源码是否已包含 TVM FFI 封装，避免重复生成。"""
    return "TVM_FFI_DLL_EXPORT_TYPED_FUNC" in src or "tvm/ffi" in src


class KernelRunner:
    """编译单个 .cu(?h) 文件，并提供基于 TVM FFI 的可调用入口。"""

    def __init__(
        self,
        cu_filename: str | Path,
        function_name: str = "launch_kernel_tvm",
        *,
        signature: KernelSignature | None = None,
        user_launch_name: str = "launch_kernel",
    ):
        cu_path = Path(cu_filename)
        if not cu_path.suffix:
            cu_path = cu_path.with_suffix(".cuh")

        if not cu_path.is_absolute():
            cu_path = Path(__file__).parent / ".." / "CudaKernelBench" / cu_path

        self._temp_cu: Path | None = None
        if signature is not None:
            if not cu_path.exists():
                raise FileNotFoundError(cu_path)
            raw_src = cu_path.read_text()
            if not _has_tvm_wrapper(raw_src):
                wrapper = generate_tvm_wrapper(signature, user_launch_name=user_launch_name)
                combined = raw_src.rstrip() + "\n\n" + wrapper + "\n"
                tmp = tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False)
                tmp.write(combined)
                tmp.flush()
                tmp.close()
                cu_path = Path(tmp.name)
                self._temp_cu = cu_path

        self.cu_path = cu_path
        self.function_name = function_name

        self.mod = compile_and_load(self.cu_path)
        self.run = getattr(self.mod, function_name)

    def __del__(self):
        try:
            if self._temp_cu is not None and self._temp_cu.exists():
                self._temp_cu.unlink()
        except Exception:
            pass

    def __call__(self, inputs: list,
                 output: torch.Tensor,
                 signature: KernelSignature,
                 workspace: Optional[List[torch.Tensor]] = None,
                 ref_model=None):
        """通过 TVM FFI 启动 kernel。"""
        ffi_args = _build_signature_args_tvm_ffi(signature, inputs, output, workspace, ref_model)
        self.run(*ffi_args)


# =========================== Error Handling ===============================
class CompilationError(RuntimeError):
    """动态导入或 nvcc 构建失败时抛出，首个参数是完整构建日志。"""


class AccuracyError(RuntimeError):
    """输出未满足精度阈值时抛出。"""


# =========================== CUDA compile/load helpers ====================
def compile_and_load(cu_path: Path) -> Module:
    """用 nvcc 将 .cu 编译为共享库。"""
    if not cu_path.exists():
        raise FileNotFoundError(cu_path)

    cu_path = cu_path.resolve()
    src_stat = cu_path.stat()
    cache_key = str(cu_path)

    with _MODULE_CACHE_LOCK:
        cached = _MODULE_CACHE.get(cache_key)
        if (
            cached is not None
            and cached.mtime_ns == src_stat.st_mtime_ns
            and cached.size == src_stat.st_size
            and cached.so_path.exists()
        ):
            return cached.module
        if cached is not None and not cached.so_path.exists():
            _MODULE_CACHE.pop(cache_key, None)

    so_path = cu_path.with_suffix(".so")
    tvm_inc = []
    system_include = []
    project_root = Path(__file__).resolve().parent.parent
    bench_inc_dir = project_root / "CudaKernelBench"
    bench_inc = [f"-I{bench_inc_dir}"] if bench_inc_dir.exists() else []
    if TVM_AVAILABLE and tvm is not None:
        tvm_root = Path(tvm.__file__).resolve().parent
        inc1 = tvm_root / "include"
        inc2 = tvm_root / "3rdparty" / "dlpack" / "include"
        inc3 = tvm_root / "3rdparty" / "dmlc-core" / "include"
        tvm_inc = [f"-I{inc}" for inc in (inc1, inc2, inc3) if inc.exists()]
        import sysconfig
        system_include_path = [
            # sysconfig.get_path("include"),  # 如需默认 Python include 可取消注释
            "$cuda_home/include",
            "$cuda_home/include/cccl",
            tvm_ffi.libinfo.find_include_path(),
            tvm_ffi.libinfo.find_dlpack_include_path(),
        ]
        # 保持 include 参数连续，避免 nvcc 解析问题。
        system_include = [f"-I{sys_inc}" for sys_inc in system_include_path if sys_inc is not None]
    elif TVM_FFI_AVAILABLE and tvm_ffi is not None:
        # 回退：仅安装 tvm_ffi 时仍需 dlpack/tvm 头文件。
        system_include_path = [
            "$cuda_home/include",
            "$cuda_home/include/cccl",
            tvm_ffi.libinfo.find_include_path(),
            tvm_ffi.libinfo.find_dlpack_include_path(),
        ]
        system_include = [f"-I{sys_inc}" for sys_inc in system_include_path if sys_inc is not None]

    needs_rebuild = True
    if so_path.exists():
        try:
            so_stat = so_path.stat()
            needs_rebuild = so_stat.st_mtime_ns < src_stat.st_mtime_ns
        except OSError:
            needs_rebuild = True

    if needs_rebuild:
        cmd = [
            "nvcc",
            "-shared",
            "-arch=sm_80",
            "-O3",
            "-Xcompiler",
            "-fPIC",
            *bench_inc,
            *system_include,
            *tvm_inc,
            str(cu_path),
            "-o",
            str(so_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise CompilationError("nvcc not found in PATH") from exc
        log = "".join([proc.stdout or "", proc.stderr or ""]).strip()
        if proc.returncode != 0:
            raise CompilationError(log)
        if not so_path.exists():
            raise CompilationError(log or "nvcc succeeded but no output .so produced.")

    mod = load_module(str(so_path))
    with _MODULE_CACHE_LOCK:
        _MODULE_CACHE[cache_key] = _ModuleCacheEntry(
            mtime_ns=src_stat.st_mtime_ns,
            size=src_stat.st_size,
            module=mod,
            so_path=so_path,
        )
    return mod


# =========================== TVM FFI Implementation ==========================

def _torch_to_tvm_ffi(tensor: torch.Tensor) -> Any:
    """将 PyTorch 张量转换为 tvm_ffi NDArray（经 DLPack）。"""
    if not TVM_FFI_AVAILABLE or tvm_ffi is None:
        raise RuntimeError("tvm_ffi is not available.")
    if not hasattr(tvm_ffi, "from_dlpack"):
        raise RuntimeError("tvm_ffi.from_dlpack not found; please upgrade tvm_ffi.")
    from torch.utils import dlpack as torch_dlpack  # 惰性导入以减少开销
    return tvm_ffi.from_dlpack(torch_dlpack.to_dlpack(tensor))


def _build_signature_args_tvm_ffi(
    signature: KernelSignature,
    inputs: List[torch.Tensor],
    output: torch.Tensor,
    workspace: Optional[List[torch.Tensor]] = None,
    ref_model: Any = None,
) -> List:
    """依据 KernelSignature 构建 tvm_ffi 调用参数列表。"""
    if not TVM_FFI_AVAILABLE or tvm_ffi is None:
        raise RuntimeError("tvm_ffi is not available.")

    ffi_args: List[Any] = []
    input_idx = 0
    workspace_idx = 0

    for arg_spec in signature.args:
        if arg_spec.arg_type == ArgType.TENSOR:
            if arg_spec.role == ArgRole.INPUT:
                tensor = inputs[input_idx]
                input_idx += 1
            elif arg_spec.role == ArgRole.OUTPUT:
                tensor = output
            elif arg_spec.role == ArgRole.WORKSPACE:
                if workspace is None or workspace_idx >= len(workspace):
                    raise RuntimeError(
                        f"Workspace tensor '{arg_spec.name}' required but not provided"
                    )
                tensor = workspace[workspace_idx]
                workspace_idx += 1
            else:
                raise RuntimeError(f"Unsupported tensor role {arg_spec.role}")
            ffi_args.append(_torch_to_tvm_ffi(tensor))
        elif arg_spec.arg_type == ArgType.INT:
            if arg_spec.compute_fn is None:
                raise RuntimeError(f"INT argument '{arg_spec.name}' has no compute_fn defined")
            ffi_args.append(int(arg_spec.compute_fn(inputs, output, ref_model)))
        elif arg_spec.arg_type == ArgType.FLOAT:
            if arg_spec.compute_fn is None:
                raise RuntimeError(f"FLOAT argument '{arg_spec.name}' has no compute_fn defined")
            ffi_args.append(float(arg_spec.compute_fn(inputs, output, ref_model)))
        else:
            raise RuntimeError(f"Unsupported arg type {arg_spec.arg_type}")

    return ffi_args


def load_and_run_cuda_tvm(
    so_path: Path,
    signature: KernelSignature,
    # inputs: List[torch.Tensor]  输入张量列表
    # output: torch.Tensor        预分配的输出张量
    # workspace: Optional[List[torch.Tensor]] = None  可选工作区张量
    # ref_model: Any = None       可选参考模型（计算标量参数）
    # func_name: str = "launch_kernel"  共享库导出的函数名
) -> Module | None:
    # ) -> Tuple[Any, List[Any]]:
    """使用 tvm_ffi 加载并调用 CUDA kernel（ctypes 回退逻辑保留但未启用）。"""
    if not so_path.exists():
        raise FileNotFoundError(so_path)

    backend_error: Optional[str] = None
    assert TVM_FFI_AVAILABLE and tvm_ffi is not None, "tvm ffi is not available"
    # try:  # 保留回退模板
    #     # ffi_args = _build_signature_args_tvm_ffi(signature, inputs, output, workspace, ref_model)
    mod = tvm_ffi.load_module(str(so_path))
    return mod
    # except AttributeError:  # 函数缺失时的回退
    #     backend_error = f"Function '{func_name}' not found in {so_path} (tvm_ffi)"
    # except Exception as exc:  # noqa: BLE001  # 其他 tvm_ffi 异常
    #     backend_error = f"tvm_ffi failed ({exc})"
    # return None


def _bench_launch_tvm(
    lib: Any,
    args: List[Any],
    dev: torch.device,
    warm: int,
    rep: int,
    func_name: str = "launch_kernel",
) -> List[float]:
    """使用签名参数对 kernel 启动做基准测试。"""
    fn_name = getattr(lib, "_kernel_backend_func", func_name)
    fn = getattr(lib, fn_name, None)
    if fn is None and hasattr(lib, "get_function"):
        fn = lib.get_function(fn_name)
    if fn is None:
        raise RuntimeError(
            f"Function '{fn_name}' not found on backend {getattr(lib, '_kernel_backend', 'unknown')}"
        )

    if TORCH_DEVICE == "cpu":
        res: List[float] = []
        for _ in range(warm):
            fn(*args)
        for _ in range(rep):
            t0 = datetime.now()
            fn(*args)
            res.append((datetime.now() - t0).total_seconds() * 1_000)
        return res

    torch.cuda.synchronize(dev)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times: List[float] = []

    for _ in range(warm):
        fn(*args)
    torch.cuda.synchronize(dev)

    for _ in range(rep):
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    return times


# ====================== RNG & 确定性设置 ======================
def _seed_everything(seed: int | None, device_idx: int | None = None):
    """设置随机种子并（可选）启用确定性后端。"""
    import os
    import random
    import numpy as np
    import torch

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if device_idx is not None:
            torch.cuda.set_device(device_idx)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 更强可复现（如不需要可注释掉）
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 或 ":16:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 某些算子无确定性实现时仅告警不报错
        torch.use_deterministic_algorithms(True, warn_only=True)


# ====================== compare_and_bench（带通用对齐与种子） ======================
def compare_and_bench(
    ref_cu: Path,
    test_cu: Path,
    *,
    device_idx: int = 0,
    warmup: int = 5,
    repeat: int = 20,
    tol: float = 1e-4,
    log_dir: str | Path | None = "run/debug",
    seed: int = 100,  # 固定默认 seed；需要环境控制时可改成 None 并用 env 读取
) -> Dict[str, Any]:
    """
    基准对比两个 CUDA `.cu` kernel（基于 tvm_ffi）。

    reference：朴素基线 .cu
    candidate：优化版 .cu
    """
    import os
    import contextlib
    from datetime import datetime

    dev = torch.device(f"cuda:{device_idx}") if TORCH_DEVICE == "cuda" else torch.device("cpu")
    if TORCH_DEVICE == "cuda":
        torch.cuda.set_device(dev)

    if seed is None:
        env_seed = os.environ.get("KERNELBENCH_SEED")
        seed = int(env_seed) if env_seed is not None else None

    ref_cu = Path(ref_cu).resolve()
    test_cu = Path(test_cu).resolve()

    so_path: Path | None = None
    backend_used = "tvm_ffi"
    backend_error: str | None = None

    try:
        ctx = torch.cuda.device(dev) if TORCH_DEVICE == "cuda" else contextlib.nullcontext()
        with ctx, torch.no_grad():
            _seed_everything(seed, device_idx)
            if not TVM_FFI_AVAILABLE:
                raise RuntimeError("tvm_ffi is not available; cannot launch kernels.")

            # 构造输入：基于 kernel 名称的注册 generator
            kernel_name = ref_cu.stem
            signature, inputs, ref_out, workspace_ref, _ = build_registered_inputs(
                kernel_name,
                device=dev,
            )
            workspace_ref = workspace_ref or None

            # 构造基线与候选 runner
            ref_runner = KernelRunner(ref_cu, signature=signature)
            test_runner = KernelRunner(test_cu, signature=signature)

            so_path = test_runner.cu_path.with_suffix(".so").resolve()

            # 对齐 launch 参数
            ref_args = _build_signature_args_tvm_ffi(signature, inputs, ref_out, workspace_ref, None)
            test_out = torch.zeros_like(ref_out, device=dev)

            workspace_test = [torch.zeros_like(w) for w in (workspace_ref or [])] if workspace_ref else None
            test_args = _build_signature_args_tvm_ffi(
                signature, inputs, test_out, workspace_test, None
            )

            # 先运行基线
            ref_runner.run(*ref_args)
            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

            test_runner.run(*test_args)
            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

            ref_cpu = ref_out.cpu()
            test_cpu = test_out.cpu()

            if ref_out.dtype != test_out.dtype:
                test_out = test_out.to(ref_out.dtype)

            diff = (test_cpu - ref_cpu).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()

            if not torch.allclose(ref_cpu, test_cpu, atol=tol, rtol=tol):
                raise ValueError(
                    f"Outputs are not close (atol={tol}, rtol={tol}). "
                    f"max_abs_err={max_err:.3e}, mean_abs_err={mean_err:.3e}"
                )

            # 基准
            setattr(ref_runner.mod, "_kernel_backend", backend_used)
            setattr(test_runner.mod, "_kernel_backend", backend_used)
            setattr(ref_runner.mod, "_kernel_backend_func", ref_runner.function_name)
            setattr(test_runner.mod, "_kernel_backend_func", test_runner.function_name)

            ref_t = _bench_launch_tvm(
                ref_runner.mod, ref_args, dev, warmup, repeat, func_name=ref_runner.function_name
            )
            test_t = _bench_launch_tvm(
                test_runner.mod, test_args, dev, warmup, repeat, func_name=test_runner.function_name
            )

            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

    except Exception:
        import traceback as _tb
        raise RuntimeError(_tb.format_exc()) from None

    result: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reference_file": str(ref_cu),
        "candidate_file": str(test_cu),
        "candidate_so": str(so_path) if so_path is not None else "",
        "tolerance": tol,
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "launch_backend": backend_used,
        "launch_backend_error": backend_error,
        "ref_latency_ms": {
            "avg": sum(ref_t) / len(ref_t),
            "min": min(ref_t),
            "max": max(ref_t),
            "all": ref_t,
        },
        "test_latency_ms": {
            "avg": sum(test_t) / len(test_t),
            "min": min(test_t),
            "max": max(test_t),
            "all": test_t,
        },
        "num_runs": repeat,
        "model_init_args": [],
        "model_init_kwargs": {},
        "seed": seed,
        "align_stats": {},  # CUDA 路径无参数对齐步骤
    }
    return result


# =========================== 命令行包装 ==================================
def _cli():
    p = argparse.ArgumentParser(description="对比并基准两个模型文件")
    p.add_argument("reference", type=Path, help="参考 .cu 路径（基线）")
    p.add_argument("candidate", type=Path, help="候选 .cu 路径")
    p.add_argument("--device", type=int, default=0, help="CUDA 设备编号")
    p.add_argument("--warmup", type=int, default=5, help="热身迭代次数")
    p.add_argument("--repeat", type=int, default=20, help="基准迭代次数")
    p.add_argument("--tol", type=float, default=1e-4, help="最大绝对误差阈值")
    p.add_argument("--dump", type=Path, help="若提供则写出 JSON 结果")
    args = p.parse_args()

    res = compare_and_bench(
        args.reference,
        args.candidate,
        device_idx=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
        tol=args.tol,
    )
    print(json.dumps(res, indent=2))

    if args.dump:
        args.dump.write_text(json.dumps(res, indent=2))
        print(f"\nSaved ⇒ {args.dump}")


if __name__ == "__main__":
    _cli()
