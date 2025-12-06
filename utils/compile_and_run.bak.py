from __future__ import annotations
"""
compare_and_bench.py：基准对比 PyTorch 参考实现与独立 CUDA `.cu`。

流程：
1. 动态导入参考 PyTorch 模型（生成输入与基准输出）。
2. 用 nvcc 编译候选 `.cu` 为共享库。
3. 通过 TVM FFI 加载 `launch_kernel`，用 DLPack 传递张量，跑基准并对比输出。
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn

# TVM FFI 相关导入
try:
    import tvm
    from tvm import runtime as tvm_runtime
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    tvm = None
    tvm_runtime = None

# 可选：稳定 C ABI 的 tvm_ffi，存在时优先零拷贝 DLPack 互操作，不存在则回退 ctypes。
try:
    import tvm_ffi  # type: ignore
    TVM_FFI_AVAILABLE = True
except ImportError:
    TVM_FFI_AVAILABLE = False
    tvm_ffi = None

from utils.kernel_signature import (
    KernelSignature, KernelArg, ArgType, ArgRole,
    get_signature, matmul_signature, rmsnorm_signature, attention_signature
)

# ---------------------------------------------------------------------------

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------


class _LaunchWrapper:
    """轻量代理：暴露后端函数与元信息。"""

    def __init__(self, mod: Any, fn: Any, func_name: str, backend: str, err: Optional[str] = None):
        self._mod = mod
        self._fn = fn
        self._kernel_backend = backend
        self._kernel_backend_func = func_name
        self._kernel_backend_error = err

    def __getattr__(self, name: str):
        if name == self._kernel_backend_func:
            return self._fn
        return getattr(self._mod, name)


class CompilationError(RuntimeError):
    """动态导入或 nvcc 构建失败时抛出，首个参数是完整构建日志。"""


class AccuracyError(RuntimeError):
    """输出未满足精度阈值时抛出。"""


# =========================== dynamic import ===============================
def _capture_import(path: Path):
    """动态导入 path，并捕获全部构建日志。

    返回：(module, full_log:str)
    异常：
      FileNotFoundError  路径不存在
      CompilationError   导入期间的 Python/ninja/nvcc 错误（首参为拼接日志）
    """
    if not path.exists():
        raise FileNotFoundError(path)

    mod_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)                     # type: ignore[arg-type]
    sys.modules[mod_name] = module
    assert spec.loader is not None

    # ---- Python 层 stdout/stderr 重定向到 StringIO -----------------------
    py_buf = io.StringIO()

    # ---- OS 层 FD 1/2 (stdout/stderr) 重定向到临时文件 -----------------
    with tempfile.TemporaryFile(mode="w+") as fd_buf, \
         contextlib.redirect_stdout(py_buf), \
         contextlib.redirect_stderr(py_buf):

        # 保存当前 FD，方便恢复
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(fd_buf.fileno(), 1)     # redirect FD 1 → temp file
            os.dup2(fd_buf.fileno(), 2)     # redirect FD 2 → temp file

            # ------------ 实际导入（构建/编译） --------------------
            spec.loader.exec_module(module)                             # pyright: ignore[attr-defined]

            fd_buf.flush()
            fd_buf.seek(0)
            subproc_log = fd_buf.read()

        except Exception as exc:  # ← 构建/链接/导入失败
            # 合并 StringIO + 临时文件日志 + 异常字符串
            fd_buf.flush(); fd_buf.seek(0)
            subproc_log = fd_buf.read()
            full_log = "".join([py_buf.getvalue(), subproc_log, str(exc)]).strip()
            raise CompilationError(full_log) from None

        finally:
            # 确保恢复原始 FD
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

    # ---------------- 成功 --------------------------------------------
    return module, py_buf.getvalue() + subproc_log


# =========================== CUDA compile/load helpers ====================
def compile_cuda(cu_path: Path) -> Path:
    """用 nvcc 将 .cu 编译为共享库。"""
    if not cu_path.exists():
        raise FileNotFoundError(cu_path)

    so_path = cu_path.with_suffix(".so")
    tvm_inc = []
    tvm_lib = []
    tvm_rpath = []
    if TVM_AVAILABLE and tvm is not None:
        tvm_root = Path(tvm.__file__).resolve().parent
        inc1 = tvm_root / "include"
        inc2 = tvm_root / "3rdparty" / "dlpack" / "include"
        inc3 = tvm_root / "3rdparty" / "dmlc-core" / "include"
        tvm_inc = [f"-I{inc}" for inc in (inc1, inc2, inc3) if inc.exists()]
        if os.environ.get("KERNELBENCH_LINK_TVM_RUNTIME", "0") == "1":
            tvm_lib_dir = tvm_root
            tvm_lib = [f"-L{tvm_lib_dir}", "-ltvm_runtime"]
            tvm_rpath = ["-Xlinker", f"-rpath={tvm_lib_dir}"]

    cmd = [
        "nvcc",
        "-shared",
        "-O3",
        "-Xcompiler",
        "-fPIC",
        *tvm_inc,
        str(cu_path),
        *tvm_lib,
        *tvm_rpath,
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
    return so_path


def _tensor_ptr(t: torch.Tensor) -> ctypes.c_void_p:
    """Legacy ctypes pointer conversion (kept for fallback)."""
    return ctypes.c_void_p(int(t.data_ptr()))


def _infer_op_type(inputs: List[torch.Tensor]) -> str:
    """Heuristic op type (initial scope: matmul | rmsnorm | attention).

    DEPRECATED: Use KernelSignature instead.
    """
    if len(inputs) == 3:
        return "attention"
    if len(inputs) >= 2:
        return "matmul"
    return "rmsnorm"


def _build_launch_call(op_type: str,
                       inputs: List[torch.Tensor],
                       output: torch.Tensor,
                       ref_model=None) -> tuple[list, list]:
    if op_type == "attention":
        if len(inputs) < 3:
            raise RuntimeError("attention launch expects three input tensors (Q, K, V)")
        q, k, v = inputs[0], inputs[1], inputs[2]
        # 预期形状：[batch, heads, seq, dim]
        batch = int(q.shape[0]) if q.ndim >= 1 else 1
        heads = int(q.shape[1]) if q.ndim >= 2 else 1
        seq = int(q.shape[2]) if q.ndim >= 3 else 1
        dim = int(q.shape[3]) if q.ndim >= 4 else int(q.numel())
        scale = 1.0 / float(dim) ** 0.5

        argtypes = [
            ctypes.c_void_p,   # Q
            ctypes.c_void_p,   # K
            ctypes.c_void_p,   # V
            ctypes.c_void_p,   # O
            ctypes.c_longlong, # batch
            ctypes.c_longlong, # heads
            ctypes.c_longlong, # seq
            ctypes.c_longlong, # dim
            ctypes.c_float,    # scale
        ]
        args = [
            _tensor_ptr(q),
            _tensor_ptr(k),
            _tensor_ptr(v),
            _tensor_ptr(output),
            ctypes.c_longlong(batch),
            ctypes.c_longlong(heads),
            ctypes.c_longlong(seq),
            ctypes.c_longlong(dim),
            ctypes.c_float(scale),
        ]
        return argtypes, args

    if op_type == "matmul":
        if len(inputs) < 2:
            raise RuntimeError("matmul launch expects two input tensors")
        a, b = inputs[0], inputs[1]
        m = int(output.shape[-2]) if output.ndim >= 2 else int(a.shape[0])
        n = int(output.shape[-1]) if output.ndim >= 1 else int(b.shape[-1] if b.ndim else 1)
        if a.ndim >= 2:
            k = int(a.shape[-1])
        elif b.ndim >= 2:
            k = int(b.shape[-2])
        else:
            k = int(a.numel())

        argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # B
            ctypes.c_void_p,  # C
            ctypes.c_longlong,  # M
            ctypes.c_longlong,  # N
            ctypes.c_longlong,  # K
        ]
        args = [
            _tensor_ptr(a),
            _tensor_ptr(b),
            _tensor_ptr(output),
            ctypes.c_longlong(m),
            ctypes.c_longlong(n),
            ctypes.c_longlong(k),
        ]
        return argtypes, args

    # RMSNorm（单输入张量）
    x = inputs[0]
    batch = int(x.shape[0]) if x.ndim >= 1 else 1
    features = int(x.shape[1]) if x.ndim >= 2 else int(x.numel())
    inner = int(x.numel() // max(1, batch * features))
    eps = float(getattr(ref_model, "eps", 1e-5) if ref_model is not None else 1e-5)

    argtypes = [
        ctypes.c_void_p,     # X
        ctypes.c_void_p,     # Y
        ctypes.c_longlong,   # batch
        ctypes.c_longlong,   # features
        ctypes.c_longlong,   # inner (rest dims)
        ctypes.c_float,      # eps
    ]
    args = [
        _tensor_ptr(x),
        _tensor_ptr(output),
        ctypes.c_longlong(batch),
        ctypes.c_longlong(features),
        ctypes.c_longlong(inner),
        ctypes.c_float(eps),
    ]
    return argtypes, args


def load_and_run_cuda(so_path: Path,
                      op_type: str,
                      inputs: List[torch.Tensor],
                      output: torch.Tensor,
                      ref_model=None):
    """从 .so 中加载 launch_kernel，设置 argtypes，执行一次。

    已弃用：优先使用带 KernelSignature 的 load_and_run_cuda_tvm。
    """
    lib = ctypes.CDLL(str(so_path))
    if not hasattr(lib, "launch_kernel"):
        raise RuntimeError(f"'launch_kernel' not found in {so_path}")
    fn = lib.launch_kernel
    argtypes, args = _build_launch_call(op_type, inputs, output, ref_model=ref_model)
    fn.restype = None
    fn.argtypes = argtypes
    fn(*args)
    return fn, args


# =========================== TVM FFI Implementation ==========================

def _torch_to_tvm(tensor: torch.Tensor) -> Any:
    """Convert PyTorch tensor to TVM NDArray via DLPack.

    Note: This is kept for potential future use with TVM-compiled modules.
    For raw extern "C" CUDA kernels, we use ctypes directly.
    """
    if not TVM_AVAILABLE:
        raise RuntimeError("TVM is not available. Install with: pip install apache-tvm")
    return tvm.nd.from_dlpack(tensor)


def _torch_to_tvm_ffi(tensor: torch.Tensor) -> Any:
    """Convert PyTorch tensor to tvm_ffi NDArray using DLPack."""
    if not TVM_FFI_AVAILABLE or tvm_ffi is None:
        raise RuntimeError("tvm_ffi is not available.")
    if not hasattr(tvm_ffi, "from_dlpack"):
        raise RuntimeError("tvm_ffi.from_dlpack not found; please upgrade tvm_ffi.")
    from torch.utils import dlpack as torch_dlpack  # lazy import to avoid overhead
    return tvm_ffi.from_dlpack(torch_dlpack.to_dlpack(tensor))


def _build_signature_args(
    signature: KernelSignature,
    inputs: List[torch.Tensor],
    output: torch.Tensor,
    workspace: Optional[List[torch.Tensor]] = None,
    ref_model: Any = None,
) -> Tuple[List, List]:
    """Build ctypes argument list based on KernelSignature.

    Returns
    -------
    Tuple[argtypes, args]
        The ctypes argtypes and argument values
    """
    argtypes = []
    args = []
    input_idx = 0
    workspace_idx = 0

    for arg_spec in signature.args:
        if arg_spec.arg_type == ArgType.TENSOR:
            argtypes.append(ctypes.c_void_p)
            if arg_spec.role == ArgRole.INPUT:
                tensor = inputs[input_idx]
                input_idx += 1
                args.append(ctypes.c_void_p(tensor.data_ptr()))
            elif arg_spec.role == ArgRole.OUTPUT:
                args.append(ctypes.c_void_p(output.data_ptr()))
            elif arg_spec.role == ArgRole.WORKSPACE:
                if workspace is None or workspace_idx >= len(workspace):
                    raise RuntimeError(
                        f"Workspace tensor '{arg_spec.name}' required but not provided"
                    )
                args.append(ctypes.c_void_p(workspace[workspace_idx].data_ptr()))
                workspace_idx += 1

        elif arg_spec.arg_type == ArgType.INT:
            argtypes.append(ctypes.c_longlong)
            if arg_spec.compute_fn is not None:
                value = int(arg_spec.compute_fn(inputs, output, ref_model))
            else:
                raise RuntimeError(
                    f"INT argument '{arg_spec.name}' has no compute_fn defined"
                )
            args.append(ctypes.c_longlong(value))

        elif arg_spec.arg_type == ArgType.FLOAT:
            argtypes.append(ctypes.c_float)
            if arg_spec.compute_fn is not None:
                value = float(arg_spec.compute_fn(inputs, output, ref_model))
            else:
                raise RuntimeError(
                    f"FLOAT argument '{arg_spec.name}' has no compute_fn defined"
                )
            args.append(ctypes.c_float(value))

    return argtypes, args


def _build_signature_args_tvm_ffi(
    signature: KernelSignature,
    inputs: List[torch.Tensor],
    output: torch.Tensor,
    workspace: Optional[List[torch.Tensor]] = None,
    ref_model: Any = None,
) -> List:
    """Build argument list for tvm_ffi using KernelSignature."""
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


def _build_signature_args_tvm_runtime(
    signature: KernelSignature,
    inputs: List[torch.Tensor],
    output: torch.Tensor,
    workspace: Optional[List[torch.Tensor]] = None,
    ref_model: Any = None,
) -> List:
    """Build argument list for tvm.runtime PackedFunc using KernelSignature."""
    if not TVM_AVAILABLE or tvm is None:
        raise RuntimeError("TVM runtime is not available.")

    rt_args: List[Any] = []
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
            rt_args.append(_torch_to_tvm(tensor))
        elif arg_spec.arg_type == ArgType.INT:
            if arg_spec.compute_fn is None:
                raise RuntimeError(f"INT argument '{arg_spec.name}' has no compute_fn defined")
            rt_args.append(int(arg_spec.compute_fn(inputs, output, ref_model)))
        elif arg_spec.arg_type == ArgType.FLOAT:
            if arg_spec.compute_fn is None:
                raise RuntimeError(f"FLOAT argument '{arg_spec.name}' has no compute_fn defined")
            rt_args.append(float(arg_spec.compute_fn(inputs, output, ref_model)))
        else:
            raise RuntimeError(f"Unsupported arg type {arg_spec.arg_type}")
    return rt_args


def load_module_tvm(so_path: Path) -> Any:
    """Load a compiled .so module via TVM runtime.

    Note: TVM load_module works best with TVM-compiled modules.
    For raw extern "C" CUDA kernels, TVM's PackedFunc cannot directly
    call them - use load_and_run_cuda_tvm which uses ctypes internally.
    """
    if not TVM_AVAILABLE:
        raise RuntimeError("TVM is not available. Install with: pip install apache-tvm")
    if not so_path.exists():
        raise FileNotFoundError(so_path)
    return tvm_runtime.load_module(str(so_path))


def load_and_run_cuda_tvm(
    so_path: Path,
    signature: KernelSignature,
    inputs: List[torch.Tensor],
    output: torch.Tensor,
    workspace: Optional[List[torch.Tensor]] = None,
    ref_model: Any = None,
    func_name: str = "launch_kernel",
    use_tvm_ffi: bool | None = None,
    use_tvm_runtime: bool | None = None,
) -> Tuple[Any, List[Any]]:
    """Load and run CUDA kernel using KernelSignature.

    The function prefers TVM-FFI (stable C ABI) if available and requested,
    otherwise tries TVM runtime PackedFunc, then falls back to ctypes.

    Parameters
    ----------
    so_path : Path
        Path to compiled .so file
    signature : KernelSignature
        Kernel argument signature
    inputs : List[torch.Tensor]
        Input tensors
    output : torch.Tensor
        Output tensor (pre-allocated)
    workspace : Optional[List[torch.Tensor]]
        Workspace tensors if required by signature
    ref_model : Any
        Reference model for computing scalar arguments
    func_name : str
        Name of the function to call (default: "launch_kernel")
    use_tvm_ffi : bool | None
        If True, force tvm_ffi; if False, force ctypes; if None (default),
        prefer tvm_ffi when available.
    use_tvm_runtime : bool | None
        If True, force tvm.runtime; if False, skip tvm.runtime; if None,
        try tvm.runtime after tvm_ffi when available.

    Returns
    -------
    Tuple[lib, args]
        The ctypes library and the argument list used
    """
    if not so_path.exists():
        raise FileNotFoundError(so_path)

    backend_error: Optional[str] = None
    backend_used = "ctypes"
    prefer_tvm_ffi = TVM_FFI_AVAILABLE if use_tvm_ffi is None else use_tvm_ffi
    prefer_tvm_runtime = TVM_AVAILABLE if use_tvm_runtime is None else use_tvm_runtime

    # Attempt TVM-FFI path first if requested and available.
    if prefer_tvm_ffi and TVM_FFI_AVAILABLE:
        try:
            ffi_args = _build_signature_args_tvm_ffi(signature, inputs, output, workspace, ref_model)
            mod = tvm_ffi.load_module(str(so_path))
            fn = getattr(mod, f"{func_name}_tvm", None)
            if fn is None:
                fn = getattr(mod, func_name, None)
            if fn is None:
                raise RuntimeError(f"Function '{func_name}' not found in {so_path} (tvm_ffi)")
            fn(*ffi_args)
            wrapper = _LaunchWrapper(mod, fn, getattr(fn, "__name__", func_name), "tvm_ffi")
            return wrapper, ffi_args
        except Exception as exc:  # noqa: BLE001
            backend_error = f"tvm_ffi failed ({exc})"
            # fallback to ctypes below

    # Next attempt TVM runtime PackedFunc path.
    if prefer_tvm_runtime and TVM_AVAILABLE and tvm is not None:
        try:
            rt_args = _build_signature_args_tvm_runtime(signature, inputs, output, workspace, ref_model)
            mod = tvm_runtime.load_module(str(so_path))
            fn = mod.get_function(f"{func_name}_tvm")
            used_name = f"{func_name}_tvm"
            if fn is None:
                fn = mod.get_function(func_name)
                used_name = func_name
            if fn is None:
                raise RuntimeError(f"Function '{func_name}' not found in {so_path} (tvm.runtime)")
            fn(*rt_args)
            wrapper = _LaunchWrapper(mod, fn, used_name, "tvm_runtime")
            return wrapper, rt_args
        except Exception as exc:  # noqa: BLE001
            backend_error = f"tvm.runtime failed ({exc})"

    lib = ctypes.CDLL(str(so_path))
    if not hasattr(lib, func_name):
        raise RuntimeError(f"Function '{func_name}' not found in {so_path}")

    fn = getattr(lib, func_name)
    argtypes, args = _build_signature_args(signature, inputs, output, workspace, ref_model)
    fn.restype = None
    fn.argtypes = argtypes
    fn(*args)
    setattr(lib, "_kernel_backend", backend_used)
    if backend_error:
        setattr(lib, "_kernel_backend_error", backend_error)

    return lib, args


def _bench_launch_tvm(
    lib: Any,
    args: List[Any],
    dev: torch.device,
    warm: int,
    rep: int,
    func_name: str = "launch_kernel",
) -> List[float]:
    """Benchmark kernel launch using signature-based args."""
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


def _bench_launch(fn, kargs, dev: torch.device, warm: int, rep: int) -> List[float]:
    """Legacy benchmark function for ctypes-based kernel launch.

    DEPRECATED: Use _bench_launch_tvm instead.
    """
    if TORCH_DEVICE == "cpu":
        res: List[float] = []
        for _ in range(warm):
            fn(*kargs)
        for _ in range(rep):
            t0 = datetime.now()
            fn(*kargs)
            res.append((datetime.now() - t0).total_seconds() * 1_000)
        return res

    torch.cuda.synchronize(dev)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times: List[float] = []

    for _ in range(warm):
        fn(*kargs)
    torch.cuda.synchronize(dev)

    for _ in range(rep):
        start.record()
        fn(*kargs)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


# =========================== timing helpers ===============================
def _run_once(model: torch.nn.Module,
              inp: List[torch.Tensor],
              dev: torch.device) -> Tuple[torch.Tensor, float]:
    model.to(dev).eval()
    inp = [x.to(dev) for x in inp]

    if TORCH_DEVICE == "cpu":
        t0 = datetime.now()
        out = model(*inp)
        ms = (datetime.now() - t0).total_seconds() * 1_000
        return out, ms

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(dev)
    start.record()
    out = model(*inp)
    end.record()
    end.synchronize()
    return out, start.elapsed_time(end)


def _bench(model: torch.nn.Module,
           inp: List[torch.Tensor],
           dev: torch.device,
           warm: int,
           rep: int) -> List[float]:
    model.to(dev).eval()
    inp = [x.to(dev) for x in inp]

    for _ in range(warm):
        model(*inp)

    if TORCH_DEVICE == "cpu":
        res = []
        for _ in range(rep):
            t0 = datetime.now()
            model(*inp)
            res.append((datetime.now() - t0).total_seconds() * 1_000)
        return res

    torch.cuda.synchronize(dev)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    times: List[float] = []
    for _ in range(rep):
        s.record()
        model(*inp)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return times


# ====================== RNG & 确定性设置 ======================
def _seed_everything(seed: int | None, device_idx: int | None = None):
    """设置随机种子并（可选）启用确定性后端。"""
    import os, random
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


# ====================== 参数对齐（通用 + 类名/导出名专用） ======================
from collections import defaultdict

def _named_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    named: dict[str, torch.Tensor] = {}
    for k, p in model.named_parameters(recurse=True):
        named[f"param::{k}"] = p
    for k, b in model.named_buffers(recurse=True):
        named[f"buffer::{k}"] = b
    return named

@torch.no_grad()
def _safe_copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.shape != src.shape:
        return False
    dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
    return True

@torch.no_grad()
def _try_map_shape_and_copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """
    形状映射覆盖：
      - depthwise 2D:   (C,1,Kh,1)<->(C,Kh), (C,1,Kh,Kw)<->(C,Kh,Kw)
      - PW/Linear:      (Out,In,1,1)<->(Out,In)
      - Conv/ConvT 3D:  (Out,In,kD,kH,kW) <-> (In,Out,kD,kH,kW) （首两维交换）
      - depthwise 3D:   (C,1,kD,kH,kW) <-> (C,kD,kH,kW)
    """
    s = tuple(src.shape)
    d = tuple(dst.shape)

    # --- depthwise 2D: (C,1,Kh,1) <-> (C,Kh)
    if len(s) == 4 and s[1] == 1 and s[3] == 1 and len(d) == 2 and s[0] == d[0] and s[2] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True
    if len(s) == 2 and len(d) == 4 and d[1] == 1 and d[3] == 1 and s[0] == d[0] and s[1] == d[2]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True

    # --- depthwise 2D: (C,1,Kh,Kw) -> (C,Kh,Kw) 及其反向
    if len(s) == 4 and s[1] == 1 and len(d) == 3 and s[0] == d[0] and s[2] == d[1] and s[3] == d[2]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).squeeze(1).contiguous())
        return True
    if len(s) == 3 and len(d) == 4 and d[1] == 1 and s[0] == d[0] and s[1] == d[2] and s[2] == d[3]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).unsqueeze(1).contiguous())
        return True

    # --- PW/Linear: (Out,In,1,1) <-> (Out,In)
    if len(s) == 4 and s[2] == 1 and s[3] == 1 and len(d) == 2 and s[0] == d[0] and s[1] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True
    if len(s) == 2 and len(d) == 4 and d[2] == 1 and d[3] == 1 and s[0] == d[0] and s[1] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True

    # --- Conv/ConvTranspose 3D: 5D 权重首两维交换
    #     (Out, In, kD, kH, kW)  <->  (In, Out, kD, kH, kW)
    if len(s) == 5 and len(d) == 5 and s[0] == d[1] and s[1] == d[0] and s[2:] == d[2:]:
        dst.copy_(src.permute(1, 0, 2, 3, 4).contiguous().to(dtype=dst.dtype, device=dst.device))
        return True

    # --- depthwise 3D: (C,1,kD,kH,kW) -> (C,kD,kH,kW) 及其反向
    if len(s) == 5 and s[1] == 1 and len(d) == 4 and s[0] == d[0] and s[2:] == d[1:]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).squeeze(1).contiguous())
        return True
    if len(s) == 4 and len(d) == 5 and d[1] == 1 and s[0] == d[0] and s[1:] == d[2:]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).unsqueeze(1).contiguous())
        return True

    return False

@torch.no_grad()
def align_params_generic(ref_model: nn.Module, test_model: nn.Module) -> dict[str, int]:
    ref_named = _named_tensors(ref_model)
    test_named = _named_tensors(test_model)

    copied_same, unique_shape_copied, mapped, skipped = 0, 0, 0, 0
    aligned_test: set[str] = set()

    # 1) 同名同形状
    for name, t_dst in test_named.items():
        t_src = ref_named.get(name, None)
        if t_src is not None and _safe_copy_(t_dst, t_src):
            copied_same += 1
            aligned_test.add(name)

    # 2) 唯一形状匹配
    shape2ref: dict[tuple, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    shape2test: dict[tuple, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for n, t in ref_named.items():
        shape2ref[tuple(t.shape)].append((n, t))
    for n, t in test_named.items():
        if n in aligned_test: 
            continue
        shape2test[tuple(t.shape)].append((n, t))

    for shp, items in shape2test.items():
        if len(items) == 1 and len(shape2ref.get(shp, [])) == 1:
            tname, t_dst = items[0]
            _, t_src = shape2ref[shp][0]
            if _safe_copy_(t_dst, t_src):
                unique_shape_copied += 1
                aligned_test.add(tname)

    # 3) 形状映射
    for name, t_dst in test_named.items():
        if name in aligned_test:
            continue
        ok = False
        for _, t_src in ref_named.items():
            if _try_map_shape_and_copy_(t_dst, t_src):
                mapped += 1
                aligned_test.add(name)
                ok = True
                break
        if not ok:
            skipped += 1

    return {
        "copied_same_shape": copied_same,
        "unique_shape_copied": unique_shape_copied,
        "mapped_shape": mapped,
        "skipped": skipped,
    }

# ——（可选）按类名/导出名注册“专用对齐器”：Model → ModelNew ——
_PAIR_ALIGNERS: dict[tuple[str, str], callable] = {}

def register_pair_aligner(ref_key: str, test_key: str):
    def deco(fn):
        _PAIR_ALIGNERS[(ref_key, test_key)] = fn
        return fn
    return deco

@register_pair_aligner("Model", "ModelNew")
@torch.no_grad()
def _align_Model_to_ModelNew(ref_model: nn.Module, test_model: nn.Module) -> dict[str, int]:
    ref_named = _named_tensors(ref_model)
    test_named = _named_tensors(test_model)

    def pick(named: dict[str, torch.Tensor], dims: int):
        cand = [(n, t) for n, t in named.items()
                if n.startswith("param::") and "weight" in n and t.ndim == dims]
        if not cand:
            cand = [(n, t) for n, t in named.items()
                    if n.startswith("param::") and t.ndim == dims]
        return cand

    # ---- 2D: Conv / ConvTranspose（4D 同形状 或 首两维交换）----
    r4 = pick(ref_named, 4); t4 = pick(test_named, 4)
    if len(r4) == 1 and len(t4) == 1:
        w_ref, w_tst = r4[0][1], t4[0][1]
        if tuple(w_ref.shape) == tuple(w_tst.shape):
            w_tst.copy_(w_ref.to(dtype=w_tst.dtype, device=w_tst.device))
            pass_bias = True
        elif (w_ref.shape[0] == w_tst.shape[1] and w_ref.shape[1] == w_tst.shape[0]
              and w_ref.shape[2:] == w_tst.shape[2:]):
            w_tst.copy_(w_ref.permute(1, 0, 2, 3).contiguous().to(dtype=w_tst.dtype, device=w_tst.device))
            pass_bias = True
        else:
            pass_bias = False

        if pass_bias:
            rb = [(n,t) for n,t in ref_named.items() if "bias" in n and n.startswith("param::") and t.ndim==1]
            tb = [(n,t) for n,t in test_named.items() if "bias" in n and n.startswith("param::") and t.ndim==1]
            if len(rb)==1 and len(tb)==1 and tuple(rb[0][1].shape)==tuple(tb[0][1].shape):
                tb[0][1].copy_(rb[0][1].to(dtype=tb[0][1].dtype, device=tb[0][1].device))
            return {"pair_aligner": 1, "copied_same_shape": int(tuple(w_ref.shape)==tuple(w_tst.shape)),
                    "mapped_shape": int(tuple(w_ref.shape)!=tuple(w_tst.shape)), "skipped": 0}

    # ---- 3D: Conv3d / ConvTranspose3d（5D 同形状 或 首两维交换）----
    r5 = pick(ref_named, 5); t5 = pick(test_named, 5)
    if len(r5) == 1 and len(t5) == 1:
        w_ref, w_tst = r5[0][1], t5[0][1]
        if tuple(w_ref.shape) == tuple(w_tst.shape):
            w_tst.copy_(w_ref.to(dtype=w_tst.dtype, device=w_tst.device))
            return {"pair_aligner": 1, "copied_same_shape": 1, "mapped_shape": 0, "skipped": 0}
        if (w_ref.shape[0] == w_tst.shape[1] and w_ref.shape[1] == w_tst.shape[0]
                and w_ref.shape[2:] == w_tst.shape[2:]):
            w_tst.copy_(w_ref.permute(1, 0, 2, 3, 4).contiguous().to(dtype=w_tst.dtype, device=w_tst.device))
            return {"pair_aligner": 1, "copied_same_shape": 0, "mapped_shape": 1, "skipped": 0}

    # ---- depthwise-3D: (C,1,kD,kH,kW) ↔ (C,kD,kH,kW) ----
    if len(r5) == 1:
        w_ref = r5[0][1]
        t4 = pick(test_named, 4)
        if len(t4) == 1:
            w_tst = t4[0][1]
            if w_ref.size(1) == 1 and tuple(w_tst.shape) == (w_ref.size(0), w_ref.size(2), w_ref.size(3), w_ref.size(4)):
                w_tst.copy_(w_ref.to(dtype=w_tst.dtype, device=w_tst.device).squeeze(1).contiguous())
                return {"pair_aligner": 1, "copied_same_shape": 0, "mapped_shape": 1, "skipped": 0}

    # 其余回退通用
    stats = align_params_generic(ref_model, test_model)
    stats["pair_aligner"] = 0
    return stats

@torch.no_grad()
def try_align_params(ref_model: nn.Module, test_model: nn.Module,
                     ref_mod=None, test_mod=None) -> dict[str, int]:
    """
    优先级：
      0) 导出名派发（_export_symbol），如 ("Model","ModelNew")
      0b) 实例类名派发
      1) 任务自定义 map_ref_to_test_params / align_params
      2) 通用自动对齐
    """
    # 0) 导出名（若 compare_and_bench 已设置）
    key_export = (getattr(ref_model, "_export_symbol", None),
                  getattr(test_model, "_export_symbol", None))
    if key_export in _PAIR_ALIGNERS:
        stats = _PAIR_ALIGNERS[key_export](ref_model, test_model)
        stats["pair_key"] = f"{key_export[0]}->{key_export[1]}"
        return stats

    # 0b) 实例类名
    key_class = (ref_model.__class__.__name__, test_model.__class__.__name__)
    if key_class in _PAIR_ALIGNERS:
        stats = _PAIR_ALIGNERS[key_class](ref_model, test_model)
        stats["pair_key"] = f"{key_class[0]}->{key_class[1]}"
        return stats

    # 1) 任务自定义
    for mod in (test_mod, ref_mod):
        if mod is None:
            continue
        for fn_name in ("map_ref_to_test_params", "align_params"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                fn(ref_model, test_model)
                return {"pair_aligner": 0, "copied_same_shape": -1, "mapped_shape": -1,
                        "skipped": -1, "pair_key": "custom_fn"}

    # 2) 通用
    stats = align_params_generic(ref_model, test_model)
    stats["pair_aligner"] = 0
    stats["pair_key"] = "generic"
    return stats



# ====================== compare_and_bench（带通用对齐与种子） ======================
def compare_and_bench(
    ref_py: Path,
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
    Benchmark a CUDA `.cu` kernel (compiled + ctypes launch) against the
    reference PyTorch model.
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

    # ------------ 动态导入（仅 reference） ------------
    ref_mod, _ = _capture_import(ref_py)

    RefModel   = getattr(ref_mod,  "Model",       None)
    get_inputs = getattr(ref_mod,  "get_inputs",  None)

    if None in (RefModel, get_inputs):
        raise RuntimeError(f"Reference '{ref_py}' 必须定义 Model 与 get_inputs()。")

    # ------------ 初始化参数 ------------
    init_args: List[Any] = []
    init_kwargs: Dict[str, Any] = {}
    get_init_inputs_ref = getattr(ref_mod, "get_init_inputs", None)

    if callable(get_init_inputs_ref):
        init_obj = get_init_inputs_ref()
        if isinstance(init_obj, dict):
            init_kwargs = dict(init_obj)
        elif isinstance(init_obj, (list, tuple)):
            init_args = list(init_obj)
        elif init_obj is not None:
            raise TypeError("get_init_inputs() 必须返回 list/tuple（作为 *args）或 dict（作为 **kwargs）。")

    def _first_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            for t in x:
                if isinstance(t, torch.Tensor):
                    return t
        raise TypeError("Model forward 未返回 Tensor（或序列中的 Tensor）。")

    try:
        ctx = torch.cuda.device(dev) if TORCH_DEVICE == "cuda" else contextlib.nullcontext()
        with ctx, torch.no_grad():
            _seed_everything(seed, device_idx)
            inp = get_inputs()
            if not isinstance(inp, (list, tuple)):
                inp = [inp]

            # 初始化 reference
            _seed_everything(seed, device_idx)
            ref_model = RefModel(*init_args, **init_kwargs)
            ref_model.eval()
            ref_model.to(dev)

            # 参考前向
            inputs_dev = [x.to(dev).contiguous() for x in inp]
            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)
            ref_out = ref_model(*inputs_dev)
            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

            ref_out = _first_tensor(ref_out).contiguous()

            # 编译 + 运行 CUDA 候选
            so_path = compile_cuda(test_cu)
            test_out = torch.empty_like(ref_out, device=dev)
            op_type = _infer_op_type(inputs_dev)
            signature = get_signature(op_type)
            lib, launch_args = load_and_run_cuda_tvm(
                so_path,
                signature,
                inputs_dev,
                test_out,
                ref_model=ref_model,
            )
            backend_used = getattr(lib, "_kernel_backend", "ctypes")
            backend_error = getattr(lib, "_kernel_backend_error", None)
            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

            # 误差检查
            if ref_out.dtype != test_out.dtype:
                test_out = test_out.to(ref_out.dtype)

            ref_cpu = ref_out.cpu()
            test_cpu = test_out.cpu()
            diff = (test_cpu - ref_cpu).abs()
            max_err  = diff.max().item()
            mean_err = diff.mean().item()

            if not torch.allclose(ref_cpu, test_cpu, atol=tol, rtol=tol):
                raise ValueError(
                    f"Outputs are not close (atol={tol}, rtol={tol}). "
                    f"max_abs_err={max_err:.3e}, mean_abs_err={mean_err:.3e}"
                )

            # 基准
            ref_t  = _bench(ref_model,  inp, dev, warmup, repeat)
            test_t = _bench_launch_tvm(lib, launch_args, dev, warmup, repeat)

            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

    except Exception:
        import traceback as _tb
        raise RuntimeError(_tb.format_exc()) from None

    result: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reference_file": str(ref_py),
        "candidate_file": str(test_cu),
        "candidate_so": str(so_path),
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
        "model_init_args": init_args,
        "model_init_kwargs": init_kwargs,
        "seed": seed,
        "align_stats": {},  # no param alignment in CUDA path
    }
    return result





# =========================== CLI wrapper ==================================
def _cli():
    p = argparse.ArgumentParser(description="Compare & bench two model files.")
    p.add_argument("reference", type=Path, help="Path to reference .py")
    p.add_argument("candidate", type=Path, help="Path to candidate .cu")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=20, help="Benchmark runs")
    p.add_argument("--tol", type=float, default=1e-4, help="Max abs error tolerance")
    p.add_argument("--dump", type=Path, help="If set, write JSON results here")
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
