from __future__ import annotations
"""
Key features
------------
* Dynamically imports two PyTorch models (reference & candidate) and **captures
  every byte** printed by Python *and* child processes (ninja / nvcc).
  - On any *build* failure, raises `CompilationError(full_log)`.
* On **runtime failure** (forward, benchmark, accuracy), re-raises
  `RuntimeError(traceback.format_exc())` so callers get the *entire*
  traceback – not just `str(exc)`.
* Benchmarks on CUDA (default) or CPU (`--cpu`).
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Any, Dict

import torch

# ---------------------------------------------------------------------------

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------


class CompilationError(RuntimeError):
    """Raised when dynamic import / nvcc build fails.

    The *first* argument is the full build log (Python + ninja/nvcc).
    """


# =========================== dynamic import ===============================
def _capture_import(path: Path):
    """Import *path* dynamically and capture **all** build logs.

    Returns
    -------
    (module, full_log : str)

    Raises
    ------
    FileNotFoundError
        *path* does not exist.
    CompilationError
        Any Python / ninja / nvcc error during import.  The exception's first
        argument is the concatenated log.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    mod_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)                     # type: ignore[arg-type]
    sys.modules[mod_name] = module
    assert spec.loader is not None

    # ---- Redirect Python-level stdout/stderr to StringIO -----------------
    py_buf = io.StringIO()

    # ---- Redirect OS-level FD 1/2 (stdout/stderr) to a temp file --------
    with tempfile.TemporaryFile(mode="w+") as fd_buf, \
         contextlib.redirect_stdout(py_buf), \
         contextlib.redirect_stderr(py_buf):

        # Save current FDs so we can restore later
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(fd_buf.fileno(), 1)     # redirect FD 1 → temp file
            os.dup2(fd_buf.fileno(), 2)     # redirect FD 2 → temp file

            # ------------ REAL IMPORT (build/compile) --------------------
            spec.loader.exec_module(module)                             # pyright: ignore[attr-defined]

            fd_buf.flush()
            fd_buf.seek(0)
            subproc_log = fd_buf.read()

        except Exception as exc:  # ← build / link / import failed
            # Combine StringIO + temp-file logs + Exception str
            fd_buf.flush(); fd_buf.seek(0)
            subproc_log = fd_buf.read()
            full_log = "".join([py_buf.getvalue(), subproc_log, str(exc)]).strip()
            raise CompilationError(full_log) from None

        finally:
            # Always restore original FDs
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

    # ---------------- SUCCESS --------------------------------------------
    return module, py_buf.getvalue() + subproc_log


# =========================== RNG & Determinism Settings ===================
def _seed_everything(seed: int | None, device_idx: int | None = None):
    """Set random seeds and (optionally) enable deterministic backends.

    Notes
    -----
    * Also configures CUDA/CuDNN for stronger reproducibility (can be disabled
      if not required).
    """
    import os, random
    import numpy as np
    import torch as _torch

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        if device_idx is not None:
            _torch.cuda.set_device(device_idx)
        _torch.cuda.manual_seed(seed)
        _torch.cuda.manual_seed_all(seed)

        # Stronger reproducibility (comment out if not needed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # or ":16:8"
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
        # Warn (not error) when an op has no deterministic implementation
        _torch.use_deterministic_algorithms(True, warn_only=True)


# = Parameter alignment (generic + class/export-name specific) =============
import torch
import torch.nn as nn
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
    Shape mapping coverage:
      - Depthwise 2D:   (C,1,Kh,1)<->(C,Kh), (C,1,Kh,Kw)<->(C,Kh,Kw)
      - PW/Linear:      (Out,In,1,1)<->(Out,In)
      - Conv/ConvT 3D:  (Out,In,kD,kH,kW) <-> (In,Out,kD,kH,kW) (swap first two dims)
      - Depthwise 3D:   (C,1,kD,kH,kW) <-> (C,kD,kH,kW)
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

    # --- depthwise 2D: (C,1,Kh,Kw) -> (C,Kh,Kw) and reverse
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

    # --- Conv/ConvTranspose 3D: swap first two dims for 5D weights
    #     (Out, In, kD, kH, kW)  <->  (In, Out, kD, kH, kW)
    if len(s) == 5 and len(d) == 5 and s[0] == d[1] and s[1] == d[0] and s[2:] == d[2:]:
        dst.copy_(src.permute(1, 0, 2, 3, 4).contiguous().to(dtype=dst.dtype, device=dst.device))
        return True

    # --- depthwise 3D: (C,1,kD,kH,kW) -> (C,kD,kH,kW) and reverse
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

    # 1) Same name & same shape
    for name, t_dst in test_named.items():
        t_src = ref_named.get(name, None)
        if t_src is not None and _safe_copy_(t_dst, t_src):
            copied_same += 1
            aligned_test.add(name)

    # 2) Unique shape match
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

    # 3) Shape mapping
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

# — (Optional) Register pair-specific aligners by class/export names: Model → ModelNew —
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

    # ---- 2D: Conv / ConvTranspose (4D: same shape or swap first two dims) ----
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

    # ---- 3D: Conv3d / ConvTranspose3d (5D: same shape or swap first two dims) ----
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

    # Otherwise, fall back to the generic path
    stats = align_params_generic(ref_model, test_model)
    stats["pair_aligner"] = 0
    return stats

@torch.no_grad()
def try_align_params(ref_model: nn.Module, test_model: nn.Module,
                     ref_mod=None, test_mod=None) -> dict[str, int]:
    """
    Priority:
      0) Dispatch by exported symbols (`_export_symbol`), e.g., ("Model", "ModelNew")
      0b) Dispatch by instance class names
      1) Task-defined `map_ref_to_test_params` / `align_params`
      2) Generic automatic alignment
    """
    # 0) Exported symbol keys (if compare_and_bench set them)
    key_export = (getattr(ref_model, "_export_symbol", None),
                  getattr(test_model, "_export_symbol", None))
    if key_export in _PAIR_ALIGNERS:
        stats = _PAIR_ALIGNERS[key_export](ref_model, test_model)
        stats["pair_key"] = f"{key_export[0]}->{key_export[1]}"
        return stats

    # 0b) Instance class names
    key_class = (ref_model.__class__.__name__, test_model.__class__.__name__)
    if key_class in _PAIR_ALIGNERS:
        stats = _PAIR_ALIGNERS[key_class](ref_model, test_model)
        stats["pair_key"] = f"{key_class[0]}->{key_class[1]}"
        return stats

    # 1) Task-defined hooks
    for mod in (test_mod, ref_mod):
        if mod is None:
            continue
        for fn_name in ("map_ref_to_test_params", "align_params"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                fn(ref_model, test_model)
                return {"pair_aligner": 0, "copied_same_shape": -1, "mapped_shape": -1,
                        "skipped": -1, "pair_key": "custom_fn"}

    # 2) Generic path
    stats = align_params_generic(ref_model, test_model)
    stats["pair_aligner"] = 0
    stats["pair_key"] = "generic"
    return stats


# ===== compare_and_bench (with generic alignment & seeding) ===============
def compare_and_bench(
    ref_py: Path,
    test_py: Path,
    *,
    device_idx: int = 0,
    warmup: int = 5,
    repeat: int = 20,
    tol: float = 1e-4,
    log_dir: str | Path | None = "run/debug",
    seed: int = 100,  # Fixed default seed; set to None and use env to control if needed
) -> Dict[str, Any]:
    """
    Benchmark *test_py* against *ref_py*.

    Only reads `get_init_inputs()` from the **reference** script and uses the
    same initialization args/kwargs for both ref/test. Also: fix randomness &
    align parameters (supports Model → ModelNew pair-specific alignment & generic alignment).
    """
    import os
    import contextlib
    from datetime import datetime

    # ------------ Device setup -------------------------------------------
    dev = torch.device(f"cuda:{device_idx}") if TORCH_DEVICE == "cuda" else torch.device("cpu")
    if TORCH_DEVICE == "cuda":
        torch.cuda.set_device(dev)

    # Optionally control seed via environment variable
    if seed is None:
        env_seed = os.environ.get("KERNELBENCH_SEED")
        seed = int(env_seed) if env_seed is not None else None

    # ------------ Dynamic import ----------------------------------------
    ref_mod, _ = _capture_import(ref_py)
    test_mod, _ = _capture_import(test_py)

    RefModel   = getattr(ref_mod,  "Model",       None)
    get_inputs = getattr(ref_mod,  "get_inputs",  None)
    ModelNew   = getattr(test_mod, "ModelNew",    None)

    if None in (RefModel, get_inputs):
        raise RuntimeError(f"Reference '{ref_py}' must define Model and get_inputs().")
    if ModelNew is None:
        raise RuntimeError(f"Candidate '{test_py}' must define class ModelNew.")

    # ------------ Get init args only from reference ----------------------
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
            raise TypeError("get_init_inputs() must return list/tuple (as *args) or dict (as **kwargs).")

    # ------------ Run & benchmark ---------------------------------------
    def _first_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            for t in x:
                if isinstance(t, torch.Tensor):
                    return t
        raise TypeError("Model forward did not return a Tensor (or a sequence containing a Tensor).")

    try:
        ctx = torch.cuda.device(dev) if TORCH_DEVICE == "cuda" else contextlib.nullcontext()
        with ctx:
            # Fix input randomness
            _seed_everything(seed, device_idx)
            inp = get_inputs()
            if not isinstance(inp, (list, tuple)):
                inp = [inp]

            # Fix parameter initialization: set seed before constructing each side
            _seed_everything(seed, device_idx)
            ref_model  = RefModel(*init_args, **init_kwargs)

            _seed_everything(seed, device_idx)
            test_model = ModelNew(*init_args, **init_kwargs)

            # Parameter alignment (prefer Model→ModelNew pair-specific, then task custom, finally generic)
            align_stats = try_align_params(ref_model, test_model, ref_mod=ref_mod, test_mod=test_mod)

            # Forward pass (sync to surface errors immediately)
            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)
            ref_out,  _ = _run_once(ref_model,  inp, dev)
            test_out, _ = _run_once(test_model, inp, dev)
            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

            # Normalize to Tensor and ensure contiguous
            ref_out  = _first_tensor(ref_out).contiguous()
            test_out = _first_tensor(test_out).contiguous()
            if ref_out.dtype != test_out.dtype:
                test_out = test_out.to(ref_out.dtype)

            # Error & allclose
            diff = (test_out - ref_out).abs()
            max_err  = diff.max().item()
            mean_err = diff.mean().item()

            if not torch.allclose(ref_out, test_out, atol=tol, rtol=tol):
                raise ValueError(
                    f"Outputs are not close (atol={tol}, rtol={tol}). "
                    f"max_abs_err={max_err:.3e}, mean_abs_err={mean_err:.3e}"
                )

            # Timing
            ref_t  = _bench(ref_model,  inp, dev, warmup, repeat)
            test_t = _bench(test_model, inp, dev, warmup, repeat)

            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

    except Exception:
        # Re-raise full traceback (captured by the caller)
        import traceback as _tb
        raise RuntimeError(_tb.format_exc()) from None

    # ------------ Aggregate results -------------------------------------
    result: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reference_file": str(ref_py),
        "candidate_file": str(test_py),
        "tolerance": tol,
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
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
        "align_stats": align_stats,  # Alignment summary (incl. whether Model→ModelNew pair-specific aligner was used)
    }
    return result


# =========================== CLI wrapper ==================================
def _cli():
    p = argparse.ArgumentParser(description="Compare & bench two model files.")
    p.add_argument("reference", type=Path, help="Path to reference .py")
    p.add_argument("candidate", type=Path, help="Path to candidate .py")
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
