from __future__ import annotations
"""Prompt builder for generating standalone CUDA `.cu` kernels (seed generation).

The resulting prompt includes:
1. Target GPU spec (from `prompts/hardware/gpu_specs.py`)
2. Few‑shot CUDA examples (baseline + optimised)
3. The task source (reference implementation/spec)
4. Output rules for emitting a single CUDA source with `extern "C" launch_kernel`.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from string import Template
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"  # GPU spec table

# --------------------------------------------------
# Few‑shot pair (CUDA baseline / optimised)
# --------------------------------------------------
FEWSHOT_BASE = ROOT / "prompts/cuda_fewshot/attention.cu"
FEWSHOT_NEW = ROOT / "prompts/cuda_fewshot/new_flash_attn.cu"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SEED_PROMPT = Template(
    dedent(
        """
You write **standalone CUDA C++**. Your output will be compiled with `nvcc`
and invoked through an `extern "C"` entry point named `launch_kernel` that launches
your kernels. Keep the `launch_kernel` argument list and outputs identical to the
reference computation.

Target GPU
- Name: $gpu_name
- Architecture: $gpu_arch
$gpu_items

Few-shot reference (do not copy verbatim; follow the style/structure)
--- Baseline CUDA ---
```cuda
$few_base
```
--- Optimized CUDA ---
```cuda
$few_new
```

Naive CUDA baseline (optimization target; keep interface identical)
```cuda
$naive_cuda
```

Output requirements (STRICT)
- Return **one** fenced block: ```cuda ... ```
- Provide all necessary headers, device/global kernels, and helper functions (if needed).
- Expose an `extern "C"` function `launch_kernel(...)` that sets up grids/blocks and calls your kernels.
- Match the computation and tensor shapes/dtypes implied by the reference; no extra APIs.
- Do NOT include any TVM/DLPack/tvm_ffi headers or macros; only CUDA code plus `launch_kernel`.
- No extra prose or non-CUDA artifacts.
"""
    )
)

default_system_prompt = """\
You are a senior CUDA-kernel optimisation specialist. Generate a single, compilable
CUDA C++ source file (.cu) containing device/global kernels plus an `extern "C"`
entry `launch_kernel(...)`. Output only one fenced block marked `cuda` with the full
code and nothing else. Preserve the `launch_kernel` signature and outputs; include only CUDA kernels,
helpers, and the launch function. Do NOT emit any TVM/DLPack/tvm_ffi headers or macros.
"""
# ---------------------------------------------------------------------------
# GPU spec loader
# ---------------------------------------------------------------------------


def _load_gpu_spec() -> dict:  # noqa: D401
    """Import `gpu_specs.py` and return the GPU_SPEC_INFO dict (robust across Python versions)."""
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module  # avoid re‑import
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "GPU_SPEC_INFO"):
        raise AttributeError("GPU_SPEC_INFO not defined in gpu_specs.py")
    return module.GPU_SPEC_INFO  # type: ignore[attr-defined]


def _load_naive_cuda(arch_path: Path) -> str:
    """加载 naive CUDA 实现（优先 .cuh，其次 .cu）。"""
    candidates = []
    if arch_path.suffix in (".cuh", ".cu") and arch_path.exists():
        if arch_path.suffix == ".cuh":
            candidates.append(arch_path)
            cu_peer = arch_path.with_suffix(".cu")
            if cu_peer.exists():
                candidates.append(cu_peer)
        else:
            cuh_peer = arch_path.with_suffix(".cuh")
            if cuh_peer.exists():
                candidates.append(cuh_peer)
            candidates.append(arch_path)

    bench_dir = ROOT / "CudaKernelBench"
    candidates.append(bench_dir / f"{arch_path.stem}.cuh")
    candidates.append(bench_dir / f"{arch_path.stem}.cu")

    for path in candidates:
        if path.exists() and "launch_kernel" in path.read_text():
            return path.read_text().strip()
    raise FileNotFoundError(f"Naive CUDA implementation not found for {arch_path}")


# ---------------------------------------------------------------------------
# Prompt builder core
# ---------------------------------------------------------------------------

def build_seed_prompt(
    arch_path: Path,
    gpu_name: str | None = None,
) -> str:
    """Build LLM prompt for CUDA‑kernel optimisation (seed generation)."""
    gpu_info = _load_gpu_spec()

    # Auto‑detect GPU if not provided
    if gpu_name is None:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(
        f"- {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    )

    few_base = FEWSHOT_BASE.read_text().strip()
    few_new = FEWSHOT_NEW.read_text().strip()
    target_src = Path(arch_path).read_text().strip()
    naive_cuda = _load_naive_cuda(Path(arch_path))

    return SEED_PROMPT.substitute(
        few_base=few_base,
        few_new=few_new,
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        target_src=target_src,
        naive_cuda=naive_cuda,
    )


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _cli() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Build LLM prompt for CUDA‑kernel optimisation (seed generation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "kernel_src",
        help="Path to the target CUDA kernel (.cuh/.cu) describing the computation to optimise",
    )
    parser.add_argument("--gpu", default=None, help="GPU name key in gpu_specs.py")
    parser.add_argument("-o", "--out", help="Save prompt to file")
    args = parser.parse_args()

    prompt = build_seed_prompt(Path(args.kernel_src), args.gpu)

    if args.out:
        Path(args.out).write_text(prompt)
        print(f"[✓] Prompt saved to {args.out}")
    else:
        print(prompt)


if __name__ == "__main__":  # pragma: no cover
    _cli()
