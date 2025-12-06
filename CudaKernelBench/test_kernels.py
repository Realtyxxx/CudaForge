"""CUDA kernel smoke tests driven by kernel/input registries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Tuple
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.compile_and_run import (  # noqa: E402
    KernelRunner,
    TVM_AVAILABLE,
)
from utils.input_generators import build_registered_inputs  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATOL = 1e-4
RTOL = 1e-4

TensorMap = Dict[str, torch.Tensor]
RefFn = Callable[[TensorMap, Dict[str, float]], torch.Tensor]


@dataclass
class KernelTest:
    name: str
    cu_file: str
    ref_fn: RefFn
    generator_kwargs: Dict[str, int] = field(default_factory=dict)
    signature_kwargs: Dict[str, float] = field(default_factory=dict)


def _matmul_ref(tensors: TensorMap, _: Dict[str, float]) -> torch.Tensor:
    return torch.matmul(tensors["A"], tensors["B"])


def _rmsnorm_ref(tensors: TensorMap, sig_kwargs: Dict[str, float]) -> torch.Tensor:
    eps = float(sig_kwargs.get("eps", 1e-5))
    x = tensors["X"]
    return x / torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)


def _attention_ref(tensors: TensorMap, _: Dict[str, float]) -> torch.Tensor:
    q = tensors["Q"]
    k = tensors["K"]
    v = tensors["V"]
    dim = q.shape[-1]
    scale = 1.0 / (dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    att = torch.softmax(scores, dim=-1)
    return torch.matmul(att, v)


KERNEL_TESTS = (
    KernelTest(
        name="matmul",
        cu_file="matmul.cuh",
        ref_fn=_matmul_ref,
        generator_kwargs={
            "m": 128,
            "n": 128,
            "k": 128,
        },
    ),
    KernelTest(
        name="matmul",
        cu_file="new_matmul_10.cuh",
        ref_fn=_matmul_ref,
        generator_kwargs={
            "m": 128,
            "n": 128,
            "k": 128,
        },
    ),
    KernelTest(
        name="rmsnorm",
        cu_file="rmsnorm.cuh",
        ref_fn=_rmsnorm_ref,
        generator_kwargs={"batch": 16, "features": 64, "inner": 4},
        signature_kwargs={"eps": 1e-5},
    ),
    KernelTest(
        name="attention",
        # cu_file="attention.cuh",
        cu_file="new_flash_attention.cuh",
        ref_fn=_attention_ref,
        generator_kwargs={"batch": 2, "heads": 2, "seq": 8, "dim": 16},
    ),
)


def run_kernel_test(test: KernelTest) -> Tuple[bool, float]:
    signature, inputs, output, workspace, tensor_map = build_registered_inputs(
        test.name,
        device=DEVICE,
        generator_kwargs=test.generator_kwargs,
        signature_kwargs=test.signature_kwargs,
    )
    runner = KernelRunner(test.cu_file, signature=signature)
    ref = test.ref_fn(tensor_map, test.signature_kwargs).to(dtype=output.dtype, device=output.device)
    runner(inputs, output, signature=signature, workspace=workspace or None)
    max_err = (output - ref).abs().max().item()
    ok = torch.allclose(output, ref, atol=ATOL, rtol=RTOL)
    return ok, max_err


def main():
    torch.manual_seed(0)

    if not TVM_AVAILABLE:
        print("WARNING: TVM is not available. Install with: pip install apache-tvm")
        print("Tests will fail without TVM.")
        return

    if DEVICE.type != "cuda":
        print("CUDA device not available; kernels require a GPU to run.")
        return

    print("Running kernel tests with TVM FFI...\n")

    all_ok = True
    for test in KERNEL_TESTS:
        ok, max_err = run_kernel_test(test)
        print(f"[{Path(test.cu_file).stem}] correct: {ok}, max error: {max_err:.3e}")
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
