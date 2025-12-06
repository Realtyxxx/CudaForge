"""
Kernel signature definitions for generic CUDA kernel invocation.

This module provides data classes to define kernel argument types, roles,
and signatures, enabling dynamic kernel invocation without hardcoded logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Callable
import torch


class ArgRole(Enum):
    """Role of a kernel argument."""
    INPUT = "input"
    OUTPUT = "output"
    WORKSPACE = "workspace"


class ArgType(Enum):
    """Type of a kernel argument."""
    TENSOR = "tensor"
    INT = "int"
    FLOAT = "float"


@dataclass
class KernelArg:
    """Definition of a single kernel argument."""
    name: str
    arg_type: ArgType
    role: ArgRole = ArgRole.INPUT
    dtype: str = "float32"  # for tensors
    # Optional: function to compute value from inputs/output/ref_model
    # Signature: (inputs: List[Tensor], output: Tensor, ref_model: Any) -> value
    compute_fn: Optional[Callable] = None

    def __post_init__(self):
        # Default compute functions for common dimension parameters
        if self.compute_fn is None and self.arg_type in (ArgType.INT, ArgType.FLOAT):
            self.compute_fn = None  # Must be provided or inferred


@dataclass
class KernelSignature:
    """Complete signature for a CUDA kernel."""
    args: List[KernelArg] = field(default_factory=list)

    def get_tensor_args(self, role: Optional[ArgRole] = None) -> List[KernelArg]:
        """Get all tensor arguments, optionally filtered by role."""
        return [
            arg for arg in self.args
            if arg.arg_type == ArgType.TENSOR and (role is None or arg.role == role)
        ]

    def get_scalar_args(self) -> List[KernelArg]:
        """Get all scalar (int/float) arguments."""
        return [
            arg for arg in self.args
            if arg.arg_type in (ArgType.INT, ArgType.FLOAT)
        ]

    @property
    def num_inputs(self) -> int:
        """Number of input tensors."""
        return len(self.get_tensor_args(ArgRole.INPUT))

    @property
    def num_outputs(self) -> int:
        """Number of output tensors."""
        return len(self.get_tensor_args(ArgRole.OUTPUT))

    @property
    def num_workspace(self) -> int:
        """Number of workspace tensors."""
        return len(self.get_tensor_args(ArgRole.WORKSPACE))


# =============================================================================
# Predefined signatures for common kernels
# =============================================================================

def matmul_signature() -> KernelSignature:
    """Signature for matmul: C[M,N] = A[M,K] @ B[K,N]"""
    return KernelSignature(args=[
        KernelArg("A", ArgType.TENSOR, ArgRole.INPUT, "float32"),
        KernelArg("B", ArgType.TENSOR, ArgRole.INPUT, "float32"),
        KernelArg("C", ArgType.TENSOR, ArgRole.OUTPUT, "float32"),
        KernelArg("M", ArgType.INT, compute_fn=lambda inputs, output, ref: output.shape[-2]),
        KernelArg("N", ArgType.INT, compute_fn=lambda inputs, output, ref: output.shape[-1]),
        KernelArg("K", ArgType.INT, compute_fn=lambda inputs, output, ref: inputs[0].shape[-1]),
    ])


def rmsnorm_signature(eps: float = 1e-5) -> KernelSignature:
    """Signature for RMSNorm: Y = X / sqrt(mean(X^2) + eps)"""
    return KernelSignature(args=[
        KernelArg("X", ArgType.TENSOR, ArgRole.INPUT, "float32"),
        KernelArg("Y", ArgType.TENSOR, ArgRole.OUTPUT, "float32"),
        KernelArg("batch", ArgType.INT, compute_fn=lambda inputs, output, ref: inputs[0].shape[0]),
        KernelArg("features", ArgType.INT, compute_fn=lambda inputs, output, ref: inputs[0].shape[1]),
        KernelArg("inner", ArgType.INT, compute_fn=lambda inputs, output, ref: (
            inputs[0].numel() // max(1, inputs[0].shape[0] * inputs[0].shape[1])
        )),
        KernelArg("eps", ArgType.FLOAT, compute_fn=lambda inputs, output, ref: (
            getattr(ref, "eps", eps) if ref is not None else eps
        )),
    ])


def attention_signature() -> KernelSignature:
    """Signature for attention: O = softmax(Q @ K^T * scale) @ V"""
    return KernelSignature(args=[
        KernelArg("Q", ArgType.TENSOR, ArgRole.INPUT, "float32"),
        KernelArg("K", ArgType.TENSOR, ArgRole.INPUT, "float32"),
        KernelArg("V", ArgType.TENSOR, ArgRole.INPUT, "float32"),
        KernelArg("O", ArgType.TENSOR, ArgRole.OUTPUT, "float32"),
        KernelArg("batch", ArgType.INT, compute_fn=lambda inputs, output, ref: inputs[0].shape[0]),
        KernelArg("heads", ArgType.INT, compute_fn=lambda inputs, output, ref: inputs[0].shape[1]),
        KernelArg("seq", ArgType.INT, compute_fn=lambda inputs, output, ref: inputs[0].shape[2]),
        KernelArg("dim", ArgType.INT, compute_fn=lambda inputs, output, ref: inputs[0].shape[3]),
        KernelArg("scale", ArgType.FLOAT, compute_fn=lambda inputs, output, ref: (
            1.0 / float(inputs[0].shape[3]) ** 0.5
        )),
    ])


# Registry for looking up signatures by name
SIGNATURE_REGISTRY = {
    "matmul": matmul_signature,
    "rmsnorm": rmsnorm_signature,
    "attention": attention_signature,
}


def get_signature(name: str, **kwargs) -> KernelSignature:
    """Get a predefined signature by name."""
    if name not in SIGNATURE_REGISTRY:
        raise ValueError(f"Unknown signature: {name}. Available: {list(SIGNATURE_REGISTRY.keys())}")
    return SIGNATURE_REGISTRY[name](**kwargs)
