"""Shared registry and helpers for kernel input generation."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from utils.kernel_signature import KernelSignature, KernelArg, ArgRole, ArgType, get_signature

TensorShapeMap = Mapping[str, Sequence[int]]
InputGenerator = Callable[..., TensorShapeMap]


def random_tensors_from_signature(
    signature: KernelSignature,
    tensor_shapes: TensorShapeMap,
    device: torch.device | str | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Materialize random tensors aligned with a ``KernelSignature``."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    def _resolve_shape(arg_name: str) -> tuple[int, ...]:
        shape_spec = tensor_shapes.get(arg_name)
        if shape_spec is None:
            raise ValueError(f"Missing shape for tensor argument '{arg_name}'")
        return tuple(int(dim) for dim in shape_spec)

    def _resolve_dtype(arg: KernelArg) -> torch.dtype:
        dtype_name = arg.dtype or "float32"
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported dtype '{dtype_name}' for argument '{arg.name}'")
        return dtype

    inputs: list[torch.Tensor] = []
    workspace: list[torch.Tensor] = []
    output_tensor: torch.Tensor | None = None
    tensor_map: dict[str, torch.Tensor] = {}

    for arg in signature.args:
        if arg.arg_type != ArgType.TENSOR:
            continue

        shape = _resolve_shape(arg.name)
        dtype = _resolve_dtype(arg)

        if arg.role == ArgRole.INPUT:
            tensor = torch.randn(*shape, device=device, dtype=dtype).contiguous()
            inputs.append(tensor)
        elif arg.role == ArgRole.OUTPUT:
            tensor = torch.zeros(*shape, device=device, dtype=dtype).contiguous()
            if output_tensor is not None:
                raise ValueError("Multiple OUTPUT tensors are not supported yet")
            output_tensor = tensor
        elif arg.role == ArgRole.WORKSPACE:
            tensor = torch.zeros(*shape, device=device, dtype=dtype).contiguous()
            workspace.append(tensor)
        else:  # pragma: no cover
            raise RuntimeError(f"Unsupported tensor role {arg.role}")

        tensor_map[arg.name] = tensor

    if output_tensor is None:
        raise ValueError("Signature must define at least one OUTPUT tensor")

    return inputs, output_tensor, workspace, tensor_map


_INPUT_GENERATORS: dict[str, InputGenerator] = {}


def register_input_generator(name: str, generator: InputGenerator) -> None:
    """Register a default tensor-shape generator for a kernel name."""
    _INPUT_GENERATORS[name] = generator


def get_input_generator(name: str) -> InputGenerator:
    try:
        return _INPUT_GENERATORS[name]
    except KeyError as exc:
        raise KeyError(f"No input generator registered for kernel '{name}'") from exc


def build_registered_inputs(
    kernel_name: str,
    *,
    device: torch.device | str | None = None,
    generator_kwargs: Optional[dict[str, Any]] = None,
    signature_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[KernelSignature, list[torch.Tensor], torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Create signature-aligned tensors using the registered generator for a kernel."""

    signature = get_signature(kernel_name, **(signature_kwargs or {}))

    tensor_shapes = get_input_generator(kernel_name)(**(generator_kwargs or {}))
    inputs, output, workspace, tensor_map = random_tensors_from_signature(
        signature,
        tensor_shapes,
        device=device,
    )
    return signature, inputs, output, workspace, tensor_map

# ! Add more input generators here


def _matmul_shape_generator(m: int = 4096, n: int = 4096, k: int = 4096) -> dict[str, tuple[int, ...]]:
    return {
        "A": (m, k),
        "B": (k, n),
        "C": (m, n),
    }


def _rmsnorm_shape_generator(batch: int = 16, features: int = 64, inner: int = 4) -> dict[str, tuple[int, ...]]:
    shape = (batch, features, inner)
    return {
        "X": shape,
        "Y": shape,
    }


def _attention_shape_generator(batch: int = 4, heads: int = 32, seq: int = 512, dim: int = 128) -> dict[str, tuple[int, ...]]:
    shape = (batch, heads, seq, dim)
    return {
        "Q": shape,
        "K": shape,
        "V": shape,
        "O": shape,
    }


register_input_generator("matmul", _matmul_shape_generator)
register_input_generator("rmsnorm", _rmsnorm_shape_generator)
register_input_generator("attention", _attention_shape_generator)


__all__ = [
    "build_registered_inputs",
    "get_input_generator",
    "register_input_generator",
    "random_tensors_from_signature",
]
