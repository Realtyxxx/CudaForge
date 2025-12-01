from __future__ import annotations
"""Prompt builder for Mind‑Evolution CUDA‑kernel search (seed‑kernel version).

Generates a **single prompt** that contains:
1. Target GPU spec (from `prompts/hardware/gpu_specs.py`)
2. **Few‑shot pair** – original *and* optimised model code blocks
3. Source architecture (`class Model`) that needs to be optimised
4. Existing kernel summaries (optional, for diversity context)
5. A **diversity requirement** section ensuring the new kernel differs from all previous ones
6. Output requirements

CLI usage
---------
```bash
python -m prompts.build_prompt KernelBench/level1/19_ReLU.py \
       --gpu "Quadro RTX 6000" -o prompt.txt
```
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
# Few‑shot pair  (before / after)
# --------------------------------------------------
FEWSHOT_BASE = ROOT / "prompts/few_shot/model_ex_add.py"   # original Model
FEWSHOT_NEW = ROOT / "prompts/few_shot/model_new_ex_add.py"  # optimised ModelNew

# ---------------------------------------------------------------------------
# Prompt template (with diversity requirement)
# ---------------------------------------------------------------------------
SEED_PROMPT_TEMPLATE = Template(
    dedent(
        """
You write custom CUDA kernels to replace the PyTorch operators in the given architecture
to get speedups. You have complete freedom to choose the set of operators you want to replace.
You may replace multiple operators with custom implementations, consider operator fusion
opportunities (for example, combining matmul+relu), or algorithmic changes (such as online
softmax). You are only limited by your imagination.

Target GPU
==========
GPU Name: $gpu_name
Architecture: $gpu_arch
Details:
$gpu_items

GPU glossary (quick reference):
$gpu_definitions

GPU‑architecture best practices to respect:
$gpu_best_practices

Here’s an example to show you the syntax of inline embedding custom CUDA operators in torch:
The example given architecture is:
```
$few_base
```
The example new arch with custom CUDA kernels looks like this:
```
$few_new
```

You are given the following architecture:
```python
$arch_src
```
Optimize the architecture named Model with custom CUDA operators! Name your optimized
output architecture ModelNew. Output the new code in codeblocks. Please generate real
code, NOT pseudocode, make sure the code compiles and is fully functional. Just output
the new model code, no other text, and NO testing code!
"""
    )
)
TEMPLATE = Template(
    dedent(
        """
Task
----
Generate **hand‑written CUDA kernels** that replace *all* PyTorch operator(s)
inside the original `class Model` (shown later).  You may fuse multiple
operators into a single kernel if that yields better performance.  Leave any
non‑replaced parts of the model unchanged.

OUTPUT RULES (STRICT) ────────────────────────────────────────────────
1. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple‑quoted CUDA string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls
      your CUDA kernels.
2. **Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose.


Few‑shot example (reference only – do **not** echo):
**Original**
```python
$few_base
```
**Optimised**
```python
$few_new
```

Target architecture (to optimise):
```python
$arch_src
```

Optimize the architecture named Model with custom CUDA operators! Name your optimized
output architecture ModelNew. Output the new code in codeblocks. Please generate real
code, NOT pseudocode, make sure the code compiles and is fully functional. Just output
the new model code, no other text, and NO testing code!

Example:
```python
# <complete ModelNew code>
```
# ==========================================================
"""
    )
)
default_system_prompt = """\
You are a senior CUDA-kernel optimisation specialist. Your job is to generate a high-quality,
compilable, and runnable Python script that builds and launches **hand-written CUDA kernels**.

OUTPUT RULES (STRICT):
output the code within:
```python
# <complete ModelNew code>
```

"""
# ---------------------------------------------------------------------------
# GPU spec loader
# ---------------------------------------------------------------------------

def _load_gpu_spec() -> tuple[dict, dict, list[str]]:  # noqa: D401
    """Import `gpu_specs.py` and return all GPU metadata (robust across Python versions)."""
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module  # avoid re‑import
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    for attr in ("GPU_SPEC_INFO", "GPU_DEFINITIONS", "GPU_BEST_PRACTICES"):
        if not hasattr(module, attr):
            raise AttributeError(f"{attr} not defined in gpu_specs.py")
    return (
        module.GPU_SPEC_INFO,  # type: ignore[attr-defined]
        module.GPU_DEFINITIONS,  # type: ignore[attr-defined]
        module.GPU_BEST_PRACTICES,  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# Prompt builder core
# ---------------------------------------------------------------------------

def build_seed_prompt(
    arch_path: Path,
    gpu_name: str | None = None,
) -> str:
    """Build LLM prompt for CUDA‑kernel optimisation (seed generation)."""
    gpu_spec, gpu_definitions, gpu_best_practices = _load_gpu_spec()

    # Auto‑detect GPU if not provided
    if gpu_name is None:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_spec:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    info = gpu_spec[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(
        f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    )
    gpu_reference = "\n".join(
        f"• {term}: {definition}" for term, definition in gpu_definitions.items()
    )
    gpu_best = "\n".join(f"• {item}" for item in gpu_best_practices)

    few_base = FEWSHOT_BASE.read_text().strip()
    few_new = FEWSHOT_NEW.read_text().strip()
    arch_src = Path(arch_path).read_text().strip()

    return SEED_PROMPT_TEMPLATE.substitute(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        gpu_definitions=gpu_reference,
        gpu_best_practices=gpu_best,
        few_base=few_base,
        few_new=few_new,
        arch_src=arch_src,
    )


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _cli() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Build LLM prompt for CUDA‑kernel optimisation (seed generation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_py", help="Path to .py containing class Model")
    parser.add_argument("--gpu", default=None, help="GPU name key in gpu_specs.py")
    parser.add_argument("-o", "--out", help="Save prompt to file")
    args = parser.parse_args()

    prompt = build_seed_prompt(Path(args.model_py), args.gpu)

    if args.out:
        Path(args.out).write_text(prompt)
        print(f"[✓] Prompt saved to {args.out}")
    else:
        print(prompt)


if __name__ == "__main__":  # pragma: no cover
    _cli()
