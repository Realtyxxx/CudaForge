"""TVM FFI 代码生成工具。

根据 ``KernelSignature`` 自动生成 TVM C 接口封装，避免 LLM 手写 FFI。
"""

from __future__ import annotations

from textwrap import dedent
from typing import List

from utils.kernel_signature import ArgRole, ArgType, KernelArg, KernelSignature

# ---------------------------------------------------------------------------
# 模板：保持与手写 FFI 一致的 include 集合与导出宏
# ---------------------------------------------------------------------------

TVM_FFI_TEMPLATE = dedent(
    r"""
    #include <dlpack/dlpack.h>
    #include <stdexcept>
    #include <stdint.h>
    #include "tvm/ffi/container/tensor.h"
    #include "tvm/ffi/dtype.h"
    #include "tvm/ffi/error.h"
    #include "tvm/ffi/extra/c_env_api.h"
    #include "tvm/ffi/function.h"


    // ---- TVM runtime binding (typed PackedFunc) --------------------------------
    static inline const float *_ptr_from_dl(const DLTensor *t) {{
      return reinterpret_cast<const float *>(static_cast<const char *>(t->data) + t->byte_offset);
    }}
    static inline float *_ptr_from_dl(DLTensor *t) {{
        return reinterpret_cast<float *>(static_cast<char *>(t->data)+t->byte_offset);
    }}
    static inline int64_t _numel(const DLTensor *t) {{
    int64_t n = 1;
    for (int i = 0; i < t->ndim; ++i)
        n *= t->shape[i];
    return n;
    }}

    extern "C" void {user_launch_name}({user_sig_decl});

    // Auto-generated TVM wrapper
    namespace {{
    void launch_kernel_tvm_entry({tvm_args_decl}) {{
        {null_checks}
        {user_launch_name}({call_args});
    }}
    }} // namespace

    TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_kernel_tvm, launch_kernel_tvm_entry);
    """
).strip()


# ---------------------------------------------------------------------------
# 辅助：类型映射 / 代码片段拼接
# ---------------------------------------------------------------------------

def _tensor_c_type(arg: KernelArg, const_input: bool = True) -> str:
    """根据 dtype/role 生成 C 指针类型。默认输入为 const。"""
    base = _dtype_to_c_scalar(arg.dtype)
    if arg.role == ArgRole.INPUT and const_input:
        return f"const {base} *{arg.name}"
    return f"{base} *{arg.name}"


def _dtype_to_c_scalar(dtype: str | None) -> str:
    """将 dtype 字符串映射到 C 基本类型名称。未知类型回退为 float。"""
    if dtype is None:
        return "float"
    name = dtype.lower()
    if name in ("float", "float32"):
        return "float"
    if name in ("double", "float64"):
        return "double"
    if name in ("half", "float16", "__half"):
        return "half"  # TODO: 需要用户源码自行包含 <cuda_fp16.h>
    return "float"


def _build_user_sig(signature: KernelSignature) -> str:
    parts: List[str] = []
    for arg in signature.args:
        if arg.arg_type == ArgType.TENSOR:
            parts.append(_tensor_c_type(arg))
        elif arg.arg_type == ArgType.INT:
            parts.append(f"int64_t {arg.name}")
        elif arg.arg_type == ArgType.FLOAT:
            parts.append(f"float {arg.name}")
    return ", ".join(parts)


def _build_tvm_sig(signature: KernelSignature) -> str:
    parts: List[str] = []
    for arg in signature.args:
        if arg.arg_type == ArgType.TENSOR:
            parts.append(f"DLTensor *{arg.name}")
        elif arg.arg_type == ArgType.INT:
            parts.append(f"int64_t {arg.name}")
        elif arg.arg_type == ArgType.FLOAT:
            parts.append(f"double {arg.name}")
    return ", ".join(parts)


def _build_null_checks(signature: KernelSignature, kernel_name: str) -> str:
    tensor_args = [a.name for a in signature.args if a.arg_type == ArgType.TENSOR]
    if not tensor_args:
        return "// 无需校验的张量参数"
    cond = " || ".join(f"{name} == nullptr" for name in tensor_args)
    return (
        f"if ({cond}) {{\n"
        f"        throw std::runtime_error(\"{kernel_name} received null tensor\");\n"
        f"    }}"
    )


def _build_call_args(signature: KernelSignature) -> str:
    args: List[str] = []
    for arg in signature.args:
        if arg.arg_type == ArgType.TENSOR:
            args.append(f"_ptr_from_dl({arg.name})")
        elif arg.arg_type == ArgType.FLOAT:
            args.append(f"static_cast<float>({arg.name})")
        else:  # INT or fallback
            args.append(arg.name)
    return ", ".join(args)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def generate_tvm_wrapper(
    signature: KernelSignature,
    user_launch_name: str = "launch_kernel",
) -> str:
    """生成与 ``KernelSignature`` 对应的 TVM FFI 入口代码字符串。"""
    user_sig = _build_user_sig(signature)
    tvm_sig = _build_tvm_sig(signature)
    null_checks = _build_null_checks(signature, user_launch_name)
    call_args = _build_call_args(signature)

    return TVM_FFI_TEMPLATE.format(
        user_launch_name=user_launch_name,
        user_sig_decl=user_sig,
        tvm_args_decl=tvm_sig,
        null_checks=null_checks,
        call_args=call_args,
    )


__all__ = ["generate_tvm_wrapper"]
