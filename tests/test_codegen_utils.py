import re

from utils.codegen_utils import generate_tvm_wrapper
from utils.kernel_signature import ArgRole, ArgType, KernelArg, KernelSignature


def test_generate_tvm_wrapper_includes_and_export():
    sig = KernelSignature(
        args=[
            KernelArg("A", ArgType.TENSOR, ArgRole.INPUT, "float32"),
            KernelArg("B", ArgType.TENSOR, ArgRole.INPUT, "float32"),
            KernelArg("C", ArgType.TENSOR, ArgRole.OUTPUT, "float32"),
            KernelArg("W", ArgType.TENSOR, ArgRole.WORKSPACE, "float32"),
            KernelArg("M", ArgType.INT),
            KernelArg("alpha", ArgType.FLOAT),
        ]
    )

    code = generate_tvm_wrapper(sig, user_launch_name="launch_kernel")
    print(code)

    # 基本包含检查：tvm 头、导出宏、extern 声明
    assert '#include "tvm/ffi/function.h"' in code
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_kernel_tvm, launch_kernel_tvm_entry);" in code
    assert 'extern "C" void launch_kernel(' in code

    # TVM entry 参数列表应包含 DLTensor* + 标量
    assert "DLTensor *A" in code and "DLTensor *B" in code
    assert "int64_t M" in code
    assert "double alpha" in code  # float 标量传 double

    # 空指针检查包含全部张量名
    null_check = re.search(r"if \((.*?)\) \{", code)
    assert null_check is not None
    for name in ("A", "B", "C", "W"):
        assert name in null_check.group(1)

    # 调用参数顺序保持签名顺序，并对 float 标量使用 static_cast
    call_sites = re.findall(r"launch_kernel\((.*?)\);", code)
    assert call_sites, "launch_kernel call not found"
    call_args = call_sites[-1]  # 取 wrapper 中的实际调用
    expected_order = ["_ptr_from_dl(A)", "_ptr_from_dl(B)", "_ptr_from_dl(C)", "_ptr_from_dl(W)", "M", "static_cast<float>(alpha)"]
    for idx, token in enumerate(expected_order):
        assert token in call_args.split(", ")[idx]


if __name__ == "__main__":
    test_generate_tvm_wrapper_includes_and_export()
