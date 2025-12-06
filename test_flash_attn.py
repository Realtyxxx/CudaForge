#!/usr/bin/env python3
"""
测试 FlashAttention kernel 的正确性
"""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.compile_and_run import compile_cuda, load_and_run_cuda

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATOL = 1e-3  # FlashAttention 可能需要稍微宽松的容差
RTOL = 1e-3


def random_attention_inputs(batch: int, heads: int, seq: int, dim: int):
    q = torch.randn(batch, heads, seq, dim, device=DEVICE, dtype=torch.float32).contiguous()
    k = torch.randn(batch, heads, seq, dim, device=DEVICE, dtype=torch.float32).contiguous()
    v = torch.randn(batch, heads, seq, dim, device=DEVICE, dtype=torch.float32).contiguous()
    o = torch.empty_like(v)
    return q, k, v, o


def check_flash_attention(batch: int = 2, heads: int = 2, seq: int = 64, dim: int = 32) -> tuple[bool, float]:
    """测试 FlashAttention kernel"""
    cu_path = Path(__file__).parent / "prompts" / "cuda_fewshot" / "new_ex_flash_attn.cu"
    
    print(f"编译 {cu_path}...")
    so_path = compile_cuda(cu_path)
    print(f"编译成功: {so_path}")
    
    q, k, v, o = random_attention_inputs(batch, heads, seq, dim)
    scale = 1.0 / (dim ** 0.5)
    
    # PyTorch 参考实现
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    att = torch.softmax(scores, dim=-1)
    ref = torch.matmul(att, v)
    
    # 运行 CUDA kernel
    print("运行 CUDA kernel...")
    load_and_run_cuda(so_path, "attention", [q, k, v], o)
    
    # 比较结果
    max_err = (o - ref).abs().max().item()
    mean_err = (o - ref).abs().mean().item()
    ok = torch.allclose(o, ref, atol=ATOL, rtol=RTOL)
    
    print(f"最大误差: {max_err:.3e}")
    print(f"平均误差: {mean_err:.3e}")
    print(f"测试通过: {ok}")
    
    return ok, max_err


def main():
    torch.manual_seed(42)
    if DEVICE.type != "cuda":
        print("CUDA 设备不可用，需要 GPU 来运行测试。")
        return
    
    print("=" * 60)
    print("测试 FlashAttention Kernel")
    print("=" * 60)
    
    # 测试不同的配置
    configs = [
        (1, 1, 32, 16),   # 小规模
        (2, 2, 64, 32),   # 中等规模
        (1, 4, 128, 64),  # 较大规模
    ]
    
    all_passed = True
    for batch, heads, seq, dim in configs:
        print(f"\n配置: batch={batch}, heads={heads}, seq={seq}, dim={dim}")
        try:
            ok, max_err = check_flash_attention(batch, heads, seq, dim)
            if not ok:
                all_passed = False
                print(f"❌ 测试失败!")
            else:
                print(f"✅ 测试通过!")
        except Exception as e:
            print(f"❌ 测试出错: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败!")
    print("=" * 60)


if __name__ == "__main__":
    main()
