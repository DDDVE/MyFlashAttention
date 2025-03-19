import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
import time
from torch.nn.functional import scaled_dot_product_attention
import ctypes

torch.set_printoptions(precision=6)
# 确保自动混合精度处于禁用状态
torch.cuda.amp.autocast(enabled=False)

def flops(B, H, N, d, causal=False, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * B * N**2 * H * d // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def get_flops_speed(B, H, N, d, time, causal=False, mode="fwd"):
    f = flops(B, H, N, d)
    return f / time / 1e12

# 动态加载扩展
# 最优版
flash_attention_best = load(
    name="flash_attention_best",
    sources=["flash_attention_extension.cpp", "flash_attention.cu"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "--expt-relaxed-constexpr",
        "-O3",
        "-lineinfo",
        "--use_fast_math",
        "-maxrregcount=64",
        "-gencode=arch=compute_75,code=sm_75",  # 根据你的GPU修改架构
        "-Xptxas -v"
    ],
    verbose=True
)

# 原始版
flash_attention_origin = load(
    name="flash_attention_origin",
    sources=["flash_attention_extension.cpp", "flash_attention.cu"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "--expt-relaxed-constexpr",
        "-O3",
        "-lineinfo",
        "--use_fast_math",
        "-maxrregcount=64",
        "-gencode=arch=compute_75,code=sm_75",  # 根据你的GPU修改架构
        "-Xptxas -v"
    ],
    verbose=True
)





assert torch.cuda.is_available(), "CUDA is not available, check your environment setup."
# 设定固定的随机种子
seed = 42
torch.manual_seed(seed)  # 设定 CPU 随机种子
torch.cuda.manual_seed(seed)  # 设定 GPU 随机种子
torch.cuda.manual_seed_all(seed)  # 确保所有 GPU 设定相同的随机数
torch.backends.cudnn.deterministic = True  # 让 cuDNN 计算可复现
torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的动态优化，保证结果一致
for i in range(5):
    # 定义输入张量
    # B, H, N, d = 64, 16, 1024, 64
    B, H, N, d = 4, 6, 128, 32
    print(f"B={B}, H={H}, N={N}, d={d}")
    Q = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
    K = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
    V = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
    O = torch.zeros_like(Q)
    # 调用扩展函数
    # flash_attention_best.flash_attention_best(Q, K, V, O, B, H, N, d)
    flash_attention_origin.flash_attention_origin(Q, K, V, O, B, H, N, d)
    print('cuda kernel finish')

    # 验证结果
    # pytorch实现
    # Q_cpu = Q.cpu().float()
    # K_cpu = K.cpu().float()
    # V_cpu = V.cpu().float()
    # Q_scaled = Q * (d ** 0.5)
    start = time.time()
    QKV = (F.softmax(Q @ K.transpose(-2,-1), dim=3)) @ V  
    torch.cuda.synchronize()
    end = time.time()
    flops_speed = get_flops_speed(B, H, N, d, (end-start))
    print(f"pytorch exec time = {(end-start) * 1000}ms, speed = {flops_speed}TFLops/s")

    diff = QKV - O
    print(diff.abs().max())

    # 检查 NaN
    assert not torch.isnan(O).any(), f"********Kernel result contains NaN at iteration {i}********"
    assert torch.allclose(QKV,O, atol=1e-2), f"********Kernel result wrong at iteration {i}"
    print("Kernel result correct")
    print(QKV.dtype)
    del Q, K, V, O, QKV
    torch.cuda.empty_cache()
