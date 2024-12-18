import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
import time
from torch.nn.functional import scaled_dot_product_attention

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
flash_attention2 = load(
    name="flash_attention2",
    sources=["flash_attention2_extension.cpp", "flash_attention2.cu"],
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

# 定义输入张量
# B, H, N, d = 64, 16, 1024, 64
B, H, N, d = 1, 1, 1024*4, 32
assert d%4==0 
print(f"B={B}, H={H}, N={N}, d={d}")
Q = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
K = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
V = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
O = torch.zeros_like(Q)

# print("Q:", Q)
# print("K:", K)
# print("V:", V)
# print("O:", O)

assert torch.cuda.is_available(), "CUDA is not available, check your environment setup."
# 调用扩展函数
flash_attention2.flash_attention2(Q, K, V, O, B, H, N, d)
print('cuda kernel finish')
# 验证结果

# pytorch实现
Q_cpu = Q.cpu().float()
K_cpu = K.cpu().float()
V_cpu = V.cpu().float()
Q_scaled = Q * (d ** 0.5)
start = time.time()
# QK = Q @ K.transpose(-2,-1)
#compute softmax of QK^T
# QKSoftmax = F.softmax(QK, dim=3)
# QKV = (F.softmax(Q @ K.transpose(-2,-1), dim=3)) @ V 
QKV = scaled_dot_product_attention(Q_scaled, K, V) 
torch.cuda.synchronize()
end = time.time()
flops_speed = get_flops_speed(B, H, N, d, (end-start))
print(f"pytorch exec time = {(end-start) * 1000}ms, speed = {flops_speed}TFLops/s")
diff = QKV - O
# print("QKV: ", QKV);
# print("diff: ", diff)
assert torch.allclose(QKV,O, atol=1e-4), "********Kernel result wrong"
print("Kernel result correct")
print(QKV.dtype)
