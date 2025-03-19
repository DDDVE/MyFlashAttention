from torch.utils.cpp_extension import load
import torch
import os
from datetime import datetime
import pytz
import time
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 确保自动混合精度处于禁用状态
torch.cuda.amp.autocast(enabled=False)
# torch.set_flush_denormals(True)


'''加载不同版本的flash attn1扩展'''
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

B_values = [1, 2, 4, 8]
H_values = [1, 2, 4, 8]
N_values = [128, 256, 512, 1024, 1024*2, 1024*3, 1024*4]
d_values = [16, 32]

# 保存所有结果
all_results = []

# 创建目录
tz = pytz.timezone('Asia/Shanghai')
timestamp = datetime.now(tz).strftime('%Y%m%d_%H%M%S')
output_dir = f"flash_attn1_self_compare_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

for B in B_values:
    for H in H_values:
        for d in d_values:
            # mha = MultiheadAttention(d * H, H, dropout=0.0, batch_first=False).to('cuda')
            results = [] # 
            for N in N_values:
                print(f"***Running for B={B}, H={H}, N={N}, d={d}")
                # 每次重新分配张量
                Q = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
                K = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
                V = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
                O_origin = torch.zeros_like(Q)
                O_best = torch.zeros_like(Q)

                # 运行最优版
                # O[:B, :H, :N, :d].zero_()
                time_fa_best = flash_attention_best.flash_attention_best(Q, K, V, O_best, B, H, N, d)

                # 运行原始版
                torch.cuda.synchronize()
                time_fa_origin = flash_attention_origin.flash_attention_origin(Q, K, V, O_origin, B, H, N, d)
                # diff = QKV-O
                # print("diff:", diff)

                # 运行pytorch
                start = time.time()
                QKV = (F.softmax(Q @ K.transpose(-2,-1), dim=3)) @ V 
                torch.cuda.synchronize()
                end = time.time()

                diff = QKV-O_best
                print("diff:", diff)
                assert not torch.isnan(O_best).any(), f"********flash attention best result contains NaN********"
                assert torch.allclose(QKV,O_best, atol=1e-2), "flash attention best result wrong"
                assert not torch.isnan(O_origin).any(), f"********flash attention origin result contains NaN********"
                assert torch.allclose(QKV,O_origin, atol=1e-2), "flash attention origin result wrong"

                

                del Q, K, V, O_best, O_origin
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                results.append({
                    'N': N,
                    'pytorch': (end-start)*1000,
                    'fa_origin': time_fa_origin.item(),
                    'fa_best': time_fa_best.item()
                })
            '''画图'''
            # 提取数据
            Ns = [r['N'] for r in results]
            pytorch_times = [r['pytorch'] for r in results]
            fa_origin_times = [r['fa_origin'] for r in results]
            fa_best_times = [r['fa_best'] for r in results]
            # 设置柱状图宽度
            width = 0.25
            x = np.arange(len(Ns))  # x轴的位置
            # 创建图形
            plt.figure(figsize=(12, 8))
            bar1 = plt.bar(x - width, pytorch_times, width, label='pytorch', color='blue')
            bar2 = plt.bar(x, fa_origin_times, width, label='Flash Attention 1 origin', color='green')
            bar3 = plt.bar(x + width, fa_best_times, width, label='Flash Attention 1 best', color='red')
            # 添加柱子顶部的标注
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', 
                             ha='center', va='bottom', fontsize=10, rotation=45)
            
            add_labels(bar1)
            add_labels(bar2)
            add_labels(bar3)
            # 设置标题和标签
            plt.title(f"Performance for B={B}, H={H}, d={d}", fontsize=14)
            plt.xlabel("N (Sequence Length)", fontsize=12)
            plt.ylabel("Time (ms)", fontsize=12)
            plt.xticks(x, Ns, fontsize=10)
            plt.legend(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            # 保存图像或显示
            plt.tight_layout()
            # 保存图片
            filename = f"performance_B{B}_H{H}_d{d}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()
print("end")

