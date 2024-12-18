import torch
import time
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load
from torch.nn.functional import scaled_dot_product_attention
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pytz

torch.set_printoptions(precision=6)
# 确保自动混合精度处于禁用状态
torch.cuda.amp.autocast(enabled=False)

# 加载flash attn1和flash attn2的扩展
flash_attention = load(
    name="flash_attention",
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

flash_attention2 = load(
    name="flash_attention2",
    sources=["flash_attention2/flash_attention2_extension.cpp", "flash_attention2/flash_attention2.cu"],
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
N_values = [128, 256, 512, 1024, 1024*2, 1024*3, 1024*4, 1024*5]
d_values = [4, 8, 16, 32]

# 保存所有结果
all_results = []

# 创建目录
tz = pytz.timezone('Asia/Shanghai')
timestamp = datetime.now(tz).strftime('%Y%m%d_%H%M%S')
output_dir = f"test_result_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

for B in B_values:
    for H in H_values:
        for d in d_values:
            results = [] # 
            for N in N_values:
                print(f"Running for B={B}, H={H}, N={N}, d={d}")
                # 每次重新分配张量
                Q = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
                K = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
                V = torch.rand((B, H, N, d), device='cuda', dtype=torch.float32).contiguous()
                O = torch.zeros_like(Q)
                # 运行cudnn
                Q_scaled = Q * (d ** 0.5)
                start = time.time()
                QKV = scaled_dot_product_attention(Q_scaled, K, V) 
                torch.cuda.synchronize()
                end = time.time()

                # 运行flash attention-1
                time_fa1 = flash_attention.flash_attention(Q, K, V, O, B, H, N, d)
                # diff = QKV-O
                # print("diff:", diff)
                assert torch.allclose(QKV,O, atol=1e-4), "flash attention-1 result wrong"

                # 运行flash attention-2
                O[:B, :H, :N, :d].zero_()
                time_fa2 = flash_attention2.flash_attention2(Q, K, V, O, B, H, N, d)
                assert torch.allclose(QKV,O, atol=1e-4), "flash attention-2 result wrong"

                del Q, K, V, O
                torch.cuda.empty_cache()
                results.append({
                    'N': N,
                    'cudnn': (end-start)*1000,
                    'fa1': time_fa1.item(),
                    'fa2': time_fa2.item()
                })
            '''画图'''
            # 提取数据
            Ns = [r['N'] for r in results]
            cudnn_times = [r['cudnn'] for r in results]
            fa1_times = [r['fa1'] for r in results]
            fa2_times = [r['fa2'] for r in results]
            # 设置柱状图宽度
            width = 0.25
            x = np.arange(len(Ns))  # x轴的位置
            # 创建图形
            plt.figure(figsize=(12, 8))
            bar1 = plt.bar(x - width, cudnn_times, width, label='CUDNN', color='blue')
            bar2 = plt.bar(x, fa1_times, width, label='Flash Attention 1', color='green')
            bar3 = plt.bar(x + width, fa2_times, width, label='Flash Attention 2', color='red')
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


                




