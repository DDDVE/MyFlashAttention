# MyFlashAttention
implement flash attention 1 and 2, and compare them with function scaled_dot_product_attention in cudnn

运行环境
    1.如果报错matplotlib缺少libstdc++29的库, 安装后需要export LD_LIBRARY_PATH, 例如/root/miniforge3/lib

运行test_and_draw.py会在当前目录下生成文件夹, 并写入三种算法的运行时间对比图
