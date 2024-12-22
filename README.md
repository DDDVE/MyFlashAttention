# MyFlashAttention
implement flash attention 1 and 2, and compare them with function scaled_dot_product_attention

# 运行环境
1.如果报错matplotlib缺少libstdc++29的库, 安装后需要export LD_LIBRARY_PATH, 例如/root/miniforge3/lib

# 如何使用
运行test_and_draw.py会在当前目录下生成文件夹, 并写入三种算法的运行时间对比图

例子:
![](test_result_20241218_193618/performance_B1_H1_d32.png)

# 实验结果
1. 三种方法的运行时间都随数据规模的增大而增加. 其中flash attn1的运行时间增加得最多, scaled_dot_product_attention的运行时间增加得最少
2. 在大多数BHd的取值中, 当N<=1024时, flash attn2的运行时间可以低于scaled_dot_product_attention. 随着N的增大, flash attn2的运行时间会逐渐超过scaled_dot_product_attention, 且差距逐渐增大
4. 除了N以外, d的取值对flash attn的运行时间影响也很大, 当d较小(如等于4时), 即使N=5k, scaled_dot_product_attention的运行时间也能达到flash attn2的60%; 但当d较大(如等于32)时, scaled_dot_product_attention在N=5k时的运行时间只有flash attn2的10%
5. scaled_dot_product_attention的运行时间对N和d的变化相对不敏感
6. B和H对三种方法的影响都较大. 对于scaled_dot_product_attention而言, 当N越大时, B和H对其的影响也会增大. 例如当N=5k, 且B或H翻倍时, scaled_dot_product_attention的运行时间也接近翻倍

# 我在实现中采用了哪些方法提升性能
1. 将数据分块从全局内存加载到共享内存, 减少访存开销
2. 使用warp归约求和以及最大值, 减少线程间同步开销
3. 在cuda编译指令中限制线程寄存器数量, 提升SM的并发度
4. 指令向量化, 使用float4实现连续访存, IO次数减少75%, 且更好地利用了内存带宽

# 如何进一步提升性能
1. 扩大共享内存, 减少矩阵的分块数量, 从而减少核函数内的循环次数
2. 将B和H分割, 放入不同的流中并行, 进一步提升并发度
