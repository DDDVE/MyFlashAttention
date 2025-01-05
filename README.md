# MyFlashAttention
implement flash attention 1 and 2, and compare them with function scaled_dot_product_attention

# 运行环境
1.如果报错matplotlib缺少libstdc++29的库, 安装后需要export LD_LIBRARY_PATH, 例如/root/miniforge3/lib

# 实验报告
## 1. Introduction
Tri Dao等人提出了flash attention-1和flash attention-2, 本实验首先使用cuda初步复现了这两个算法, 验证了结果的正确性后尝试在初步复现的基础上进一步优化, 并和pytorch的标准attention计算过程比较运行时间. 本报告章节划分如下: 第二章, 介绍flash attention-1和flash attention-2的复现过程; 第三章, 对比自身实现中优化前后的运行时间; 第四章, 与pytorch的标准attention比较运行时间; 第五章, 实验结果总结. 本实验结果均基于GPU Tesla 4.

## 2. Implement
### 2.1 flash attention-1
#### 2.1.1 Preliminary Implementation
实现代码在flash_attention.cu中.

flash attention-1的算法如下:
![image](https://github.com/user-attachments/assets/4105f7c9-0b5d-43ec-a7bb-769a4af60003)
对于规模为B\*H\*N\*d的输入, 本实验采用的实现方式是, 将线程网格大小设置为B\*H\*1, 因此每个线程块处理一个N\*d矩阵; 将线程块大小设置为Bc\*Br\*1, 并在启动核函数之前采用动态分配的方式提前分配足够的共享内存. 由于线程块三个维度的乘积不能超过1024, 因此Bc\*Br不能超过1024.

在核函数内部, 首先获取Qi, Ki, Vi等变量的首地址, 注意按32字节对齐. 然后开启两层循环: 外层循环用j遍历Tc, 对于每个j, 将对应分块的K和V从全局内存读取到线程块共享内存上; 内层循环用i遍历Tr, 对于每个i, 将对应分块的Q和O从全局内存读取到线程块共享内存上, 并按照上图进行运算. 对于每一个i, 运算完毕后都要将对应分块的O写回全局内存.

#### 2.1.2 Optimize
对于flash attention-1的初步实现, 本实验主要采取以下优化方式:
1. 块内共享内存首地址按16字节对齐, 方便通过float4读取.
2. 加载QKVO到共享内存的时候, 使用float4结构读取, 这样一次可以将4个float读取到寄存器, 减少全局内存IO次数. 将分块的O写回全局内存的时候同理.
3. 计算Q matmul K的时候, 也是用float4从共享内存读取和计算.
4. 计算行和与行最大值的时候使用warp归约运算. 注意这要求Bc不超过32.
5. 使用tensor core加速Sij = Qi * Kj^T的过程

### 2.2 flash attention-2
实现代码在flash_attention2/flash_attention2.cu中.

flash attention-2的算法如下
![image](https://github.com/user-attachments/assets/b140b522-20ed-4336-b2a8-bd8e5926670a)
对于规模为B\*H\*N\*d的输入, 将线程网格大小设置为B\*H\*Tr, 线程块大小同上, 采用的优化方法也相同, 区别在于flash attention-2的核函数内部只有一层循环, 即使用j遍历Tc. 而对Tr的遍历分配到了不同的线程块, 每个线程块只处理一个i(0 <= i < Tr).

## 3. Self Compare
本章对比了flash attention-1的初步实现和优化后的实现在运行时间上的差异, 并以pytorch的标准attention作为基线, 即以(F.softmax(Q @ K.transpose(-2,-1), dim=3)) @ V这一运算过程消耗的时间作为基线. 可以运行flash_attn1_self_compare.py得到对比结果, 该脚本会在当前目录下创建flash_attn1_self_compare_时间戳这一子目录, 并写入不同数据规模下的比较结果, 例如:
![performance_B8_H8_d32](https://github.com/user-attachments/assets/9262b1c1-f571-4e8e-8295-b1eda88490a5)


## 4. Compare flash attention-1 and flash attention-2 
本章对比了flash attention-12的优化后的实现在运行时间上的差异, 并以pytorch的标准attention作为基线, 即以(F.softmax(Q @ K.transpose(-2,-1), dim=3)) @ V这一运算过程消耗的时间作为基线. 可以运行compare_with_pytorch.py得到对比结果, 该脚本会在当前目录下创建flash_attn_1_2_compare_时间戳这一子目录, 并写入不同数据规模下的对比结果, 例如:
![performance_B8_H8_d8](https://github.com/user-attachments/assets/c0ab8050-06cb-4eb1-b8cf-abb1ffb77759)

## 5. Results
在flash attention-1的优化方面, 随着d的增大, 章节2.1.2中使用的优化措施带来的运行时间的减少越明显, 最大可以减少到约1/3.

在flash attention-1和flash attention-2的对比方面, 2由于将核函数中的循环从两层减少到一层, 运行时间大大减少. 具体地, 当BHd很小时, 2的运行时间比1可以减少近百倍. 但是随着BHd的增大, 2和1的运行时间差距在减小.
与pytorch的标准attention做对比, 当d小于16的时候, flash attention-2的运行时间可以低于pytorch; 当d大于等于16的时候, 2的运行时间会反超pytorch且差距随着d的增大而增大



# 如何进一步提升性能
1.发掘更多可以运用cuda硬件加速的计算过程, 如矩阵逐元素乘法等
