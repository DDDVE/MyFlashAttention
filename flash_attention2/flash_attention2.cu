#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#define SPEED_FACTOR (1000000000000)

float get_flops_speed(int B, int H, int N, int d, float time, bool causal=false, int mode=0)
{
    printf("Get time [%f]s.\n", time);
    /* 0=fwd 1=bwd 2=both */
    int flops = 4 * B * N * N * H * d;
    if (causal)
    {
        flops /= 2;
    }
    if (mode == 1)
    {
        flops *= 2.5;
    }
    if (mode == 2)
    {
        flops *= 3.5;
    }
    return flops / (time * SPEED_FACTOR);
}

__global__ void initToInfinity_float(float* m, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
    {
        return;
    }
    m[idx] = -INFINITY;
}

/* 读取二维数据 */
__device__ __forceinline__ float read_2d(const float* array, int row, int col, int cols_per_row)
{
    return array[row * cols_per_row + col];
}

/* 写入二维数据 */
__device__ __forceinline__ void write_2d(float* array, int row, int col, int cols_per_row, float& data)
{
    array[row * cols_per_row + col] = data;
}

/* 读取四维数据 */
__device__ __forceinline__ float read_4d(const float* array, int batch, int head, int row, int col, int heads_per_batch, int rows_per_head, int cols_per_row)
{
    return array[batch * (heads_per_batch * rows_per_head * cols_per_row) + 
        head * (rows_per_head * cols_per_row) + 
        row * cols_per_row + 
        col];
}

/* 写入四维数据 */
__device__ __forceinline__ void write_4d(float* array, int batch, int head, int row, int col, 
int heads_per_batch, int rows_per_head, int cols_per_row, float& data)
{
    array[batch * (heads_per_batch * rows_per_head * cols_per_row) + 
        head * (rows_per_head * cols_per_row) + 
        row * cols_per_row + 
        col] = data;
}

/* warp内部归约求和 */
__inline__ __device__ float warpReduceSum(float val, int Bc) {
    for (int offset = Bc / 2; offset > 0; offset /= 2) {  
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // 线程 0 存储最终和
}

/* warp内部归约求最大值 */
__device__ float warpReduceMax(float val, int Bc) {
    // 循环进行归约，offset 每次减半，直到 offset=0
    for (int offset = Bc / 2; offset > 0; offset /= 2) {
        // 读取当前线程向右偏移 offset 线程的值
        float tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, tmp);  // 取两个线程的最大值
    }
    return val;
}

__global__ void flash_attention2(float* __restrict__ Q, float* __restrict__ K, float* __restrict__ V, 
                                float* __restrict__ O, float* __restrict__ l, float* __restrict__ m,
                            int B, int H, int N, int d, int Br, int Bc, int Tr, int Tc)
{
    /* 获取当前线程位置 */
    // 得出当前线程位于哪个B和H
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int i = blockIdx.z;
    // 线程对应子矩阵中的行号列号
    int local_col_id = threadIdx.x;
    int local_row_id = threadIdx.y;
    /* 划分共享内存 */
    extern __shared__ __align__(16) float shared_mem[];
    // 划分Qi
    float* shared_Qi = shared_mem;                                      // Br * d
    // 划分Ki
    float* shared_Ki = shared_Qi + ((Br * d + 31) / 32 * 32);           // Bc * d
    // 划分Vi
    float* shared_Vi = shared_Ki + ((Bc * d + 31) / 32 * 32);           // Bc * d
    // 划分Oi
    float* shared_Oi = shared_Vi + ((Bc * d + 31) / 32 * 32);           // Br * d
    // 划分mi
    float* shared_mi = shared_Oi + ((Br * d + 31) / 32 * 32);           // Br
    // 划分li
    float* shared_li = shared_mi + ((Br * 1 + 31) / 32 * 32);           // Br
    // 划分Sij
    float* shared_Sij = shared_li + ((Br * 1 + 31) / 32 * 32);          // 存储当前块负责的局部QK^t, Br * Bc

    // 向量化加载时的末尾索引
    int float4_end = d / 4 * 4;
    int float4_num = d / 4;

    float tmp = 0;
    float mi_j_minus_1;
    float exp_mi_minus;
    float exp_m;

    // 全局行坐标
    int global_row_id = i * Br + local_row_id;
    // 判断是否掩盖当前线程
    bool mask = (global_row_id >= N);
    if (mask)
    {
        return;
    }
    // 实际的Br
    int actual_Br = (i == Tr - 1) ? (N % Br == 0 ? Br : N % Br) : Br;
    // 加载Q O
    for (int col = local_col_id; col < float4_num; col += Bc)
    {
        int global_offset = batch_id * (H * N * float4_num) + head_id * (N * float4_num) + global_row_id * float4_num + col;
        float4 tmp = ((float4*)Q)[global_offset];
        *(float4*)(&shared_Qi[local_row_id * d + col*4]) = tmp;

        tmp = ((float4*)O)[global_offset];
        *(float4*)(&shared_Oi[local_row_id * d + col*4]) = tmp;
    }
    // 逐元素加载剩余部分
    // for (int col = float4_num*4; col < d; col += Bc)
    // {
    //     // 加载Q
    //     tmp = read_4d(Q, batch_id, head_id, global_row_id, col, H, N, d);
    //     //printf("Get Q[%d][%d] is [%f]\n", local_row_id, col, tmp);
    //     write_2d(shared_Qi, local_row_id, col, d, tmp);
    //     // 加载O
    //     tmp = read_4d(O, batch_id, head_id, global_row_id, col, H, N, d);
    //     //printf("Get O[%d][%d] is [%f]\n", local_row_id, col, tmp);
    //     write_2d(shared_Oi, local_row_id, col, d, tmp);
    // }
    // 加载li mi
    if (local_col_id == 0)
    {
        tmp = read_4d(l, batch_id, head_id, global_row_id, 0, H, N, 1);
        write_2d(shared_li, local_row_id, 0, 1, tmp);
        tmp = read_4d(m, batch_id, head_id, global_row_id, 0, H, N, 1);
        write_2d(shared_mi, local_row_id, 0, 1, tmp);
    }

    for (int j = 0; j < Tc; j++)
    {
        //__syncthreads();
        // 全局列坐标
        int global_col_id = j * Bc + local_col_id;
        bool mask = global_col_id >= N;
        if (mask)
        {
            continue;
        }
        // 实际的Bc
        int actual_Bc = (j == Tc - 1) ? (N % Bc == 0 ? Bc : N % Bc) : Bc;
        /* 加载KV */
        // 向量化加载
        for (int col = local_row_id; col < float4_num; col += actual_Br)
        {
            int global_offset = batch_id * (H * N * float4_num) + head_id * (N * float4_num) + global_col_id * float4_num + col;
            float4 tmp = ((float4*)K)[global_offset];
            // 加载K
            *(float4*)(&shared_Ki[local_col_id * d + col*4]) = tmp;
            // 加载V
            tmp = ((float4*)V)[global_offset];
            *(float4*)(&shared_Vi[local_col_id * d + col*4]) = tmp;
        }
        // 逐元素加载剩余部分
        // for (int col = float4_num*4; col < d; col += actual_Br)
        // {
        //     // 加载K
        //     tmp = read_4d(K, batch_id, head_id, global_col_id, col, H, N, d);
        //     //printf("Get K[%d][%d] is [%f]\n", local_col_id, col, tmp);
        //     write_2d(shared_Ki, local_col_id, col, d, tmp);
        //     // 加载V
        //     tmp = read_4d(V, batch_id, head_id, global_col_id, col, H, N, d);
        //     //printf("Get V[%d][%d] is [%f]\n", local_col_id, col, tmp);
        //     write_2d(shared_Vi, local_col_id, col, d, tmp);
        // }
        __syncthreads();
        /* 开始计算 */
        // 计算Sij = Qi * Ki^t
        tmp = 0;
        // 先处理能被4整除的部分
        for (int col = 0; col < float4_end; col += 4)
        {
            float4 q_val = *((float4*)&shared_Qi[local_row_id * d + col]);
            float4 k_val = *((float4*)&shared_Ki[local_col_id * d + col]);
            tmp += q_val.x * k_val.x + q_val.y * k_val.y + q_val.z * k_val.z + q_val.w * k_val.w;
        }
        // 处理剩余不满4的部分
        // for (int col = float4_end; col < d; col++)
        // {
        //     tmp += (read_2d(shared_Qi, local_row_id, col, d) * read_2d(shared_Ki, local_col_id, col, d));
        // }
        // 将结果写到Sij的[local_row_id, local_col_id]处, 无需原子因为一个线程只处理Sij的一个位置
        write_2d(shared_Sij, local_row_id, local_col_id, Bc, tmp);
        __syncthreads();

        /* 计算mi_j=max(mi_j-1, rowmax(sij)) */
        // 这里不能判断mask, 否则warp会卡死
        mi_j_minus_1 = read_2d(shared_mi, local_row_id, 0, 1);
        float row_max = warpReduceMax(tmp, actual_Bc);
        if (local_col_id % warpSize == 0)
        {
            float mi_j = fmaxf(mi_j_minus_1, row_max);
            write_2d(shared_mi, local_row_id, 0, 1, mi_j);
        }
        __syncthreads();

        float mi_j = read_2d(shared_mi, local_row_id, 0, 1);

        /* 计算Pi_j = exp(Si_j - mi_j)*/
        tmp = expf(tmp - mi_j);
        write_2d(shared_Sij, local_row_id, local_col_id, Bc, tmp);

        /* 计算li_j = exp(mi_j-1 - mi_j) * li_j-1 + rowsum(Pi) */
        float li_j_minus_1 = read_2d(shared_li, local_row_id, 0, 1);
        exp_mi_minus = expf(mi_j_minus_1 - mi_j);
        exp_m = exp_mi_minus * li_j_minus_1;
        float rowsum = warpReduceSum(tmp, actual_Bc);
        if (local_col_id % warpSize == 0)
        {
            float li_j = exp_m + rowsum;
            write_2d(shared_li, local_row_id, 0, 1, li_j);
        }
        __syncthreads();

        /* 计算Oi_j = diag(exp(mi_j-1 - mi_j)) * Oi_j-1 + Pi * Vj */
        for (int col = local_col_id; col < d; col += actual_Bc)
        {
            float Oi_j_minus_1 = read_2d(shared_Oi, local_row_id, col, d);
            Oi_j_minus_1 *= exp_mi_minus;
            for (int k = 0; k < actual_Bc; k++)
            {
                Oi_j_minus_1 += (read_2d(shared_Sij, local_row_id, k, Bc) * read_2d(shared_Vi, k, col, d));
            }
            write_2d(shared_Oi, local_row_id, col, d, Oi_j_minus_1);
        }
        __syncthreads();
    }
    //__syncthreads();
    /* 计算Oi = Oi / li */
    float li_tc = read_2d(shared_li, local_row_id, 0, 1);
    // 向量化
    for (int col = local_col_id; col < float4_num; col += Bc)
    {
        int global_offset = batch_id * (H * N * float4_num) + head_id * (N * float4_num) + global_row_id * float4_num + col;
        float4 Oi = *(float4*)(&shared_Oi[local_row_id * d + col*4]);
        Oi.x /= li_tc; Oi.y /= li_tc; Oi.z /= li_tc; Oi.w /= li_tc;
            
        //write_2d(shared_Oi, local_row_id, col, d, Oi);
        // 写回全局内存
        ((float4*)O)[global_offset] = Oi;
    }

    // for (int col = local_col_id; col < d; col += Bc)
    // {
    //     float Oi = read_2d(shared_Oi, local_row_id, col, d);
    //     Oi /= li_tc;
    //     write_2d(shared_Oi, local_row_id, col, d, Oi);
    //     // 写回全局内存
    //     write_4d(O, batch_id, head_id, global_row_id, col, H, N, d, Oi);
    // }
}

/* 主机函数 */
float flashAttentionHost_2(float* Q, float* K, float* V, float* O, int B, int H, int N, int d)
{
    printf("global K is [%p]\n", K);
    printf("global V is [%p]\n", V);
    /* 把共享内存上限设置为64KB */
    cudaFuncSetAttribute(
        flash_attention2,                      // 核函数名
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        65536                                 // 最大 64 KB
    );
    /* 获取SRAM大小M */
    cudaDeviceProp device_prop;
    int device_id;
    // 获取设备id
    cudaGetDevice(&device_id);
    std::cout << "Get device id = [" << device_id << "]" << std::endl;
    // 获取当前设备属性
    cudaGetDeviceProperties(&device_prop, device_id);
    // 获取共享内存大小
    int M = device_prop.sharedMemPerBlock;
    std::cout << "Get shared mem size = [" << M << "]" << std::endl;
    /* 计算块大小 */
    int Br = 16, Bc = 8;
    // 调整 Br 和 Bc 的计算，确保共享内存不超标
    // int Bc = std::min(N, M / (16 * d));
    // int Br = std::min(d, M / (32 * Bc));

    // 如果计算出的 Br 或 Bc 太小，可以再进一步限制
    // Br = std::max(Br, 32); // Br 至少为 32
    // Bc = std::max(Bc, 32); // Bc 至少为 32

    std::cout << "Br is [" << Br << "], Bc is [" << Bc << "]" << std::endl;
    // 判断共享内存是否超出限制
    int shared_mem_size = 
                ((Br * d + 31) / 32 * 32) + //shared_Qi
                ((Bc * d + 31) / 32 * 32) + // shared_Ki
                ((Bc * d + 31) / 32 * 32) + // shared_Vi
                ((Br * d + 31) / 32 * 32) + // shared_Oi
                ((Br * 1 + 31) / 32 * 32) + // shared_mi
                ((Br * 1 + 31) / 32 * 32) + // shared_li
                ((Br * Bc + 31) / 32 * 32); // shared_Sij
    shared_mem_size *= sizeof(float);
    std::cerr << "Sahred memory need [" << shared_mem_size << "], have [" << M << "]" << std::endl;
    if (shared_mem_size > M)
    {
        return -1;
    }

    /* 初始化事件 */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* 分配设备空间 */
    // 分配l l需要每个BH都独立, 所以需要分配B*H个
    float* device_l;
    cudaMalloc(&device_l, B * H * N * sizeof(float));
    cudaMemset(device_l, 0, B * H * N * sizeof(float));
    // 分配m m需要每个BH都独立, 所以需要分配B*H个
    float* device_m;
    cudaMalloc(&device_m, B * H * N * sizeof(float));
    // 使用核函数设置为负无穷
    int thread_per_block = 1024;
    int block_num = (B * H * N + thread_per_block - 1) / thread_per_block;
    initToInfinity_float<<<block_num, thread_per_block>>>(device_m, B * H * N);
    cudaDeviceSynchronize();

    /* 启动核函数 */
    // 计算有几个行块几个列块
    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;
    std::cout << "Tr = [" << Tr << "], Tc = [" << Tc << "]" << std::endl;
    // 设置线程块和线程网格的大小
    dim3 block_dim(Bc, Br, 1);
    dim3 grid_dim(B  , H, Tr);
    // 开始计时
    cudaEventRecord(start, 0);
    // 启动核函数 指定共享内存大小
    flash_attention2<<<grid_dim, block_dim, shared_mem_size>>>(Q, K, V, O, device_l, device_m, B, H, N, d, Br, Bc, Tr, Tc);
    // 停止计时
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // 检查是否有错误
    cudaError_t err = cudaGetLastError();
    float miliseconds = -1;
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        // 计算时间
        cudaEventElapsedTime(&miliseconds, start, stop);
        float flops_speed = get_flops_speed(B, H, N, d, miliseconds / 1000);
        std::cout << "Kernel exec time [" << miliseconds << "] ms. Speed [" << flops_speed << "] TFLops/s" <<  std::endl;
    }
    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 释放资源
    cudaFree(device_l);
    cudaFree(device_m);

    return miliseconds;
}