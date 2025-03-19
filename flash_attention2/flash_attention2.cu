#include <cuda_runtime.h>
#ifdef __INTELLISENSE__
#define __CUDACC__
#define __CUDA_ARCH__ 750
#endif
#include <mma.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#define SPEED_FACTOR (1000000000000)

using namespace nvcuda;

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

/* 初始化共享内存 */
__device__ void flush_shared_memory(half* shared_Qi, half* shared_Ki, float* shared_Sij, int d, int Br, int Bc)
{
    int local_col_id = threadIdx.x;
    int local_row_id = threadIdx.y;
    shared_Sij[local_row_id * Bc + local_col_id] = 0;
    for (int col = local_col_id; col < d; col += Bc)
    {
        shared_Qi[local_row_id * d + col] = __float2half(0);
    }
    for (int col = local_row_id; col < d; col += Br)
    {
        shared_Ki[local_col_id * d + col] = __float2half(0);
    }
}

const int WMMA_M = 16; // A矩阵行数
const int WMMA_N = 16; // B矩阵列数
const int WMMA_K = 16; // A,B乘法展开维度
/*
计算Sij = Qi * Ki子矩阵(16 * 16)
维度d必须是16的倍数
*/
__device__ void compute_sij_wmma(half* shared_Qi, half* shared_Ki, float* shared_Sij, int d, int Br, int Bc)
{
    // 获取当前warp id和lane id
    int warp_id = threadIdx.y; // 一个warp计算一个16*16的子块
    if (warp_id >= 4) // 只需要4个warp计算32*32的Sij
    {
        return;
    }
    int warp_id_y = warp_id / 2; // 负责Sij的上下, 0=上, 1=下
    int warp_id_x = warp_id % 2; // 负责Sij的左右, 0=左, 1=右
    int lane_id = threadIdx.x; // warp内的id

    // 定义WMMA片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_C;
    wmma::fill_fragment(frag_C, 0.0f); // 初始化结果

    // 遍历 d/16 维度块计算Sij
    for (int k = 0; k < d; k += WMMA_K)
    {
        // 载入Qi和Ki片段
        wmma::load_matrix_sync(frag_A, shared_Qi + warp_id_y * d * WMMA_M + k, d); // 按行存储
        wmma::load_matrix_sync(frag_B, shared_Ki + warp_id_x * d * WMMA_N + k, d); // 按列存储
        //__syncwarp();
        // 计算Sij = Qi * Ki
        wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
    }
    // 计算写回Sij的行列偏移
    int Sij_offset_x = warp_id_x * WMMA_N; // 0 or 16
    int Sij_offset_y = warp_id_y * WMMA_M; // 0 or 16
    // 将结果写回共享内存
    wmma::store_matrix_sync(shared_Sij + Sij_offset_y * Bc + Sij_offset_x, frag_C, Bc, wmma::mem_row_major);
}

/*
使用simd放缩shared_Oi的值
*/
__device__
void scale_shared_oi(float* shared_Oi, int local_row_id, int local_col_id, int d, int padding_d,
                    int actual_Bc, int float4_num, float scale_factor)
{
    for (int col = local_col_id; col < float4_num; col += actual_Bc)    
    {
        float4 Oi = *(float4*)(&shared_Oi[local_row_id * padding_d + col*4]);
        Oi.x *= scale_factor;
        Oi.y *= scale_factor;
        Oi.z *= scale_factor;
        Oi.w *= scale_factor;
        *(float4*)(&shared_Oi[local_row_id * padding_d + col*4]) = Oi;
    }
    // 处理不整除的部分
    for (int col = float4_num * 4; col < d; col += actual_Bc)
    {
        float tmp = shared_Oi[local_row_id * padding_d + col];
        tmp *= scale_factor;
        shared_Oi[local_row_id * padding_d + col] = tmp;
    }
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
    // 填充d为16的倍数
    int padding_d = (d + 15) / 16 * 16;

    /* 划分共享内存 */
    extern __shared__ __align__(16) float shared_mem[];
    // 划分Qi
    half* shared_Qi = (half*)shared_mem;                                      // Br * d
    // 划分Ki
    half* shared_Ki = shared_Qi + ((Br * padding_d + 15) / 16 * 16);          // Bc * d
    // 使shared_Ki向上对齐到float边界, 保证后续float*访问安全
    shared_Ki = (half*)(((uintptr_t)shared_Ki + 7) & ~7);
    // 现在shared_Ki是half*类型, 且对齐到了4字节边界
    // 划分Vi
    float* shared_Vi = (float*)(shared_Ki + ((Bc * padding_d + 31) / 32 * 32));           // Bc * d
    // 划分Oi
    float* shared_Oi = shared_Vi + ((Bc * padding_d + 31) / 32 * 32);           // Br * d
    // 划分V^T
    float* shared_Vi_T = shared_Oi + ((Br * padding_d + 31) / 32 * 32); // Bc * d
    // 划分mi
    float* shared_mi = shared_Vi_T + ((Bc * padding_d + 31) / 32 * 32);           // Br
    // 划分li
    float* shared_li = shared_mi + ((Br * 1 + 31) / 32 * 32);           // Br
    // 划分Sij
    float* shared_Sij = shared_li + ((Br * 1 + 31) / 32 * 32);          // 存储当前块负责的局部QK^t, Br * Bc

    flush_shared_memory(shared_Qi, shared_Ki, shared_Sij, padding_d, Br, Bc);

    float tmp = 0;
    half half_tmp = __float2half(tmp);
    /* 将padding部分置为0 */
    for (int col = d; col < padding_d; col += Br)
    {
        write_2d(shared_Vi, local_col_id, col, padding_d, tmp);
        shared_Vi_T[col * Bc + local_col_id] = 0;
    }
    for (int col = d; col < padding_d; col += Bc)
    {
        write_2d(shared_Oi, local_row_id, col, padding_d, tmp);
    }

    // 向量化加载时的末尾索引
    int float4_end = d / 4 * 4;
    int float4_num = d / 4;

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
        half2 half_tmp_1, half_tmp_2;
        half_tmp_1 = __floats2half2_rn(tmp.x, tmp.y);
        half_tmp_2 = __floats2half2_rn(tmp.z, tmp.w);
        *(half2*)(&shared_Qi[local_row_id * padding_d + col*4]) = half_tmp_1;
        *(half2*)(&shared_Qi[local_row_id * padding_d + col*4 + 2]) = half_tmp_2;

        tmp = ((float4*)O)[global_offset];
        *(float4*)(&shared_Oi[local_row_id * padding_d + col*4]) = tmp;
    }
    // 逐元素加载剩余部分
    for (int col = float4_num*4; col < d; col += Bc)
    {
        // 加载Q
        tmp = read_4d(Q, batch_id, head_id, global_row_id, col, H, N, d);
        shared_Qi[local_row_id * padding_d + col] = __float2half(tmp);
        // 加载O
        tmp = read_4d(O, batch_id, head_id, global_row_id, col, H, N, d);
        shared_Oi[local_row_id * padding_d + col] = tmp;
    }
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
            half2 half_tmp_1, half_tmp_2;
            half_tmp_1 = __floats2half2_rn(tmp.x, tmp.y);
            half_tmp_2 = __floats2half2_rn(tmp.z, tmp.w);
            // 加载K
            *(half2*)(&shared_Ki[local_col_id * padding_d + col*4]) = half_tmp_1;
            *(half2*)(&shared_Ki[local_col_id * padding_d + col*4 + 2]) = half_tmp_2;

            // 加载V
            tmp = ((float4*)V)[global_offset];
            *(float4*)(&shared_Vi[local_col_id * padding_d + col*4]) = tmp;
            // 转置v
            shared_Vi_T[col*4 * Bc + local_col_id] = tmp.x;
            shared_Vi_T[(col*4+1) * Bc + local_col_id] = tmp.y;
            shared_Vi_T[(col*4+2) * Bc + local_col_id] = tmp.z;
            shared_Vi_T[(col*4+3) * Bc + local_col_id] = tmp.w;
        }
        // 逐元素加载剩余部分
        for (int col = float4_num*4; col < d; col += actual_Br)
        {
            // 加载K
            tmp = read_4d(K, batch_id, head_id, global_col_id, col, H, N, d);
            shared_Ki[local_col_id * padding_d + col] = __float2half(tmp);
            // 加载V
            tmp = read_4d(V, batch_id, head_id, global_col_id, col, H, N, d);
            shared_Vi[local_col_id * padding_d + col] = tmp;
            // 转置v
            shared_Vi_T[col * Bc + local_col_id] = tmp;
        }
        __syncthreads();
        /* 开始计算 */
        // 计算Sij = Qi * Ki^t
        /* wmma */
        compute_sij_wmma(shared_Qi, shared_Ki, shared_Sij, padding_d, Br, Bc);
        __syncthreads();

        /* 计算mi_j=max(mi_j-1, rowmax(sij)) */
        // 这里不能判断mask, 否则warp会卡死
        mi_j_minus_1 = read_2d(shared_mi, local_row_id, 0, 1);
        tmp = shared_Sij[local_row_id * Bc + local_col_id];
        float row_max = warpReduceMax(tmp, actual_Bc);
        if (local_col_id % warpSize == 0)
        {
            float mi_j = fmaxf(mi_j_minus_1, row_max);
            write_2d(shared_mi, local_row_id, 0, 1, mi_j);
        }
        __syncwarp();

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
        __syncwarp();

        /* 计算Oi_j = diag(exp(mi_j-1 - mi_j)) * Oi_j-1 + Pi * Vj */
        // 首先放缩Oi
        scale_shared_oi(shared_Oi, local_row_id, local_col_id, d, padding_d, actual_Bc, float4_num, exp_mi_minus);


        for (int col = local_col_id; col < d; col += actual_Bc)
        {
            float Oi_j_minus_1 = read_2d(shared_Oi, local_row_id, col, d);
            //Oi_j_minus_1 *= exp_mi_minus;
            for (int k = 0; k < actual_Bc; k++)
            {
                Oi_j_minus_1 += (read_2d(shared_Sij, local_row_id, k, Bc) * read_2d(shared_Vi, k, col, d));
            }
            write_2d(shared_Oi, local_row_id, col, padding_d, Oi_j_minus_1);
        }
        // for (int col = local_col_id; col < d; col += actual_Bc)
        // {
        //     float Oi_j_minus_1 = read_2d(shared_Oi, local_row_id, col, d);
        //     Oi_j_minus_1 *= exp_mi_minus;
        //     for (int k = 0; k < actual_Bc; k++)
        //     {
        //         Oi_j_minus_1 += (read_2d(shared_Sij, local_row_id, k, Bc) * read_2d(shared_Vi, k, col, d));
        //     }
        //     write_2d(shared_Oi, local_row_id, col, padding_d, Oi_j_minus_1);
        // }
        __syncthreads();
    }
    //__syncthreads();
    /* 计算Oi = Oi / li */
    float li_tc = read_2d(shared_li, local_row_id, 0, 1);
    // 向量化
    for (int col = local_col_id; col < float4_num; col += Bc)
    {
        int global_offset = batch_id * (H * N * float4_num) + head_id * (N * float4_num) + global_row_id * float4_num + col;
        float4 Oi = *(float4*)(&shared_Oi[local_row_id * padding_d + col*4]);
        Oi.x /= li_tc; Oi.y /= li_tc; Oi.z /= li_tc; Oi.w /= li_tc;
        // 写回全局内存
        ((float4*)O)[global_offset] = Oi;
    }

    for (int col = float4_num; col < d; col += Bc)
    {
        float Oi = shared_Oi[local_row_id * padding_d + col];
        Oi /= li_tc;
        // 写回全局内存
        write_4d(O, batch_id, head_id, global_row_id, col, H, N, d, Oi);
    }
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
    int Br = 32, Bc = 32;

    std::cout << "Br is [" << Br << "], Bc is [" << Bc << "]" << std::endl;
    // 判断共享内存是否超出限制
    int padding_d = (d + 15) / 16 * 16;
    int shared_mem_size = 
                ((Br * padding_d + 31) / 32 * 32) + //shared_Qi
                ((Bc * padding_d + 31) / 32 * 32) + // shared_Ki
                ((Bc * padding_d + 31) / 32 * 32)*2 + // shared_Vi
                ((Br * padding_d + 31) / 32 * 32) + // shared_Oi
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