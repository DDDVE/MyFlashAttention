#include <cuda_runtime.h>
#ifdef __INTELLISENSE__
#define __CUDACC__
#define __CUDA_ARCH__ 750
#endif
#include <mma.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

using namespace nvcuda;

#define SPEED_FACTOR (1000000000000)

float get_flops_speed(int B, int H, int N, int d, float time, bool causal=false, int mode=0)
{
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
    return flops / time / SPEED_FACTOR;
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
template <typename T>
__device__ __forceinline__ T read_2d(const T* array, int row, int col, int cols_per_row)
{
    return array[row * cols_per_row + col];
}

/* 写入二维数据 */
template <typename T>
__device__ __forceinline__ void write_2d(T* array, int row, int col, int cols_per_row, T& data)
{
    array[row * cols_per_row + col] = data;
}

/* 读取四维数据 */
template <typename T>
__device__ __forceinline__ T read_4d(const T* array, int batch, int head, int row, int col, int heads_per_batch, int rows_per_head, int cols_per_row)
{
    return array[batch * (heads_per_batch * rows_per_head * cols_per_row) + 
        head * (rows_per_head * cols_per_row) + 
        row * cols_per_row + 
        col];
}

/* 写入四维数据 */
template <typename T>
__device__ __forceinline__ void write_4d(T* array, int batch, int head, int row, int col, 
int heads_per_batch, int rows_per_head, int cols_per_row, T& data)
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
    // int lane_id = threadIdx.x; // warp内的id

    // 定义WMMA片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_C; // TODO Sij改为half
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

/*
将shared_oi与shared_Oi_temp求和, 并把结果写回shared_Oi
*/
__device__
void add_shared_oi_temp(float* shared_Oi, float* shared_Oi_temp, int local_row_id, int local_col_id, int d, int padding_d,
                        int float4_num, int actual_Bc)
{
    for (int col = local_col_id; col < float4_num; col += actual_Bc)    
    {
        float4* tmp = (float4*)(&shared_Oi[local_row_id * padding_d + col*4]);
        float4* tmp_1 = (float4*)(&shared_Oi_temp[local_row_id * padding_d + col*4]);
        tmp->x += tmp_1->x;
        tmp->y += tmp_1->y;
        tmp->z += tmp_1->z;
        tmp->w += tmp_1->w;
    }
    // 处理不整除的部分
    for (int col = float4_num * 4 + local_col_id; col < d; col += actual_Bc)
    {
        float* tmp = &shared_Oi[local_row_id * padding_d + col];
        float* tmp_1 = &shared_Oi_temp[local_row_id * padding_d + col];
        *tmp += (*tmp_1);
    }
}


/*
计算Sij * Vj
*/
__device__
void wmma_sij_vj(half* shared_Sij, half* shared_Vj, float* shared_Oi_temp, int d, int Br, int Bc, int need_warp_num)
{
    // 获取当前warp_id和lane_id
    int warp_id = threadIdx.y;
    // 固定用两个warp
    if (warp_id >= 2)
    {
        return;
    }
    int warp_id_x, warp_id_y; // x表示左右, y表示上下
    // warp_id_x = (need_warp_num == 2) ? 0 : (warp_id % 2);
    // warp_id_y = (need_warp_num == 2) ? (warp_id % 2) : (warp_id / 2);
    warp_id_y = warp_id;
    
    // 定义wmma片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_C;
    wmma::fill_fragment(frag_C, 0.0f); // 初始化结果

    
    // 遍历Vj的d/16, 先遍历Vj分块后的第一列
    for (int k = 0; k < d; k += WMMA_K)
    {
        // 载入warp自己要用的Sij, 先载入Sij的第一列块
        wmma::load_matrix_sync(frag_A, shared_Sij + warp_id_y * Bc * WMMA_M, Bc);
        // 载入Vj 第一列块
        wmma::load_matrix_sync(frag_B, shared_Vj + k * Bc, Bc);
        // 计算
        wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

        // 判断是否需要载入Sij的第二列块
        if (need_warp_num == 4)
        {
            // 加载Sij第二列块
            wmma::load_matrix_sync(frag_A, shared_Sij + warp_id_y * Bc * WMMA_M + WMMA_N, Bc);
            // 载入Vj第二列块
            wmma::load_matrix_sync(frag_B, shared_Vj + k * Bc + WMMA_N, Bc);
            // 计算
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
        // 写回共享内存
        wmma::store_matrix_sync(shared_Oi_temp + warp_id_y * WMMA_M * d + k * WMMA_K, frag_C, d, wmma::mem_row_major);
    }
}

/* flash attention核函数, 每个线程块处理一个Br Bc分块 */
__global__ void flash_attention_best(float* __restrict__ Q, float* __restrict__ K, float* __restrict__ V, 
                                float* __restrict__ O, float* __restrict__ l, float* __restrict__ m,
                            int B, int H, int N, int d, int Br, int Bc, int Tr, int Tc)
{
    /* 获取当前线程位置 */
    // 得出当前线程位于哪个B和H
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
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
    half* shared_Ki = shared_Qi + ((Br * padding_d + 15) / 16 * 16);           // Bc * d
    // 使shared_Ki向上对齐到float边界, 保证后续float*访问安全
    shared_Ki = (half*)(((uintptr_t)shared_Ki + 7) & ~7);
    // 现在shared_Ki是half*类型, 且对齐到了4字节边界
    // 划分Vi
    float* shared_Vi = (float*)(shared_Ki + ((Bc * padding_d + 31) / 32 * 32));           // Bc * d
    // 划分Oi
    float* shared_Oi = shared_Vi + ((Bc * padding_d + 31) / 32 * 32);           // Br * d
    // 划分Vi^T
    float* shared_Vi_T = shared_Oi + ((Br * padding_d + 31) / 32 * 32); // Bc * d
    // 划分mi
    float* shared_mi = shared_Vi_T + ((Bc * padding_d + 31) / 32 * 32);           // Br
    // 划分li
    float* shared_li = shared_mi + ((Br * 1 + 31) / 32 * 32);           // Br
    // 划分Sij
    float* shared_Sij = shared_li + ((Br * 1 + 31) / 32 * 32);          // 存储当前块负责的局部QK^t, Br * Bc
    // 划分mij
    float* shared_mij = shared_Sij + ((Br * Bc + 31) / 32 * 32);        // 存储每一次Sij的行最大值, Br
    // 划分lij
    float* shared_lij = shared_mij + ((Br * 1 + 31) / 32 * 32);        // Br
    // 划分minew
    float* shared_mi_new = shared_lij + ((Br * 1 + 31) / 32 * 32);      // Br
    // 划分linew
    float* shared_li_new = shared_mi_new + ((Br * 1 + 31) / 32 * 32);   // Br

    flush_shared_memory(shared_Qi, shared_Ki, shared_Sij, padding_d, Br, Bc);

    float tmp = 0;
    half half_tmp = __float2half(tmp);
    /* 将padding部分置为0 */
    for (int col = d; col < padding_d; col += Br)
    {
        //write_2d(shared_Ki, local_col_id, col, padding_d, half_tmp);
        write_2d(shared_Vi, local_col_id, col, padding_d, tmp);
        shared_Vi_T[col * Bc + local_col_id] = tmp;
    }
    for (int col = d; col < padding_d; col += Bc)
    {
        //write_2d(shared_Qi, local_row_id, col, padding_d, half_tmp);
        write_2d(shared_Oi, local_row_id, col, padding_d, tmp);
    }

    // 向量化加载时的末尾索引
    int float4_end = d / 4 * 4;
    int float4_num = d / 4;

    
    /* 遍历所有分块 */ 
    for (int j = 0; j < Tc; j++)
    {
        // 全局列坐标
        int global_col_id = j * Bc + local_col_id;
        bool mask = global_col_id >= N;
        if (mask)
        {
            return;
        }
        // 实际的Bc
        int actual_Bc = (j == Tc - 1) ? (N % Bc == 0 ? Bc : N % Bc) : Bc;
        // 加载KV
        // 向量化加载
        for (int col = local_row_id; col < float4_num; col += Br)
        {
            int global_offset = batch_id * (H * N * float4_num) + head_id * (N * float4_num) + global_col_id * float4_num + col;
            float4 tmp = ((float4*)K)[global_offset];
            // 加载K
            half2 half_tmp_1, half_tmp_2;
            half_tmp_1 = __floats2half2_rn(tmp.x, tmp.y);
            half_tmp_2 = __floats2half2_rn(tmp.z, tmp.w);
            *(half2*)(&shared_Ki[local_col_id * padding_d + col*4]) = half_tmp_1;
            *(half2*)(&shared_Ki[local_col_id * padding_d + col*4 + 2]) = half_tmp_2;

            // 加载V
            tmp = ((float4*)V)[global_offset];
            *(float4*)(&shared_Vi[local_col_id * padding_d + col*4]) = tmp;
            // 转置V
            shared_Vi_T[col*4 * Bc + local_col_id] = tmp.x;
            shared_Vi_T[(col*4+1) * Bc + local_col_id] = tmp.y;
            shared_Vi_T[(col*4+2) * Bc + local_col_id] = tmp.z;
            shared_Vi_T[(col*4+3) * Bc + local_col_id] = tmp.w;
        }
        // 逐元素加载剩余部分
        for (int col = float4_num * 4 + local_row_id; col < d; col += Br)
        {
            // 加载K
            tmp = read_4d(K, batch_id, head_id, global_col_id, col, H, N, d);
            shared_Ki[local_col_id * padding_d + col] = __float2half_rn(tmp);
            // 加载V
            tmp = read_4d(V, batch_id, head_id, global_col_id, col, H, N, d);
            shared_Vi[local_col_id * padding_d + col] = tmp;
            // 转置V
            shared_Vi_T[col * Bc + local_col_id] = tmp;
        }

        for (int i = 0; i < Tr; i++)
        {
            __syncthreads();
            // 全局行坐标
            int global_row_id = i * Br + local_row_id;
            // 判断是否掩盖当前线程
            bool mask = (global_row_id >= N);
            if (mask)
            {
                continue;
            }
            // 实际的Br
            // int actual_Br = (i == Tr - 1) ? (N % Br == 0 ? Br : N % Br) : Br;

            /* 从全局内存中加载数据 */
            /* 加载Q O */
            // 向量化加载
            for (int col = local_col_id; col < float4_num; col += actual_Bc)
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
            for (int col = float4_num*4 + local_col_id; col < d; col += actual_Bc)
            {
                // 加载Q
                tmp = read_4d(Q, batch_id, head_id, global_row_id, col, H, N, d);
                shared_Qi[local_row_id * padding_d + col] = __float2half_rn(tmp);
                // write_2d(shared_Qi, local_row_id, col, padding_d, tmp);
                // 加载O
                tmp = read_4d(O, batch_id, head_id, global_row_id, col, H, N, d);
                shared_Oi[local_row_id * padding_d + col] = tmp;
                // write_2d(shared_Oi, local_row_id, col, padding_d, tmp);
            }

            // 加载li mi
            if (local_col_id == 0)
            {
                tmp = read_4d(l, batch_id, head_id, global_row_id, 0, H, N, 1);
                write_2d(shared_li, local_row_id, 0, 1, tmp);
                tmp = read_4d(m, batch_id, head_id, global_row_id, 0, H, N, 1);
                write_2d(shared_mi, local_row_id, 0, 1, tmp);
            }
            __syncthreads();

            /* 开始计算 */
            // 计算Sij = Qi * Ki^t

            /* wmma */
            compute_sij_wmma(shared_Qi, shared_Ki, shared_Sij, padding_d, Br, Bc);
            // if (local_row_id == 0 && local_col_id == 0)
            // {
            //     for (int m = 0; m < Bc * Br; m++)
            //     {
            //         if (isnan(shared_Sij[m]))
            //         {
            //             printf("shared_Sij[%d] is nan %f.\n", m, shared_Sij[m]);
            //         }
            //     }
            // }
            __syncthreads();

            // 计算mij = rowmax(Sij), 
            tmp = read_2d(shared_Sij, local_row_id, local_col_id, Bc);
            float row_max = warpReduceMax(tmp, actual_Bc);
            if (local_col_id == 0)
            {
                write_2d(shared_mij, local_row_id, 0, 1, row_max);
                // 更新minew
                tmp = read_2d(shared_mi, local_row_id, 0, 1);
                tmp = fmaxf(tmp, row_max);
                write_2d(shared_mi_new, local_row_id, 0, 1, tmp);
                // lij清零
                tmp = 0;
                write_2d(shared_lij, local_row_id, 0, 1, tmp);
            }
            //__syncthreads();
            __syncwarp();
            // 计算Pij = exp(Sij - mij)
            // 读取Sij和mij
            tmp = read_2d(shared_Sij, local_row_id, local_col_id, Bc);
            // 计算exp(平移)
            tmp = __expf(tmp - read_2d(shared_mij, local_row_id, 0, 1));
            // 直接写回Sij
            write_2d(shared_Sij, local_row_id, local_col_id, Bc, tmp);

            // 计算rowsum
            float warpsum = warpReduceSum(tmp, actual_Bc);
            if (local_col_id % warpSize == 0)
            {
                atomicAdd(&(shared_lij[local_row_id]), warpsum);
            }
            // __syncthreads();
            __syncwarp();

            // 计算linew = exp(mi - minew) * li + exp(mij - minew) * lij
            if (local_col_id == 0)
            {
                float mi = read_2d(shared_mi, local_row_id, 0, 1);
                float minew = read_2d(shared_mi_new, local_row_id, 0, 1);
                float mij = read_2d(shared_mij, local_row_id, 0, 1);
                float li = read_2d(shared_li, local_row_id, 0, 1);
                float lij = read_2d(shared_lij, local_row_id, 0, 1);
                tmp = __expf(mi - minew) * li + __expf(mij - minew) * lij;
                write_2d(shared_li_new, local_row_id, 0, 1, tmp);
            }
            // __syncthreads();
            __syncwarp();

            /* 计算diag(lnew)^-1 * (diag(li)*exp(mi-minew)*Oi + exp(mij - minew)*Sij*Vj) */
            // // bank串行化, 读取后广播
            // float mi, minew, mij, li, linew;
            // if (local_col_id % 4 == 0)
            // {
            //     mi = read_2d(shared_mi, local_row_id, 0, 1);
            //     minew = read_2d(shared_mi_new, local_row_id, 0, 1);
            //     mij = read_2d(shared_mij, local_row_id, 0, 1);
            //     li = read_2d(shared_li, local_row_id, 0, 1);
            //     linew = read_2d(shared_li_new, local_row_id, 0, 1);
            // }
            // mi = __shfl_sync(0xFFFFFFFF, mi, local_col_id & (~3)); // // 让每 4 个线程共享
            // minew = __shfl_sync(0xFFFFFFFF, minew, local_col_id & (~3));
            // mij = __shfl_sync(0xFFFFFFFF, mij, local_col_id & (~3));
            // li = __shfl_sync(0xFFFFFFFF, li, local_col_id & (~3));
            // linew = __shfl_sync(0xFFFFFFFF, linew, local_col_id & (~3));
            float mi = read_2d(shared_mi, local_row_id, 0, 1);
            float minew = read_2d(shared_mi_new, local_row_id, 0, 1);
            float mij = read_2d(shared_mij, local_row_id, 0, 1);
            float li = read_2d(shared_li, local_row_id, 0, 1);
            float linew = read_2d(shared_li_new, local_row_id, 0, 1);
            float exp_mij_minew = __expf(mij - minew);
            float li_exp_mi_minew = __expf(mi - minew) * li;

            // 首先将shared_Oi放缩li_exp_mi_minew
            scale_shared_oi(shared_Oi, local_row_id, local_col_id, d, padding_d, actual_Bc, float4_num, li_exp_mi_minew);
            __syncwarp();
            // 根据actual_Bc计算需要的warp数量
            int need_warp_num = (actual_Bc / 16) * 2;
            if (need_warp_num != 0)
            {
                // wmma
                wmma_sij_vj(shared_Sij, shared_Vi_T, shared_Oi_temp, padding_d, Br, Bc, need_warp_num);
                __syncthreads();
            }
            // 传统方式计算剩下的Sij * Vj
            need_warp_num = (need_warp_num / 2) * 16;
            if (need_warp_num < actual_Bc)
            {
                for (int col = local_col_id; col < d; col += actual_Bc)
                {
                    float tmp1 = 0;
                    for (int k = need_warp_num; k < actual_Bc; k++)
                    {
                        tmp1 += shared_Sij[local_row_id * Bc + k] * shared_Vi_T[col * Bc + k];
                    }
                    // 写入shared_Oi_temp
                    shared_Oi_temp[local_row_id * padding_d + col] = tmp1;
                }
            }
            // 将Sij * Vj的结果放缩exp(mij - minew)
            scale_shared_oi(shared_Oi_temp, local_row_id, local_col_id, d, padding_d, actual_Bc, float4_num, exp_mij_minew);
            // 与shared_Oi求和
            add_shared_oi_temp(shared_Oi, shared_Oi_temp, local_row_id, local_col_id, d, padding_d, float4_num, actual_Bc);
            // TODO 将shared_Oi放缩diag(lnew)^-1
            scale_shared_oi(shared_Oi, local_row_id, local_col_id, d, padding_d, actual_Bc, float4_num, 1.0f/linew);

            


            if (actual_Bc % 16 == 0)
            {
                // wmma计算Sij*Vj

            }
            else
            {
                // 乘以Oi
                for (int col = local_col_id; col < d; col += actual_Bc)
                {
                    float Oi = read_2d(shared_Oi, local_row_id, col, padding_d);

                    float tmp_1 = 0, tmp_2 = 0;
                    // for (int k = 0; k < actual_Bc; k++) 
                    // {
                    //     tmp_1 += shared_Sij[local_row_id * Bc + k] * shared_Vi_T[col * Bc + k];
                    //     //tmp_2 += (read_2d(shared_Sij, local_row_id, k, Bc) * read_2d(shared_Vi, k, col, padding_d));
                    // }
                    for (int k = 0; k < actual_Bc; k+=4) 
                    {
                        float4 float4_tmp_1 = *(float4*)&shared_Sij[local_row_id * Bc + k];
                        float4 float4_tmp_2 = *(float4*)&shared_Vi_T[col * Bc + k];
                        tmp_1 += float4_tmp_1.x * float4_tmp_2.x + 
                                float4_tmp_1.y * float4_tmp_2.y + 
                                float4_tmp_1.z * float4_tmp_2.z + 
                                float4_tmp_1.w * float4_tmp_2.w;
                    }
                    tmp_1 *= exp_mij_minew;
                    Oi += tmp_1;
                    Oi /= linew;
                    shared_Oi[local_row_id * padding_d + col] = Oi;
                }
            }
            __syncthreads();
            


            // 从共享内存写回全局内存
            // 写Oi 向量化
            for (int col = local_col_id; col < float4_num; col += actual_Bc)    
            {
                int global_offset = batch_id * (H * N * float4_num) + head_id * (N * float4_num) + global_row_id * float4_num + col;
                float4 tmp = *(float4*)(&shared_Oi[local_row_id * padding_d + col*4]);
                ((float4*)O)[global_offset] = tmp;
            }
            // 处理不整除的部分
            for (int col = float4_num * 4 + local_col_id; col < d; col += actual_Bc)
            {
                float tmp = read_2d(shared_Oi, local_row_id, col, padding_d);
                write_4d(O, batch_id, head_id, global_row_id, col, H, N, d, tmp);
            }
                
            if (local_col_id == 0)
            {
                // 写linew
                tmp = read_2d(shared_li_new, local_row_id, 0, 1);
                write_4d(l, batch_id, head_id, global_row_id, 0, H, N, 1, tmp);
                // 写minew
                tmp = read_2d(shared_mi_new, local_row_id, 0, 1);
                write_4d(m, batch_id, head_id, global_row_id, 0, H, N, 1, tmp);
            }
        }
    }
}

/* 主机函数 最终优化版本*/
float flashAttentionHost_best(float* Q, float* K, float* V, float* O, int B, int H, int N, int d)
{
    /* 把共享内存上限设置为64KB */
    cudaFuncSetAttribute(
        flash_attention_best,                      // 核函数名
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
                ((Br * Bc + 31) / 32 * 32) + // shared_Sij
                ((Br * 1 + 31) / 32 * 32) + // shared_mij
                ((Br * 1 + 31) / 32 * 32) + // shared_lij
                ((Br * 1 + 31) / 32 * 32) + // shared_mi_new
                ((Br * 1 + 31) / 32 * 32) ; // shared_li_new
    shared_mem_size *= sizeof(float);
    std::cerr << "Sahred memory need [" << shared_mem_size << "], have [" << M << "]" << std::endl;
    if (shared_mem_size > M)
    {
        return -1;
    }

    /* 打印kernel最优线程块配置 */
    int min_grid_size = 0, block_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, flash_attention_best, 0, 0);
    std::cout << "optimal block size is [" << block_size << "]" << std::endl;
 
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
    dim3 grid_dim(B , H, 1);
    // 开始计时
    cudaEventRecord(start, 0);
    // 启动核函数 指定共享内存大小
    flash_attention_best<<<grid_dim, block_dim, shared_mem_size>>>(Q, K, V, O, device_l, device_m, B, H, N, d, Br, Bc, Tr, Tc);
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


/**************************************************************以下为原始版本*********************************************************/
/* flash attention核函数, 原始版 */
__global__ void flash_attention_origin(float* __restrict__ Q, float* __restrict__ K, float* __restrict__ V, 
                                float* __restrict__ O, float* __restrict__ l, float* __restrict__ m,
                            int B, int H, int N, int d, int Br, int Bc, int Tr, int Tc)
{
    /* 获取当前线程位置 */
    // 得出当前线程位于哪个B和H
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    // 线程对应子矩阵中的行号列号
    int local_col_id = threadIdx.x;
    int local_row_id = threadIdx.y;
    /* 划分共享内存 */
    extern __shared__  float shared_mem[];
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
    // 划分mij
    float* shared_mij = shared_Sij + ((Br * Bc + 31) / 32 * 32);        // 存储每一次Sij的行最大值, Br
    // 划分lij
    float* shared_lij = shared_mij + ((Br * 1 + 31) / 32 * 32);        // Br
    // 划分minew
    float* shared_mi_new = shared_lij + ((Br * 1 + 31) / 32 * 32);      // Br
    // 划分linew
    float* shared_li_new = shared_mi_new + ((Br * 1 + 31) / 32 * 32);   // Br


    float tmp = 0;
    /* 遍历所有分块 */ 
    for (int j = 0; j < Tc; j++)
    {
        // 全局列坐标
        int global_col_id = j * Bc + local_col_id;
        bool mask = global_col_id >= N;
        if (mask)
        {
            return;
        }
        // 实际的Bc
        int actual_Bc = (j == Tc - 1) ? (N % Bc == 0 ? Bc : N % Bc) : Bc;
        // 加载KV
        for (int col = local_row_id; col < d; col += Br)
        {
            // 加载K
            tmp = read_4d(K, batch_id, head_id, global_col_id, col, H, N, d);
            write_2d(shared_Ki, local_col_id, col, d, tmp);
            // 加载V
            tmp = read_4d(V, batch_id, head_id, global_col_id, col, H, N, d);
            write_2d(shared_Vi, local_col_id, col, d, tmp);
        }

        for (int i = 0; i < Tr; i++)
        {
            __syncthreads();
            // 全局行坐标
            int global_row_id = i * Br + local_row_id;
            // 判断是否掩盖当前线程
            bool mask = (global_row_id >= N);
            if (mask)
            {
                continue;
            }
            // 实际的Br
            // int actual_Br = (i == Tr - 1) ? (N % Br == 0 ? Br : N % Br) : Br;

            /* 从全局内存中加载数据 */
            /* 加载Q O */
            for (int col = local_col_id; col < d; col += actual_Bc)
            {
                // 加载Q
                tmp = read_4d(Q, batch_id, head_id, global_row_id, col, H, N, d);
                write_2d(shared_Qi, local_row_id, col, d, tmp);
                // 加载O
                tmp = read_4d(O, batch_id, head_id, global_row_id, col, H, N, d);
                write_2d(shared_Oi, local_row_id, col, d, tmp);
            }

            // 加载li mi
            if (local_col_id == 0)
            {
                tmp = read_4d(l, batch_id, head_id, global_row_id, 0, H, N, 1);
                write_2d(shared_li, local_row_id, 0, 1, tmp);
                tmp = read_4d(m, batch_id, head_id, global_row_id, 0, H, N, 1);
                write_2d(shared_mi, local_row_id, 0, 1, tmp);
            }
            __syncthreads();

            /* 开始计算 */
            // 计算Sij = Qi * Ki^t
            tmp = 0;
            for (int col = 0; col < d; col++)
            {
                tmp += (read_2d(shared_Qi, local_row_id, col, d) * read_2d(shared_Ki, local_col_id, col, d));
            }
            // 将结果写到Sij的[local_row_id, local_col_id]处, 无需原子因为一个线程只处理Sij的一个位置
            write_2d(shared_Sij, local_row_id, local_col_id, Bc, tmp);
            __syncthreads();

            // 计算mij = rowmax(Sij), 
            tmp = read_2d(shared_Sij, local_row_id, local_col_id, Bc);
            float row_max = -INFINITY;
            if (local_col_id == 0)
            {
                for (int col = 0; col < actual_Bc; col++)
                {
                    row_max = fmaxf(row_max, read_2d(shared_Sij, local_row_id, col, Bc));
                }
                write_2d(shared_mij, local_row_id, 0, 1, row_max);
                // 更新minew
                tmp = read_2d(shared_mi, local_row_id, 0, 1);
                tmp = fmaxf(tmp, row_max);
                write_2d(shared_mi_new, local_row_id, 0, 1, tmp);
                // lij清零
                tmp = 0;
                write_2d(shared_lij, local_row_id, 0, 1, tmp);
            }
            __syncthreads();
            // 计算Pij = exp(Sij - mij)
            // 读取Sij和mij
            tmp = read_2d(shared_Sij, local_row_id, local_col_id, Bc);
            // 计算exp(平移)
            // if (local_row_id == 0 && local_col_id == 0)
            // {
            //     printf("tmp - read_2d(shared_mij, local_row_id, 0, 1) = %f.\n", tmp - read_2d(shared_mij, local_row_id, 0, 1));
            // }
            tmp = __expf(tmp - read_2d(shared_mij, local_row_id, 0, 1));
            // 直接写回Sij
            write_2d(shared_Sij, local_row_id, local_col_id, Bc, tmp);
            __syncthreads();

            // 计算rowsum
            float warpsum = 0;
            if (local_col_id == 0)
            {
                for (int col = 0; col < actual_Bc; col++)
                {
                    warpsum += read_2d(shared_Sij, local_row_id, col, Bc);
                }
                atomicAdd(&(shared_lij[local_row_id]), warpsum);
            }
            __syncthreads();

            // 计算linew = exp(mi - minew) * li + exp(mij - minew) * lij
            if (local_col_id == 0)
            {
                float mi = read_2d(shared_mi, local_row_id, 0, 1);
                float minew = read_2d(shared_mi_new, local_row_id, 0, 1);
                float mij = read_2d(shared_mij, local_row_id, 0, 1);
                float li = read_2d(shared_li, local_row_id, 0, 1);
                float lij = read_2d(shared_lij, local_row_id, 0, 1);
                //printf("mi - minew = %f. mij - minew = %f.\n", mi - minew, mij - minew);
                tmp = __expf(mi - minew) * li + __expf(mij - minew) * lij;
                write_2d(shared_li_new, local_row_id, 0, 1, tmp);
            }
            __syncthreads();

            /* 计算diag(lnew)^-1 * (diag(li)*exp(mi-minew)*Oi + exp(mij - minew)*Sij*Vj) */
            float mi = read_2d(shared_mi, local_row_id, 0, 1);
            float minew = read_2d(shared_mi_new, local_row_id, 0, 1);
            float mij = read_2d(shared_mij, local_row_id, 0, 1);
            float li = read_2d(shared_li, local_row_id, 0, 1);
            float linew = read_2d(shared_li_new, local_row_id, 0, 1);
            float exp_mij_minew = __expf(mij - minew);

            __syncthreads();

            // 乘以Oi
            for (int col = local_col_id; col < d; col += actual_Bc)
            {
                tmp = __expf(mi - minew) * li;
                float Oi = read_2d(shared_Oi, local_row_id, col, d);
                tmp *= Oi;
                float tmp_1 = 0;
                for (int k = 0; k < actual_Bc; k++) 
                {
                    tmp_1 += (read_2d(shared_Sij, local_row_id, k, Bc) * read_2d(shared_Vi, k, col, d));
                }
                tmp_1 *= exp_mij_minew;
                tmp += tmp_1;
                tmp /= linew;
                write_2d(shared_Oi, local_row_id, col, d, tmp);
            }


            // 从共享内存写回全局内存
            for (int col = local_col_id; col < d; col += actual_Bc)
            {
                float tmp = read_2d(shared_Oi, local_row_id, col, d);
                write_4d(O, batch_id, head_id, global_row_id, col, H, N, d, tmp);
            }
                
            if (local_col_id == 0)
            {
                // 写linew
                tmp = read_2d(shared_li_new, local_row_id, 0, 1);
                write_4d(l, batch_id, head_id, global_row_id, 0, H, N, 1, tmp);
                // 写minew
                tmp = read_2d(shared_mi_new, local_row_id, 0, 1);
                write_4d(m, batch_id, head_id, global_row_id, 0, H, N, 1, tmp);
            }
        }
    }
}

/*主机函数原始版*/
float flashAttentionHost_origin(float* Q, float* K, float* V, float* O, int B, int H, int N, int d)
{
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
    int shared_mem_size = 
                ((Br * d + 31) / 32 * 32) + //shared_Qi
                ((Bc * d + 31) / 32 * 32) + // shared_Ki
                ((Bc * d + 31) / 32 * 32) + // shared_Vi
                ((Br * d + 31) / 32 * 32) + // shared_Oi
                ((Br * 1 + 31) / 32 * 32) + // shared_mi
                ((Br * 1 + 31) / 32 * 32) + // shared_li
                ((Br * Bc + 31) / 32 * 32) + // shared_Sij
                ((Br * 1 + 31) / 32 * 32) + // shared_mij
                ((Br * 1 + 31) / 32 * 32) + // shared_lij
                ((Br * 1 + 31) / 32 * 32) + // shared_mi_new
                ((Br * 1 + 31) / 32 * 32) ; // shared_li_new
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
    dim3 grid_dim(B , H, 1);
    // 开始计时
    cudaEventRecord(start, 0);
    // 启动核函数 指定共享内存大小
    flash_attention_origin<<<grid_dim, block_dim, shared_mem_size>>>(Q, K, V, O, device_l, device_m, B, H, N, d, Br, Bc, Tr, Tc);
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

