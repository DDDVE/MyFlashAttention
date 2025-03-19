#include <torch/extension.h>
#include <vector>

/*最终优化版*/
extern float flashAttentionHost_best(float* Q, float* K, float* V, float* O, int B, int H, int N, int d);

/*未优化版*/
extern float flashAttentionHost_origin(float* Q, float* K, float* V, float* O, int B, int H, int N, int d);

/* python接口函数 最优版*/
torch::Tensor flash_attention_torch_best(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                        torch::Tensor O, int B, int H, int N, int d)
{
    // 检查张量是否在GPU上
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(O.is_cuda(), "O must be a CUDA tensor");

    // 确保张量是连续的
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    O = O.contiguous();

    // 调用主机函数 
    float time = flashAttentionHost_best(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), B, H, N, d);

    // 将时间传回 Python 层
    return torch::tensor({time}, torch::TensorOptions().dtype(torch::kFloat32));
}
/***********************************************************python接口函数 未优化版***************************************************************/
torch::Tensor flash_attention_torch_origin(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                        torch::Tensor O, int B, int H, int N, int d)
{
    // 检查张量是否在GPU上
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(O.is_cuda(), "O must be a CUDA tensor");

    // 确保张量是连续的
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    O = O.contiguous();

    // 调用主机函数 
    float time = flashAttentionHost_origin(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), B, H, N, d);

    // 将时间传回 Python 层
    return torch::tensor({time}, torch::TensorOptions().dtype(torch::kFloat32));
}

// 定义模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attention_best", &flash_attention_torch_best,
        "Torch wrapper for CUDA flash attention function, best edition");

    m.def("flash_attention_origin", &flash_attention_torch_origin,
        "Torch wrapper for CUDA flash attention function, origin edition");
}

