#include "../bm_binary_operator.hpp"

auto bm_cuda_tensor2f_add = bm_tensor_add<cuda::tensor<float, 2>>;
auto bm_cuda_tensor2f_sub = bm_tensor_add<cuda::tensor<float, 2>>;
auto bm_cuda_tensor2f_mul = bm_tensor_add<cuda::tensor<float, 2>>;
auto bm_cuda_tensor2f_div = bm_tensor_add<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_add)->Arg(10_K);
BENCHMARK(bm_cuda_tensor2f_sub)->Arg(10_K);
BENCHMARK(bm_cuda_tensor2f_mul)->Arg(10_K);
BENCHMARK(bm_cuda_tensor2f_div)->Arg(10_K);
