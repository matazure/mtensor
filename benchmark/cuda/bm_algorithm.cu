
#include "../bm_algorithm.hpp"

auto bm_cuda_tensor1f_copy = bm_tensor_copy<cuda::tensor<float, 1>>;
auto bm_cuda_tensor2f_copy = bm_tensor_copy<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor1f_copy)->Arg(1_G);
BENCHMARK(bm_cuda_tensor2f_copy)->Arg(32_K);

auto bm_cuda_tensor2f_fill = bm_tensor_fill<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_fill)->Arg(32_K);

auto bm_cuda_tensor2f_for_each = bm_tensor_for_each<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_for_each)->Arg(32_K);

auto bm_cuda_tensor2f_transform = bm_tensor_transform<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_transform)->Arg(32_K);
