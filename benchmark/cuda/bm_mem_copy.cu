#include "../bm_mem_copy.hpp"

auto bm_cuda_tensor1f_d2d_mem_copy =
    bm_tensor_mem_copy<cuda::tensor<float, 1>, cuda::tensor<float, 1>>;
BENCHMARK(bm_cuda_tensor1f_d2d_mem_copy)->Arg(100_M);

auto bm_cuda_tensor1f_d2h_mem_copy = bm_tensor_mem_copy<cuda::tensor<float, 1>, tensor<float, 1>>;
BENCHMARK(bm_cuda_tensor1f_d2h_mem_copy)->Arg(100_M);

auto bm_cuda_tensor1f_h2d_mem_copy = bm_tensor_mem_copy<tensor<float, 1>, cuda::tensor<float, 1>>;
BENCHMARK(bm_cuda_tensor1f_h2d_mem_copy)->Arg(100_M);

auto bm_cuda_tensor1f_h2h_mem_copy = bm_tensor_mem_copy<tensor<float, 1>, tensor<float, 1>>;
BENCHMARK(bm_cuda_tensor1f_h2h_mem_copy)->Arg(100_M);