#include "bm_mem_copy.hpp"

auto bm_host_tensor1f_h2h_mem_copy = bm_tensor_mem_copy<tensor<float, 1>, tensor<float, 1>>;
BENCHMARK(bm_host_tensor1f_h2h_mem_copy)->Arg(100_M);