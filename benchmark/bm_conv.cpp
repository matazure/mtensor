#include "bm_conv.hpp"

auto bm_host_tensor2f_general_roll_conv = bm_tensor_general_roll_conv<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_general_roll_conv)->Arg(128);

auto bm_host_tensor2f_general_unroll_conv = bm_tensor_general_unroll_conv<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_general_unroll_conv)->Arg(128);