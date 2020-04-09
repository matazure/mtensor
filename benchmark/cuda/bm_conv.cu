#include "../bm_conv.hpp"

auto bm_cuda_tensor2f_general_roll_conv = bm_tensor_general_roll_conv<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_general_roll_conv)->Arg(512)->Arg(1_K)->Arg(10_K);

auto bm_cuda_tensor2f_general_unroll_conv = bm_tensor_general_unroll_conv<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_general_unroll_conv)->Arg(512)->Arg(1_K)->Arg(2_K)->Arg(10_K);

auto bm_cuda_tensor2f_padding_layout_general_unroll_conv =
    bm_tensor_padding_layout_general_unroll_conv<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_padding_layout_general_unroll_conv)
    ->Arg(512)
    ->Arg(1_K)
    ->Arg(2_K)
    ->Arg(10_K);