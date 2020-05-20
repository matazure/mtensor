#include "bm_conv.hpp"

auto bm_host_tensor2f_general_roll_conv = bm_tensor2f_general_roll_conv<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_general_roll_conv)->Arg(128)->Arg(10_K);

auto bm_host_tensor2f_general_unroll_conv = bm_tensor2f_general_unroll_conv<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_general_unroll_conv)->Arg(128)->Arg(10_K);

auto bm_host_tensor2_view_conv_local_tensor3x3 =
    bm_tensor2_view_conv_local_tensor3x3<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2_view_conv_local_tensor3x3)->Arg(128)->Arg(10_K);

auto bm_host_tensor_view_conv_tensor = bm_tensor_view_conv_tensor<tensor<float, 2>>;
BENCHMARK(bm_host_tensor_view_conv_tensor)->Arg(128)->Arg(10_K);

auto bm_host_tensor2_view_conv_neighbors_weights3x3 =
    bm_tensor2_view_conv_neighbors_weights3x3<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2_view_conv_neighbors_weights3x3)->Arg(128)->Arg(10_K);