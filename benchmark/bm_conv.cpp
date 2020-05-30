#include "bm_conv.hpp"

// auto bm_host_tensor2f_general_roll_conv = bm_tensor2f_general_roll_conv<tensor<float, 2>>;
// BENCHMARK(bm_host_tensor2f_general_roll_conv)->Arg(128)->Arg(10_K);

// auto bm_host_tensor2f_general_unroll_conv = bm_tensor2f_general_unroll_conv<tensor<float, 2>>;
// BENCHMARK(bm_host_tensor2f_general_unroll_conv)->Arg(128)->Arg(10_K);

auto bm_host_tensor2_view_conv_local_tensor3x3 =
    bm_tensor2_view_conv_local_tensor3x3<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2_view_conv_local_tensor3x3)->Arg(128)->Arg(10_K);

auto bm_host_tensor_view_conv_tensor3x3 = bm_tensor_view_conv_tensor3x3<tensor<float, 2>>;
BENCHMARK(bm_host_tensor_view_conv_tensor3x3)->Arg(128)->Arg(10_K);

auto bm_host_tensor2_view_conv_stride2_relu6_local_tensor3x3 =
    bm_tensor2_view_conv_stride2_relu6_local_tensor3x3<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2_view_conv_stride2_relu6_local_tensor3x3)->Arg(128)->Arg(10_K);

// auto bm_host_tensor2_view_conv_neighbors_weights3x3 =
//     bm_tensor2_view_conv_neighbors_weights3x3<tensor<float, 2>>;
// BENCHMARK(bm_host_tensor2_view_conv_neighbors_weights3x3)->Arg(128)->Arg(10_K);

auto bm_host_tensor2point4f_view_conv_local_tensor3x3 =
    bm_tensor2_view_conv_local_tensor3x3<tensor<point4f, 2>>;
BENCHMARK(bm_host_tensor2point4f_view_conv_local_tensor3x3)->Arg(128)->Arg(2_K);

#ifdef __GNUC__
#ifdef __AVX__
typedef float simd_type __attribute__((vector_size(32)));
#else
typedef float simd_type __attribute__((vector_size(16)));
#endif
auto bm_host_tensor2simd_view_conv_local_tensor3x3 =
    bm_tensor2_view_conv_local_tensor3x3<tensor<simd_type, 2>>;
BENCHMARK(bm_host_tensor2simd_view_conv_local_tensor3x3)->Arg(128)->Arg(2_K);
#endif
