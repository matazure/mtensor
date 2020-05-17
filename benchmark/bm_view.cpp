#include "bm_view.hpp"

auto bm_host_tensor2f_view_slice = bm_tensor_view_slice<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_view_slice)->Arg(10_K);

auto bm_host_tensor2f_view_stride = bm_tensor_view_stride<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_view_stride)->Arg(10_K);

auto bm_host_tensor2f_view_zip = bm_tensor_view_zip2<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_view_zip)->Arg(10_K);

auto bm_host_tensor2f_view_eye = bm_tensor_view_eye<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_view_eye)->Arg(10_K);

// auto bm_host_tensor2f_view_meshgrid = bm_tensor_view_meshgrid2<host_t, float>;
// BENCHMARK(bm_host_tensor2f_view_meshgrid)->Arg(10_K);
