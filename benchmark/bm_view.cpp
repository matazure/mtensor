#include "bm_view.hpp"

auto bm_host_tensor2f_view_crop = bm_tensor_crop<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_view_crop)->Arg(10_K);

auto bm_host_tensor2f_view_stride = bm_tensor_stride<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_view_stride)->Arg(10_K);
