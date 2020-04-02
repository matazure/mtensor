#include "bm_view.hpp"

auto bm_host_tensor2f_crop = bm_tensor_crop<tensor<float, 2>>;
// BENCHMARK(bm_host_tensor2f_crop)->Arg(32_K);

auto bm_host_tensor2f_stride = bm_tensor_stride<tensor<float, 2>>;
// BENCHMARK(bm_host_tensor2f_stride)->Arg(32_K);
