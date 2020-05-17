#include "../bm_view.hpp"

auto bm_cuda_tensor2f_view_slice = bm_tensor_view_slice<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_view_slice)->Arg(10_K);

auto bm_cuda_tensor2f_view_stride = bm_tensor_view_stride<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_view_stride)->Arg(10_K);

auto bm_cuda_tensor2f_view_zip = bm_tensor_view_zip2<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_view_zip)->Arg(10_K);

auto bm_cuda_tensor2f_view_eye = bm_tensor_view_eye<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_view_eye)->Arg(10_K);

auto bm_cuda_tensor2f_view_meshgrid = bm_tensor_view_meshgrid2<device_t, float>;
BENCHMARK(bm_cuda_tensor2f_view_meshgrid)->Arg(10_K);
