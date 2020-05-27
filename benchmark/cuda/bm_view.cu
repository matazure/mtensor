#include "../bm_view.hpp"

auto bm_cuda_tensor2f_view_slice = bm_tensor_view_slice<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_view_slice)->Arg(10_K);

auto bm_cuda_tensor2f_view_stride = bm_tensor_view_stride<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_view_stride)->Arg(10_K);

auto bm_cuda_tensor2f_view_gather_scalar_axis0 =
    bm_tensor_view_gather_scalar_axis0<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_view_gather_scalar_axis0)->Arg(100_M);
auto bm_cuda_tensor2f_view_gather_scalar_axis1 =
    bm_tensor_view_gather_scalar_axis1<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_view_gather_scalar_axis1)->Arg(100_M);

// if no use lambda , it does not work in cuda.
// auto bm_cuda_tensor2f_view_zip = bm_tensor_view_zip2<cuda::tensor<float, 2>>;
// BENCHMARK(bm_cuda_tensor2f_view_zip)->Arg(10_K);

auto bm_cuda_tensor2f_view_eye = bm_tensor_view_eye<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_view_eye)->Arg(10_K);

// because tuple hase runtime error
// auto bm_cuda_tensor2f_view_meshgrid = bm_tensor_view_meshgrid2<device_t, float>;
// BENCHMARK(bm_cuda_tensor2f_view_meshgrid)->Arg(10_K);
