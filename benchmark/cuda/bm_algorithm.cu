
#include "../bm_algorithm.hpp"

auto bm_cuda_tensor1f_copy = bm_tensor_copy<cuda::tensor<float, 1>>;
auto bm_cuda_tensor2f_copy = bm_tensor_copy<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor1f_copy)->Arg(1_G);
BENCHMARK(bm_cuda_tensor2f_copy)->Arg(10_K);
auto bm_cuda_tensor2p4f_copy = bm_tensor_copy<cuda::tensor<point4f, 2>>;
BENCHMARK(bm_cuda_tensor2p4f_copy)->Arg(8_K);
auto bm_cuda_tensor2a4f_copy = bm_tensor_copy<cuda::tensor<std::array<float, 4>, 2>>;
BENCHMARK(bm_cuda_tensor2a4f_copy)->Arg(8_K);

auto bm_cuda_tensor2f_fill = bm_tensor_fill<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_fill)->Arg(10_K);
auto bm_cuda_tensor2lt_fill = bm_tensor_fill<cuda::tensor<local_tensor<float, dim<2, 2>>, 2>>;
BENCHMARK(bm_cuda_tensor2lt_fill)->Arg(2_K);

auto bm_cuda_tensor2f_for_each = bm_tensor_for_each<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_for_each)->Arg(10_K);

auto bm_cuda_tensor2f_transform = bm_tensor_transform<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_transform)->Arg(10_K);

auto bm_cuda_tensor2f_column_major_layout_copy =
    bm_tensor_copy<cuda::tensor<float, 2, column_major_layout<2>>>;
BENCHMARK(bm_cuda_tensor2f_column_major_layout_copy)->Arg(10_K);
auto bm_cuda_tensor2f_row_layout_copy = bm_tensor_copy<cuda::tensor<float, 2, row_major_layout<2>>>;
BENCHMARK(bm_cuda_tensor2f_row_layout_copy)->Arg(10_K);
auto bm_cuda_tensor2f_padding_layout_copy =
    bm_tensor_copy<cuda::tensor<float, 2, padding_layout<2>>>;
BENCHMARK(bm_cuda_tensor2f_padding_layout_copy)->Arg(10_K);