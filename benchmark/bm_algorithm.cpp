#include "bm_algorithm.hpp"

void bm_host_memcpy(benchmark::State& state) {
    int ts_size = state.range(0);
    tensor<float, 1> ts_src(ts_size);
    tensor<float, 1> ts_dst(ts_size);

    while (state.KeepRunning()) {
        memcpy(ts_dst.data(), ts_src.data(), sizeof(ts_src[0]) * ts_src.size());
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(float));
}

BENCHMARK(bm_host_memcpy)->Arg(100_M);

auto bm_host_tensor1f_copy = bm_tensor_copy<tensor<float, 1>>;
auto bm_host_tensor2f_copy = bm_tensor_copy<tensor<float, 2>>;
BENCHMARK(bm_host_tensor1f_copy)->Arg(100_M);
BENCHMARK(bm_host_tensor2f_copy)->Arg(10_K);

auto bm_cuda_tensor2p4f_copy = bm_tensor_copy<tensor<point4f, 2>>;
BENCHMARK(bm_cuda_tensor2p4f_copy)->Arg(2_K);
auto bm_cuda_tensor2a4f_copy = bm_tensor_copy<tensor<std::array<float, 4>, 2>>;
BENCHMARK(bm_cuda_tensor2a4f_copy)->Arg(2_K);

auto bm_host_tensor2f_fill = bm_tensor_fill<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_fill)->Arg(10_K);
auto bm_host_tensor2lt_fill = bm_tensor_fill<tensor<local_tensor<float, dim<2, 2>>, 2>>;
BENCHMARK(bm_host_tensor2lt_fill)->Arg(2_K);

auto bm_host_tensor2f_for_each = bm_tensor_for_each<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_for_each)->Arg(10_K);

auto bm_host_tensor2f_transform = bm_tensor_transform<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_transform)->Arg(10_K);

auto bm_host_tensor2f_column_major_layout_copy =
    bm_tensor_copy<tensor<float, 2, column_major_layout<2>>>;
BENCHMARK(bm_host_tensor2f_column_major_layout_copy)->Arg(10_K);

auto bm_host_tensor2f_row_layout_copy = bm_tensor_copy<tensor<float, 2, row_major_layout<2>>>;
BENCHMARK(bm_host_tensor2f_row_layout_copy)->Arg(10_K);

auto bm_host_tensor2f_padding_layout_copy = bm_tensor_copy<tensor<float, 2, padding_layout<2>>>;
BENCHMARK(bm_host_tensor2f_padding_layout_copy)->Arg(10_K);
