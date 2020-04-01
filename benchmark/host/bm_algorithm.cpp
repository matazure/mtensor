#include "../bm_algorithm.hpp"

void bm_host_memcpy(benchmark::State& state) {
    int ts_size = state.range(0);
    tensor<float, 1> ts_src(ts_size);
    tensor<float, 1> ts_dst(ts_size);

    while (state.KeepRunning()) {
        memcpy(ts_dst.data(), ts_src.data(), sizeof(ts_src[0]) * ts_src.size());
        benchmark::DoNotOptimize(ts_src.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(float));
}

void bm_host_raw1f_copy(benchmark::State& state) {
    auto ts_size = state.range(0);
    tensor<float, 1> ts_src(ts_size);
    tensor<float, 1> ts_dst(ts_size);

    int total_size = ts_src.size();
    auto p_src = ts_src.data();
    auto p_dst = ts_dst.data();

    while (state.KeepRunning()) {
        for (int i = 0; i < total_size; ++i) {
            p_dst[i] = p_src[i];
        }

        benchmark::DoNotOptimize(p_dst);
    }

    state.SetBytesProcessed(state.iterations() * ts_dst.size() * sizeof(float));
    state.SetItemsProcessed(state.iterations() * ts_dst.size());
}

BENCHMARK(bm_host_memcpy)->Arg(100_M);
BENCHMARK(bm_host_raw1f_copy)->Arg(100_M);

auto bm_host_tensor1f_copy = bm_tensor_copy<tensor<float, 1>>;
auto bm_host_tensor2f_copy = bm_tensor_copy<tensor<float, 2>>;
BENCHMARK(bm_host_tensor1f_copy)->Arg(100_M);
BENCHMARK(bm_host_tensor2f_copy)->Arg(10_K);

template <typename tensor_type>
void bm_tensor_array_index_copy(benchmark::State& state) {
    int ts_size = state.range(0);
    constexpr int_t rank = tensor_type::rank;
    pointi<rank> shape;
    fill(shape, ts_size);

    tensor_type ts_src(shape);
    tensor_type ts_dst(shape);

    while (state.KeepRunning()) {
        for_index(shape,
                  [ts_src, ts_dst] __matazure__(pointi<rank> idx) { ts_dst(idx) = ts_src(idx); });
        // cuda::copy(ts_src, ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(ts_src[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()));
}

auto bm_host_tensor1f_array_index_copy = bm_tensor_array_index_copy<tensor<float, 1>>;
auto bm_host_tensor2f_array_index_copy = bm_tensor_array_index_copy<tensor<float, 2>>;
BENCHMARK(bm_host_tensor1f_array_index_copy)->Arg(100_M);
BENCHMARK(bm_host_tensor2f_array_index_copy)->Arg(10_K);

BENCHMARK_MAIN();

// int main() { return 0; }
