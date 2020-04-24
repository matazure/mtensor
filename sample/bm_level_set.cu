#include <benchmark/benchmark.h>
#include "sample_level_set.hpp"

using namespace matazure;

template <typename _Tensor>
void bm_level_set(benchmark::State& state) {
    typedef typename _Tensor::value_type value_type;
    static const int_t rank = _Tensor::rank;
    pointi<rank> shape;
    fill(shape, state.range(0));

    _Tensor ts_g(shape);
    _Tensor ts_phi(shape);

    int_t iters = 10;

    while (state.KeepRunning()) {
        auto ts_re = drlse_edge(ts_phi, ts_g, 5, 0.2, -3, 1.5, 1, iters);
        benchmark::DoNotOptimize(ts_re);
    }

    state.SetItemsProcessed(state.iterations() * ts_phi.size() * iters);
}

auto bm_host_tensor2f_level_set = bm_level_set<tensor<float, 2>>;
BENCHMARK(bm_host_tensor2f_level_set)->Arg(1000);

auto bm_cuda_tensor2f_level_set = bm_level_set<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor2f_level_set)->Arg(10000);

BENCHMARK_MAIN();