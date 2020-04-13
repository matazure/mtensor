#include "bm_config.hpp"

void bm_tensor_for_index_flops(benchmark::State& state) {
    pointi<2> shape{1000, 1000};
    // fill(shape, state.range(0));
    tensor<pointi<4>, 2> ts_src(shape);
    tensor<pointi<4>, 2> ts_dst(shape);

    while (state.KeepRunning()) {
        for_index(shape, [=](pointi<2> idx) {
            auto tmp = ts_src(idx);
            for (size_t i = 0; i < 1000; ++i) {
                tmp += tmp;
            }
            ts_dst(idx) = tmp;
        });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetItemsProcessed(state.iterations() * 1000000 * 1000);
}

BENCHMARK(bm_tensor_for_index_flops);

void bm_tensor_omp_for_index_flops(benchmark::State& state) {
    pointi<2> shape{1000, 1000};
    // fill(shape, state.range(0));
    tensor<pointi<4>, 2> ts_src(shape);
    tensor<pointi<4>, 2> ts_dst(shape);

    while (state.KeepRunning()) {
        for_index(omp_policy{}, shape, [=](pointi<2> idx) {
            auto tmp = ts_src(idx);
            for (size_t i = 0; i < 1000; ++i) {
                tmp += tmp;
            }
            ts_dst(idx) = tmp;
        });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetItemsProcessed(state.iterations() * 1000000 * 1000);
}

BENCHMARK(bm_tensor_omp_for_index_flops);
