#include "../bm_config.hpp"

static void bm_gold_tensor2f_copy(benchmark::State& state) {
    auto sides = state.range(0);
    tensor<float, 2> ts_src(pointi<2>{sides, sides});
    tensor<float, 2> ts_dst(pointi<2>{sides, sides});

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
}

void bm_la_tensor2f_copy(benchmark::State& state) {
    auto sides = state.range(0);
    tensor<float, 2> ts_src(pointi<2>{sides, sides});
    tensor<float, 2> ts_dst(pointi<2>{sides, sides});

    while (state.KeepRunning()) {
        copy(ts_src, ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * ts_dst.size() * sizeof(float));
}

void bm_aa_tensor2f_copy(benchmark::State& state) {
    auto sides = state.range(0);
    tensor<float, 2> ts_src(pointi<2>{sides, sides});
    tensor<float, 2> ts_dst(pointi<2>{sides, sides});

    int total_size = ts_src.size();

    while (state.KeepRunning()) {
        for_index(ts_dst.shape(), [&](pointi<2> index) { ts_dst(index) = ts_src(index); });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * ts_dst.size() * sizeof(float));
}

BENCHMARK(bm_gold_tensor2f_copy)->Arg(10_K);
BENCHMARK(bm_la_tensor2f_copy)->Arg(10_K);
BENCHMARK(bm_aa_tensor2f_copy)->Arg(10_K);

// BENCHMARK

BENCHMARK_MAIN();

// int main() { return 0; }
