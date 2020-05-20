#include "bm_config.hpp"

template <int_t element_size>
void bm_tesnor1f_for_index_flops(benchmark::State& state) {
#ifdef __GNUC__
    //比32则好像会使用32, 测试机器支持avx
    typedef float simd_type __attribute__((vector_size(32)));
#else
    //无法确保pointf使用的是simd类型
    typedef pointf<8> simd_type;
#endif

    pointi<1> shape{100_K};
    tensor<point<simd_type, element_size>, 1> ts_src(shape);
    tensor<point<simd_type, element_size>, 1> ts_dst(shape);

    int_t iterations = 1000;

    while (state.KeepRunning()) {
        for_index(shape, [=](pointi<1> idx) {
            auto tmp = ts_src(idx);
            for (size_t i = 0; i < iterations; ++i) {
                tmp += tmp;
            }
            ts_dst(idx) = tmp;
        });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetItemsProcessed(state.iterations() * ts_src.size() * element_size * sizeof(simd_type) /
                            4 * iterations * 2);
}

template <int_t element_size>
void bm_tesnor1f_omp_for_index_flops(benchmark::State& state) {
#ifdef __GNUC__
    typedef float simd_type __attribute__((vector_size(32)));
#else
    typedef pointf<8> simd_type;
#endif

    pointi<1> shape{100_K};
    tensor<point<simd_type, element_size>, 1> ts_src(shape);
    tensor<point<simd_type, element_size>, 1> ts_dst(shape);

    int_t iterations = 1000;

    while (state.KeepRunning()) {
        for_index(omp_policy{}, shape, [=](pointi<1> idx) {
            auto tmp = ts_src(idx);
            for (size_t i = 0; i < iterations; ++i) {
                tmp += tmp;
            }
            ts_dst(idx) = tmp;
        });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetItemsProcessed(state.iterations() * ts_src.size() * element_size * sizeof(simd_type) /
                            4 * iterations * 2);
}

BENCHMARK_TEMPLATE1(bm_tesnor1f_for_index_flops, 1);
BENCHMARK_TEMPLATE1(bm_tesnor1f_for_index_flops, 2);
BENCHMARK_TEMPLATE1(bm_tesnor1f_for_index_flops, 3);
BENCHMARK_TEMPLATE1(bm_tesnor1f_for_index_flops, 4);
BENCHMARK_TEMPLATE1(bm_tesnor1f_for_index_flops, 5);
BENCHMARK_TEMPLATE1(bm_tesnor1f_for_index_flops, 6);
BENCHMARK_TEMPLATE1(bm_tesnor1f_for_index_flops, 7);
BENCHMARK_TEMPLATE1(bm_tesnor1f_for_index_flops, 8);
BENCHMARK_TEMPLATE1(bm_tesnor1f_for_index_flops, 9);

BENCHMARK_TEMPLATE1(bm_tesnor1f_omp_for_index_flops, 1);
BENCHMARK_TEMPLATE1(bm_tesnor1f_omp_for_index_flops, 2);
BENCHMARK_TEMPLATE1(bm_tesnor1f_omp_for_index_flops, 3);
BENCHMARK_TEMPLATE1(bm_tesnor1f_omp_for_index_flops, 4);
BENCHMARK_TEMPLATE1(bm_tesnor1f_omp_for_index_flops, 5);
BENCHMARK_TEMPLATE1(bm_tesnor1f_omp_for_index_flops, 6);
BENCHMARK_TEMPLATE1(bm_tesnor1f_omp_for_index_flops, 7);
BENCHMARK_TEMPLATE1(bm_tesnor1f_omp_for_index_flops, 8);
BENCHMARK_TEMPLATE1(bm_tesnor1f_omp_for_index_flops, 9);
