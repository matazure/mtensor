#include <benchmark/benchmark.h>
#include <bm_config.hpp>

#include <matazure/tensor>

#include <emmintrin.h>
#include <immintrin.h>

using namespace matazure;

// void bm_prod_gold(benchmark::State &state){
// 	pointi<2> ext;
// 	fill(ext, state.range(0));
// 	matrix<float> ts_lhs(ext);
// 	matrix<float, last_major_t> ts_rhs(ext);
// 	matrix<float> ts_output(pointi<2>{ts_lhs.shape()[0], ts_lhs.shape()[1]});
// 	fill(ts_lhs, 1.1f);
// 	fill(ts_rhs, 1.2f);
//
// 	while (state.KeepRunning()){
// 		for (int_t )
// 	}
//
// 	state.SetItemsProcessed(state.iterations() * ts_lhs.shape()[0] * ts_lhs.shape()[1] * ts_rhs.shape()[1]);
// }

void bm_prod(benchmark::State &state){
	matrix<float, last_major_t> ts_lhs(pointi<2>{state.range(0), state.range(1)});
	matrix<float> ts_rhs(pointi<2>{state.range(1), state.range(2)});
	matrix<float> ts_output(pointi<2>{ts_lhs.shape()[0], ts_rhs.shape()[1]});
	fill(ts_lhs, 1.1f);
	fill(ts_rhs, 1.2f);

	while (state.KeepRunning()){

		auto t = ts_lhs(66, 0);


		auto lts_re = numeric::prod_general(ts_lhs, ts_rhs);
		copy(lts_re, ts_output);
	}

	state.SetItemsProcessed(state.iterations() * ts_lhs.shape()[0] * ts_lhs.shape()[1] * ts_rhs.shape()[1]);
}
BENCHMARK(bm_prod)->Args({2048, 32, 2048})->UseRealTime();
