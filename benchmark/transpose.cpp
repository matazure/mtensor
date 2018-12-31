#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

void bm_matrixf4x4_transpose(benchmark::State &state) {
	auto n = state.range(0);

	tensor<point<simd_vector<float, 4>, 4>, 1> ts_src(state.range(0));
	tensor<point<simd_vector<float, 4>, 4>, 1> ts_dst(state.range(0));

	while (state.KeepRunning()) {
		for (int i = 0; i < ts_src.size(); ++i) {
			ts_dst[i] = puzzle::transpose(ts_src[i]);
		}
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(state.iterations() * (16)* n);
	state.SetBytesProcessed(state.iterations() * 16 * 4 * n);
}
BENCHMARK(bm_matrixf4x4_transpose)
->Args({ 1024 * 1024 * 40 })
->Args({ 1024 * 1024 })
->Args({ 1024 * 512 })
->Args({ 1024 * 64 })
->Args({ 1024 * 16 })
->Args({ 1024 })
->Args({ 128 })
->Args({ 4 })
->UseRealTime();

void bm_stensorf4x4_transpose(benchmark::State &state) {
	auto n = state.range(0);

	tensor<static_tensor<float, dim<4, 4>>, 1> ts_src(state.range(0));
	tensor<static_tensor<float, dim<4, 4>>, 1> ts_dst(state.range(0));

	while (state.KeepRunning()) {
		for (int i = 0; i < ts_src.size() ;  ++i) {
			ts_dst[i] = puzzle::transpose(ts_src[i]);
		}
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(state.iterations() * (16)* n);
	state.SetBytesProcessed(state.iterations() * 16 * 4 * n);
}
BENCHMARK(bm_stensorf4x4_transpose)
->Args({ 4 })
->Args({ 128 })
->Args({ 1024 })
->Args({ 1024 * 16 })
->Args({ 1024 * 64 })
->Args({ 1024 * 512 })
->Args({ 1024 * 1024 })
->Args({ 1024 * 1024 * 2})
->Args({ 1024 * 1024 * 4 })
->UseRealTime();
