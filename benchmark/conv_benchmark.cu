#include <benchmark/benchmark.h>
#include <matazure/tensor>
#include <matazure/cuda/puzzle/conv.hpp>

using namespace matazure;

__constant__ static_tensor<float, 3, 3> mask;

template <typename _ValueType>
void BM_cu_conv_global(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cu_tensor<_ValueType, 2> ts_src(ext);
	fill(mask, _ValueType(0));
	cu_tensor<_ValueType, 2> ts_re(ts_src.extent());

	while (state.KeepRunning()) {
		cuda::puzzle::conv<8,8>(ts_src, mask, ts_re);
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}


BENCHMARK_TEMPLATE1(BM_cu_conv_global, float)->RangeMultiplier(2)->Range(128, 4096)->UseRealTime();
