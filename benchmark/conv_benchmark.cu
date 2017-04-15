#include <benchmark/benchmark.h>
#include <matazure/tensor>
#include <matazure/cuda/puzzle/conv.hpp>

using namespace matazure;

__constant__ static_tensor<float, 3, 3> mask;

MATAZURE_PUZZEL_CONV_BLOCK(conv_block5x5, mask)

template <typename _ValueType>
void BM_cu_conv_block(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cu_tensor<_ValueType, 2> ts_src(ext);
	cu_tensor<_ValueType, 2> ts_re(ts_src.extent());

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block5x5<16,16>(ts_src, ts_re);
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}


BENCHMARK_TEMPLATE1(BM_cu_conv_block, float)->RangeMultiplier(2)->Range(128, 4096)->UseRealTime();
