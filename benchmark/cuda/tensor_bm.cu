#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

template <typename _ValueType>
void BM_cu_tensor_construct_and_destruct(benchmark::State& state) {
	int_t size = 0;
	while (state.KeepRunning()) {
		DoNotOptimize(cuda::tensor<_ValueType,1> ts(state.range(0)));
		size = ts.size();
		benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(size) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(BM_cu_tensor_construct_and_destruct, byte)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_tensor_construct_and_destruct, int32_t)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_tensor_construct_and_destruct, float)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_tensor_construct_and_destruct, double)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
