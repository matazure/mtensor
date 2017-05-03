#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

template <typename _ValueType>
void BM_tensor_construct_and_destruct(benchmark::State& state) {
	int_t size = 0;
	while (state.KeepRunning()) {
		tensor<_ValueType,1> ts(state.range(0));
		size = ts.size();
	}

	auto bytes_size = static_cast<size_t>(size) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

template <typename _ValueType>
void BM_cu_tensor_construct_and_destruct(benchmark::State& state) {
	int_t size = 0;
	while (state.KeepRunning()) {
		cuda::tensor<_ValueType,1> ts(state.range(0));
		size = ts.size();
	}

	auto bytes_size = static_cast<size_t>(size) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(BM_tensor_construct_and_destruct, byte)->Range(1<<10, 1 << 28)->UseRealTime();
BENCHMARK_TEMPLATE1(BM_tensor_construct_and_destruct, int32_t)->Range(1<<10, 1 << 28)->UseRealTime();
BENCHMARK_TEMPLATE1(BM_tensor_construct_and_destruct, float)->Range(1<<10, 1 << 28)->UseRealTime();
BENCHMARK_TEMPLATE1(BM_tensor_construct_and_destruct, double)->Range(1<<10, 1 << 28)->UseRealTime();

BENCHMARK_TEMPLATE1(BM_cu_tensor_construct_and_destruct, byte)->Range(1<<10, 1 << 28)->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_tensor_construct_and_destruct, int32_t)->Range(1<<10, 1 << 28)->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_tensor_construct_and_destruct, float)->Range(1<<10, 1 << 28)->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_tensor_construct_and_destruct, double)->Range(1<<10, 1 << 28)->UseRealTime();
