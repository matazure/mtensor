#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

template <typename _ValueType>
__global__ void for_each_gold_kenel(_ValueType *p_dst, int_t count){
	for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
		p_dst[i] = static_cast<_ValueType>(1);
	}
}

template <typename _ValueType>
void BM_cu_for_each_gold(benchmark::State& state) {
	cuda::tensor<_ValueType, 1> ts_src(state.range(0));

	while (state.KeepRunning()) {
		cuda::parallel_execution_policy policy;
		policy.total_size(ts_src.size());
		cuda::configure_grid(policy, for_each_gold_kenel<_ValueType>);
		for_each_gold_kenel<<< policy.grid_size(),
			policy.block_size(),
			policy.shared_mem_bytes(),
			policy.stream() >>>(ts_src.data(), ts_src.size());

		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

template <typename _ValueType>
void BM_cu_for_each(benchmark::State& state) {
	cuda::tensor<_ValueType, 1> ts_src(state.range(0));

	while (state.KeepRunning()) {
		cuda::for_each(ts_src, [] __matazure__ (_ValueType &e) {
			e = 1.0f;
		});

		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(BM_cu_for_each_gold, byte)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
// BENCHMARK_TEMPLATE1(BM_cu_for_each_gold, int16_t)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
// BENCHMARK_TEMPLATE1(BM_cu_for_each_gold, int32_t)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
// BENCHMARK_TEMPLATE1(BM_cu_for_each_gold, int64_t)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
// BENCHMARK_TEMPLATE1(BM_cu_for_each_gold, float)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
// BENCHMARK_TEMPLATE1(BM_cu_for_each_gold, double)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_for_each, byte)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_for_each, int16_t)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_for_each, int32_t)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_for_each, int64_t)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_for_each, float)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
BENCHMARK_TEMPLATE1(BM_cu_for_each, double)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
