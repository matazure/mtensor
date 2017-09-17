#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

template <typename _ValueType>
void bm_cu_tensor_construct_and_destruct(benchmark::State& state) {
	int_t size = 0;
	while (state.KeepRunning()) {
		//DoNotOptimize(cuda::tensor<_ValueType,1> ts(state.range(0)));
		//size = ts.size();
		benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(size) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(bm_cu_tensor_construct_and_destruct, byte)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_cu_tensor_construct_and_destruct, int32_t)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_cu_tensor_construct_and_destruct, float)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_cu_tensor_construct_and_destruct, double)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
