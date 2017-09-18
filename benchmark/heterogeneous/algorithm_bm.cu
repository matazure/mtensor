#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

#ifdef USE_CUDA

template <typename _ValueType>
__global__ void gold_fill_kernel(_ValueType *p_dst, int_t count, _ValueType v){
	for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
		p_dst[i] = v;
	}
}

template <typename _ValueType>
void bm_gold_cu_fill(benchmark::State& state) {
	cuda::tensor<_ValueType, 1> ts_src(state.range(0));

	while (state.KeepRunning()) {
		cuda::parallel_execution_policy policy;
		policy.total_size(ts_src.size());
		cuda::configure_grid(policy, gold_fill_kernel<_ValueType>);
		gold_fill_kernel<<< policy.grid_size(),
			policy.block_size(),
			policy.shared_mem_bytes(),
			policy.stream() >>>(ts_src.data(), ts_src.size(), zero<_ValueType>::value());

		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(bm_gold_cu_fill, byte)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_fill, int16_t)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_fill, int32_t)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_fill, int64_t)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_fill, float)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_fill, double)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();

#endif

#ifdef USE_CUDA

#endif

template <typename _ValueType>
void bm_hete_for_each(benchmark::State& state) {
	TENSOR<_ValueType, 1> ts_src(state.range(0));

	while (state.KeepRunning()) {
		fill(ts_src, zero<_ValueType>::value());
	#ifdef USE_CUDA
		cuda::device_synchronize();
	#endif
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(bm_hete_for_each, byte)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_hete_for_each, int16_t)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_hete_for_each, int32_t)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_hete_for_each, int64_t)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_hete_for_each, float)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_hete_for_each, double)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
