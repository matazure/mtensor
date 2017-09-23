#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

#ifdef USE_CUDA

template <typename _ValueType>
__global__ void gold_fill_rank1_kernel(_ValueType *p_dst, int_t count, _ValueType v){
	for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
		p_dst[i] = v;
	}
}

template <typename _ValueType>
void bm_gold_cu_fill_rank1(benchmark::State& state) {
	cuda::tensor<_ValueType, 1> ts_src(state.range(0));

	while (state.KeepRunning()) {
		cuda::parallel_execution_policy policy;
		policy.total_size(ts_src.size());
		cuda::configure_grid(policy, gold_fill_rank1_kernel<_ValueType>);
		gold_fill_rank1_kernel<<< policy.grid_size(),
			policy.block_size(),
			policy.shared_mem_bytes(),
			policy.stream() >>>(ts_src.data(), ts_src.size(), zero<_ValueType>::value());

		cuda::device_synchronize();

		benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(bm_gold_cu_fill_rank1, byte)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<byte, 1, device_tag>(), bm_config::max_shape<byte, 1, device_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_fill_rank1, int16_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<int16_t, 1, device_tag>(), bm_config::max_shape<int16_t, 1, device_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_fill_rank1, int32_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<int32_t, 1, device_tag>(), bm_config::max_shape<int32_t, 1, device_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_fill_rank1, int64_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<int64_t, 1, device_tag>(), bm_config::max_shape<int64_t, 1, device_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_fill_rank1, float)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<float, 1, device_tag>(), bm_config::max_shape<float, 1, device_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_fill_rank1, double)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<double, 1, device_tag>(), bm_config::max_shape<double, 1, device_tag>())->UseRealTime();

template <typename _ValueType>
__global__ void gold_copy_rank1_kernel(_ValueType *p_src, _ValueType *p_dst, int_t count){
	for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
		p_dst[i] = p_src[i];
	}
}

template <typename _ValueType>
void bm_gold_cu_copy_rank1(benchmark::State& state) {
	cuda::tensor<_ValueType, 1> ts_src(state.range(0));
	cuda::tensor<_ValueType, 1> ts_dst(ts_src.size());
	fill(ts_src, zero<_ValueType>::value());

	while (state.KeepRunning()) {
		cuda::parallel_execution_policy policy;
		policy.total_size(ts_src.size());
		cuda::configure_grid(policy, gold_copy_rank1_kernel<_ValueType>);
		gold_copy_rank1_kernel<<< policy.grid_size(),
			policy.block_size(),
			policy.shared_mem_bytes(),
			policy.stream() >>>(ts_src.data(), ts_dst.data(), ts_src.size());

		cuda::device_synchronize();

		benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(bm_gold_cu_copy_rank1, byte)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<byte, 1, device_tag>(), bm_config::max_shape<byte, 1, device_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_copy_rank1, int16_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<int16_t, 1, device_tag>(), bm_config::max_shape<int16_t, 1, device_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_copy_rank1, int32_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<int32_t, 1, device_tag>(), bm_config::max_shape<int32_t, 1, device_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_copy_rank1, int64_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<int64_t, 1, device_tag>(), bm_config::max_shape<int64_t, 1, device_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_copy_rank1, float)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<float, 1, device_tag>(), bm_config::max_shape<float, 1, device_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_cu_copy_rank1, double)->RangeMultiplier(bm_config::range_multiplier<byte, 1, device_tag>())->Range(bm_config::min_shape<double, 1, device_tag>(), bm_config::max_shape<double, 1, device_tag>())->UseRealTime();

#endif

#ifdef USE_HOST

template <typename _ValueType>
void bm_gold_host_fill_rank1(benchmark::State& state) {
	tensor<_ValueType, 1> ts_src(state.range(0));
	auto p_data = ts_src.data();
	auto size = ts_src.size();
	fill(ts_src, zero<_ValueType>::value());

	while (state.KeepRunning()) {
		for (int_t i = 0; i < size; ++i){
			p_data[i] = zero<_ValueType>::value();
		}
		benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(bm_gold_host_fill_rank1, byte)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<byte, 1, host_tag>(), bm_config::max_shape<byte, 1, host_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_host_fill_rank1, int16_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<int16_t, 1, host_tag>(), bm_config::max_shape<int16_t, 1, host_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_host_fill_rank1, int32_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<int32_t, 1, host_tag>(), bm_config::max_shape<int32_t, 1, host_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_host_fill_rank1, int64_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<int64_t, 1, host_tag>(), bm_config::max_shape<int64_t, 1, host_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_host_fill_rank1, float)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<float, 1, host_tag>(), bm_config::max_shape<float, 1, host_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_host_fill_rank1, double)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<double, 1, host_tag>(), bm_config::max_shape<double, 1, host_tag>())->UseRealTime();

template <typename _ValueType>
void bm_gold_host_copy_rank1(benchmark::State& state) {
	tensor<_ValueType, 1> ts_src(state.range(0));
	tensor<_ValueType, 1> ts_dst(ts_src.size());
	fill(ts_src, zero<_ValueType>::value());
	auto p_src = ts_src.data();
	auto p_dst = ts_dst.data();
	auto size = ts_src.size();

	while (state.KeepRunning()) {
		for (int_t i = 0; i < size; ++i){
			p_dst[i] = p_src[i];
		}
		benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(bm_gold_host_copy_rank1, byte)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<byte, 1, host_tag>(), bm_config::max_shape<byte, 1, host_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_host_copy_rank1, int16_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<int16_t, 1, host_tag>(), bm_config::max_shape<int16_t, 1, host_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_host_copy_rank1, int32_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<int32_t, 1, host_tag>(), bm_config::max_shape<int32_t, 1, host_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_host_copy_rank1, int64_t)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<int64_t, 1, host_tag>(), bm_config::max_shape<int64_t, 1, host_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_host_copy_rank1, float)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<float, 1, host_tag>(), bm_config::max_shape<float, 1, host_tag>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_gold_host_copy_rank1, double)->RangeMultiplier(bm_config::range_multiplier<byte, 1, host_tag>())->Range(bm_config::min_shape<double, 1, host_tag>(), bm_config::max_shape<double, 1, host_tag>())->UseRealTime();

#endif

template <typename _Tensor>
void bm_hete_tensor_fill(benchmark::State& state) {
	auto shape = pointi<_Tensor::rank>::all(state.range(0));
	_Tensor ts_src(shape);

	while (state.KeepRunning()) {
		fill(ts_src, zero<typename _Tensor::value_type>::value());
	#ifdef USE_CUDA
		cuda::device_synchronize();
	#endif
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

auto bm_hete_tensor_fill_byte_rank1 = bm_hete_tensor_fill<HETE_TENSOR<byte, 1>>;
auto bm_hete_tensor_fill_int16_rank1 = bm_hete_tensor_fill<HETE_TENSOR<int16_t, 1>>;
auto bm_hete_tensor_fill_int32_rank1 = bm_hete_tensor_fill<HETE_TENSOR<int32_t, 1>>;
auto bm_hete_tensor_fill_int64_rank1 = bm_hete_tensor_fill<HETE_TENSOR<int64_t, 1>>;
auto bm_hete_tensor_fill_float_rank1 = bm_hete_tensor_fill<HETE_TENSOR<float, 1>>;
auto bm_hete_tensor_fill_double_rank1 = bm_hete_tensor_fill<HETE_TENSOR<double, 1>>;
BENCHMARK(bm_hete_tensor_fill_byte_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<byte, 1, HETE_TAG>(), bm_config::max_shape<byte, 1, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_fill_int16_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<int16_t, 1, HETE_TAG>(), bm_config::max_shape<int16_t, 1, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_fill_int32_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<int32_t, 1, HETE_TAG>(), bm_config::max_shape<int32_t, 1, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_fill_int64_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<int64_t, 1, HETE_TAG>(), bm_config::max_shape<int64_t, 1, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_fill_float_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<float, 1, HETE_TAG>(), bm_config::max_shape<float, 1, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_fill_double_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<double, 1, HETE_TAG>(), bm_config::max_shape<double, 1, HETE_TAG>())->UseRealTime();

auto bm_hete_tensor_fill_float_rank2 = bm_hete_tensor_fill<HETE_TENSOR<float, 2>>;
auto bm_hete_tensor_fill_float_rank3 = bm_hete_tensor_fill<HETE_TENSOR<float, 3>>;
auto bm_hete_tensor_fill_float_rank4 = bm_hete_tensor_fill<HETE_TENSOR<float, 4>>;
BENCHMARK(bm_hete_tensor_fill_float_rank2)->RangeMultiplier(bm_config::range_multiplier<byte, 2, HETE_TAG>())->Range(bm_config::min_shape<byte, 2, HETE_TAG>(), bm_config::max_shape<byte, 2, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_fill_float_rank3)->RangeMultiplier(bm_config::range_multiplier<byte, 3, HETE_TAG>())->Range(bm_config::min_shape<byte, 3, HETE_TAG>(), bm_config::max_shape<byte, 3, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_fill_float_rank4)->RangeMultiplier(bm_config::range_multiplier<byte, 4, HETE_TAG>())->Range(bm_config::min_shape<byte, 4, HETE_TAG>(), bm_config::max_shape<byte, 4, HETE_TAG>())->UseRealTime();

auto bm_hete_tensor_fill_float_rank2_last_major_layout = bm_hete_tensor_fill<HETE_TENSOR<float, 2, last_major_layout<2>>>;
BENCHMARK(bm_hete_tensor_fill_float_rank2_last_major_layout)->RangeMultiplier(bm_config::range_multiplier<byte, 2, HETE_TAG>())->Range(bm_config::min_shape<float, 2, HETE_TAG>(), bm_config::max_shape<float, 2, HETE_TAG>())->UseRealTime();

template <typename _Tensor>
void bm_hete_tensor_copy(benchmark::State& state) {
	auto shape = pointi<_Tensor::rank>::all(state.range(0));
	_Tensor ts_src(shape);
	_Tensor ts_dst(shape);
	fill(ts_src, zero<typename _Tensor::value_type>::value());

	while (state.KeepRunning()) {
		copy(ts_src, ts_dst);
	#ifdef USE_CUDA
		cuda::device_synchronize();
	#endif
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

auto bm_hete_tensor_copy_byte_rank1 = bm_hete_tensor_copy<HETE_TENSOR<byte, 1>>;
auto bm_hete_tensor_copy_int16_rank1 = bm_hete_tensor_copy<HETE_TENSOR<int16_t, 1>>;
auto bm_hete_tensor_copy_int32_rank1 = bm_hete_tensor_copy<HETE_TENSOR<int32_t, 1>>;
auto bm_hete_tensor_copy_int64_rank1 = bm_hete_tensor_copy<HETE_TENSOR<int64_t, 1>>;
auto bm_hete_tensor_copy_float_rank1 = bm_hete_tensor_copy<HETE_TENSOR<float, 1>>;
auto bm_hete_tensor_copy_double_rank1 = bm_hete_tensor_copy<HETE_TENSOR<double, 1>>;
BENCHMARK(bm_hete_tensor_copy_byte_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<byte, 1, HETE_TAG>(), bm_config::max_shape<byte, 1, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_copy_int16_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<int16_t, 1, HETE_TAG>(), bm_config::max_shape<int16_t, 1, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_copy_int32_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<int32_t, 1, HETE_TAG>(), bm_config::max_shape<int32_t, 1, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_copy_int64_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<int64_t, 1, HETE_TAG>(), bm_config::max_shape<int64_t, 1, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_copy_float_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<float, 1, HETE_TAG>(), bm_config::max_shape<float, 1, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_copy_double_rank1)->RangeMultiplier(bm_config::range_multiplier<byte, 1, HETE_TAG>())->Range(bm_config::min_shape<double, 1, HETE_TAG>(), bm_config::max_shape<double, 1, HETE_TAG>())->UseRealTime();

auto bm_hete_tensor_copy_float_rank2 = bm_hete_tensor_copy<HETE_TENSOR<float, 2>>;
auto bm_hete_tensor_copy_float_rank3 = bm_hete_tensor_copy<HETE_TENSOR<float, 3>>;
auto bm_hete_tensor_copy_float_rank4 = bm_hete_tensor_copy<HETE_TENSOR<float, 4>>;
BENCHMARK(bm_hete_tensor_copy_float_rank2)->RangeMultiplier(bm_config::range_multiplier<byte, 2, HETE_TAG>())->Range(bm_config::min_shape<byte, 2, HETE_TAG>(), bm_config::max_shape<byte, 2, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_copy_float_rank3)->RangeMultiplier(bm_config::range_multiplier<byte, 3, HETE_TAG>())->Range(bm_config::min_shape<byte, 3, HETE_TAG>(), bm_config::max_shape<byte, 3, HETE_TAG>())->UseRealTime();
BENCHMARK(bm_hete_tensor_copy_float_rank4)->RangeMultiplier(bm_config::range_multiplier<byte, 4, HETE_TAG>())->Range(bm_config::min_shape<byte, 4, HETE_TAG>(), bm_config::max_shape<byte, 4, HETE_TAG>())->UseRealTime();

auto bm_hete_tensor_copy_float_rank2_last_major_layout = bm_hete_tensor_copy<HETE_TENSOR<float, 2, last_major_layout<2>>>;
BENCHMARK(bm_hete_tensor_copy_float_rank2_last_major_layout)->RangeMultiplier(bm_config::range_multiplier<byte, 2, HETE_TAG>())->Range(bm_config::min_shape<float, 2, HETE_TAG>(), bm_config::max_shape<float, 2, HETE_TAG>())->UseRealTime();
