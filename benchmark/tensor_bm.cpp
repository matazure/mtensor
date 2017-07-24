#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

template <typename _ValueType>
void BM_tensor_construct_and_destruct(benchmark::State& state) {
	int_t size = 0;
	while (state.KeepRunning()) {
		tensor<_ValueType,1> ts(state.range(0));
		size = ts.size();
		//benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(size) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(BM_tensor_construct_and_destruct, byte)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
// BENCHMARK_TEMPLATE1(BM_tensor_construct_and_destruct, int32_t)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
// BENCHMARK_TEMPLATE1(BM_tensor_construct_and_destruct, float)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
// BENCHMARK_TEMPLATE1(BM_tensor_construct_and_destruct, double)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();

template <typename _ValueType>
void BM_tensor_linear_access_glod(benchmark::State& state){
	auto size = state.range(0);
	auto p_data = new _ValueType[size];
	while (state.KeepRunning()) {
		for (int_t i = 0; i < size; ++i) {
			p_data[i] = _ValueType(i);
		}
		//benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(size) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(BM_tensor_linear_access_glod, byte)->Range(1 << 10, 1 << (bm_config::max_host_memory_exponent()))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_linear_access_glod, int)->Range(1 << 10, 1 << (bm_config::max_host_memory_exponent() - 2 ))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_linear_access_glod, float)->Range(1 << 10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();

template <typename _TensorType>
void BM_tensor_linear_access(benchmark::State& state) {
	pointi<_TensorType::rank> ext{};
	fill(ext, state.range(0));
	_TensorType ts(ext);
	while (state.KeepRunning()) {
		for(int_t i = 0; i < ts.size(); ++i){
			ts[i] = typename _TensorType::value_type(i);
		}
		//benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(typename _TensorType::value_type));
}
auto BM_tensor_byte_1d_linear_access = BM_tensor_linear_access<tensor<byte, 1>>;
BENCHMARK(BM_tensor_byte_1d_linear_access)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent()))->UseRealTime();
auto BM_tensor_int_1d_linear_access = BM_tensor_linear_access<tensor<int, 1>>;
BENCHMARK(BM_tensor_int_1d_linear_access)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();
auto BM_tensor_float_1d_linear_access = BM_tensor_linear_access<tensor<float, 1>>;
BENCHMARK(BM_tensor_float_1d_linear_access)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent() - 2))->UseRealTime();

template <typename _ValueType>
void BM_tensor_1d_array_index(benchmark::State& state) {
	pointi<1> ext{};
	fill(ext, state.range(0));
	tensor<_ValueType, 1> ts(ext);
	while (state.KeepRunning()) {
		for(int_t i = 0; i < ts.size(); ++i){
			ts[i] = _ValueType(i);
		}
		//benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(BM_tensor_1d_array_index, byte)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent()))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_1d_array_index, int)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent()))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_1d_array_index, float)->Range(1<<10, 1 << (bm_config::max_host_memory_exponent()))->UseRealTime();

template <typename _ValueType>
void BM_tensor_2d_array_index_gold(benchmark::State& state) {
	pointi<2> ext{};
	fill(ext, state.range(0));
	tensor<_ValueType, 2> ts(ext);
	auto width = ts.shape()[0];
	auto p_data = ts.data();
	while (state.KeepRunning()) {
		for (int_t j = 0; j < ts.shape()[1]; ++j){
			for(int_t i = 0; i < ts.shape()[0]; ++i){
				p_data[i + j * width] = _ValueType(i);
			}
		}
		//benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(BM_tensor_2d_array_index_gold, byte)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_host_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_2d_array_index_gold, int)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_host_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_2d_array_index_gold, float)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_host_memory_exponent() -2 ) / 2))->UseRealTime();

template <typename _ValueType>
void BM_tensor_2d_array_index(benchmark::State& state) {
	pointi<2> ext{};
	fill(ext, state.range(0));
	tensor<_ValueType, 2> ts(ext);
	while (state.KeepRunning()) {
		for (int_t j = 0; j < ts.shape()[1]; ++j){
			for(int_t i = 0; i < ts.shape()[0]; ++i){
				ts[pointi<2>{i, j}] = _ValueType(i);
			}
		}
		//benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(BM_tensor_2d_array_index, byte)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_host_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_2d_array_index, int)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_host_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_2d_array_index, float)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_host_memory_exponent() -2 ) / 2))->UseRealTime();

template <typename _ValueType>
void BM_tensor_2d_last_marjor_array_index(benchmark::State& state) {
	pointi<2> ext{};
	fill(ext, state.range(0));
	tensor<_ValueType, 2, last_major_t> ts(ext);
	while (state.KeepRunning()) {
		for (int_t j = 0; j < ts.shape()[1]; ++j){
			for(int_t i = 0; i < ts.shape()[0]; ++i){
				ts[pointi<2>{j, i}] = _ValueType(i);
			}
		}
		//benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(BM_tensor_2d_last_marjor_array_index, byte)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_host_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_2d_last_marjor_array_index, int)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_host_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_2d_last_marjor_array_index, float)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_host_memory_exponent() -2 ) / 2))->UseRealTime();

template <typename _ValueType>
void BM_tensor_3d_array_index(benchmark::State& state) {
	pointi<3> ext{};
	fill(ext, state.range(0));
	tensor<_ValueType, 3> ts(ext);
	while (state.KeepRunning()) {
		for (int_t k = 0; k < ts.shape()[2]; ++k){
			for (int_t j = 0; j < ts.shape()[1]; ++j){
				for(int_t i = 0; i < ts.shape()[0]; ++i){
					ts[pointi<3>{i, j, k}] = _ValueType(i);
				}
			}
		}
		//benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(BM_tensor_3d_array_index, byte)->RangeMultiplier(2)->Range(1<<4, 1 << ((bm_config::max_host_memory_exponent() - 2) / 3))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_3d_array_index, int)->RangeMultiplier(2)->Range(1<<4, 1 << ((bm_config::max_host_memory_exponent() - 2) / 3))->UseRealTime();
BENCHMARK_TEMPLATE(BM_tensor_3d_array_index, float)->RangeMultiplier(2)->Range(1<<4, 1 << ((bm_config::max_host_memory_exponent() -2 ) / 3))->UseRealTime();
