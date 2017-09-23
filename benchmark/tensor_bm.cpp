#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

template <typename _ValueType>
void bm_tensor_construct_and_destruct(benchmark::State& state) {
	int_t size = 0;
	while (state.KeepRunning()) {
		tensor<_ValueType,1> ts(state.range(0));
		size = ts.size();
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	first_major_layout<3> tmfdas(pointi<3>{1,2,3});

	auto bytes_size = static_cast<size_t>(size) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK_TEMPLATE1(bm_tensor_construct_and_destruct, byte)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
// BENCHMARK_TEMPLATE1(bm_tensor_construct_and_destruct, int32_t)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
// BENCHMARK_TEMPLATE1(bm_tensor_construct_and_destruct, float)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
// BENCHMARK_TEMPLATE1(bm_tensor_construct_and_destruct, double)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();

template <typename _ValueType>
void bm_tensor_linear_index_glod0(benchmark::State& state){
	auto size = state.range(0);
	auto p_data = new _ValueType[size];
	while (state.KeepRunning()) {
		auto p_tmp = p_data;
		for (int_t i = 0; i < size; ++i) {
			*p_tmp = _ValueType(i);
			++p_tmp;
		}
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	delete[] p_data;
	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(size) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(bm_tensor_linear_index_glod0, byte)->Range(bm_config::min_shape<float, 1>(), 1 << (bm_config::max_memory_exponent()))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_linear_index_glod0, int)->Range(bm_config::min_shape<float, 1>(), 1 << (bm_config::max_memory_exponent() - 2 ))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_linear_index_glod0, float)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();

template <typename _ValueType>
void bm_tensor_linear_index_glod(benchmark::State& state){
	auto size = state.range(0);
	auto p_data = new _ValueType[size];
	while (state.KeepRunning()) {
		for (int_t i = 0; i < size; ++i) {
			p_data[i] = _ValueType(i);
		}
		//benchmark::ClobberMemory();
	}

	delete[] p_data;
	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(size) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(bm_tensor_linear_index_glod, byte)->Range(bm_config::min_shape<float, 1>(), 1 << (bm_config::max_memory_exponent()))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_linear_index_glod, int)->Range(bm_config::min_shape<float, 1>(), 1 << (bm_config::max_memory_exponent() - 2 ))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_linear_index_glod, float)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();

template <typename _TensorType>
void bm_tensor_linear_index(benchmark::State& state) {
	pointi<_TensorType::rank> ext{};
	fill(ext, state.range(0));
	_TensorType ts(ext);
	while (state.KeepRunning()) {
		for(int_t i = 0; i < ts.size(); ++i){
			ts[i] = typename _TensorType::value_type(i);
		}
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(typename _TensorType::value_type));
}
auto bm_tensor_byte_1d_linear_index = bm_tensor_linear_index<tensor<byte, 1>>;
BENCHMARK(bm_tensor_byte_1d_linear_index)->Range(bm_config::min_shape<float, 1>(), 1 << (bm_config::max_memory_exponent()))->UseRealTime();
auto bm_tensor_int_1d_linear_index = bm_tensor_linear_index<tensor<int, 1>>;
BENCHMARK(bm_tensor_int_1d_linear_index)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
auto bm_tensor_float_1d_linear_index = bm_tensor_linear_index<tensor<float, 1>>;
BENCHMARK(bm_tensor_float_1d_linear_index)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();

template <typename _ValueType>
void bm_tensor_1d_array_index(benchmark::State& state) {
	pointi<1> ext{};
	fill(ext, state.range(0));
	tensor<_ValueType, 1> ts(ext);
	while (state.KeepRunning()) {
		for(int_t i = 0; i < ts.size(); ++i){
			ts[i] = _ValueType(i);
		}
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(bm_tensor_1d_array_index, byte)->Range(bm_config::min_shape<float, 1>(), 1 << (bm_config::max_memory_exponent()))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_1d_array_index, int)->Range(bm_config::min_shape<float, 1>(), 1 << (bm_config::max_memory_exponent()))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_1d_array_index, float)->Range(bm_config::min_shape<float, 1>(), 1 << (bm_config::max_memory_exponent()))->UseRealTime();

template <typename _ValueType>
void bm_tensor_2d_array_index_gold(benchmark::State& state) {
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
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(bm_tensor_2d_array_index_gold, byte)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_2d_array_index_gold, int)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_2d_array_index_gold, float)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_memory_exponent() -2 ) / 2))->UseRealTime();

template <typename _ValueType>
void bm_tensor_2d_array_index(benchmark::State& state) {
	pointi<2> ext{};
	fill(ext, state.range(0));
	tensor<_ValueType, 2> ts(ext);
	while (state.KeepRunning()) {
		for (int_t j = 0; j < ts.shape()[1]; ++j){
			for(int_t i = 0; i < ts.shape()[0]; ++i){
				ts[pointi<2>{i, j}] = _ValueType(i);
			}
		}
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(bm_tensor_2d_array_index, byte)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_2d_array_index, int)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_2d_array_index, float)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_memory_exponent() -2 ) / 2))->UseRealTime();

template <typename _ValueType>
void bm_tensor_2d_last_marjor_array_index(benchmark::State& state) {
	pointi<2> ext{};
	fill(ext, state.range(0));
	tensor<_ValueType, 2, last_major_layout<2>> ts(ext);
	while (state.KeepRunning()) {
		for (int_t j = 0; j < ts.shape()[1]; ++j){
			for(int_t i = 0; i < ts.shape()[0]; ++i){
				ts[pointi<2>{j, i}] = _ValueType(i);
			}
		}
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(bm_tensor_2d_last_marjor_array_index, byte)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_2d_last_marjor_array_index, int)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_memory_exponent() - 2) / 2))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_2d_last_marjor_array_index, float)->RangeMultiplier(4)->Range(1<<5, 1 << ((bm_config::max_memory_exponent() -2 ) / 2))->UseRealTime();

template <typename _ValueType>
void bm_tensor_3d_array_index(benchmark::State& state) {
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
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts.size()) * sizeof(_ValueType));
}
BENCHMARK_TEMPLATE(bm_tensor_3d_array_index, byte)->RangeMultiplier(2)->Range(1<<4, 1 << ((bm_config::max_memory_exponent() - 2) / 3))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_3d_array_index, int)->RangeMultiplier(2)->Range(1<<4, 1 << ((bm_config::max_memory_exponent() - 2) / 3))->UseRealTime();
BENCHMARK_TEMPLATE(bm_tensor_3d_array_index, float)->RangeMultiplier(2)->Range(1<<4, 1 << ((bm_config::max_memory_exponent() -2 ) / 3))->UseRealTime();
