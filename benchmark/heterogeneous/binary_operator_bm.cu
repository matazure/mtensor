#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

#ifdef USE_CUDA

template <typename _ValueType>
__global__ void gold_tensor_rank1_mul_kenel(_ValueType *p_dst, _ValueType *p1, _ValueType *p2, int_t count){
	for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
		p_dst[i] = p1[i] * p2[i];
	}
}

template <typename _ValueType>
void bm_gold_cu_tensor_rank1_mul(benchmark::State& state) {
	cuda::tensor<_ValueType, 1> ts0(state.range(0));
	cuda::tensor<_ValueType, 1> ts1(state.range(0));
	fill(ts0, zero<_ValueType>::value());
	fill(ts1, zero<_ValueType>::value());

	while (state.KeepRunning()) {
		cuda::tensor<_ValueType, 1> ts_re(ts0.shape());
		cuda::execution_policy policy;
		cuda::configure_grid(policy, gold_tensor_rank1_mul_kenel<_ValueType>);
		gold_tensor_rank1_mul_kenel<<< policy.grid_size(),
			policy.block_size(),
			policy.shared_mem_bytes(),
			policy.stream() >>>(ts_re.data(), ts0.data(), ts1.data(), ts_re.size());
		cuda::device_synchronize();

		benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(ts0.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 3);
	state.SetItemsProcessed(state.iterations() * bytes_size);
}

#define BM_GOLD_CU_TENSOR_RANK1_MUL(ValueType) \
auto bm_gold_cu_tensor_##ValueType##_rank1_mul = bm_gold_cu_tensor_rank1_mul<ValueType>; \
BENCHMARK(bm_gold_cu_tensor_##ValueType##_rank1_mul)->RangeMultiplier(bm_config::range_multiplier<ValueType, 1, device_tag>())->Range(bm_config::min_shape<ValueType, 1, device_tag>(), bm_config::max_shape<ValueType, 1, device_tag>())->UseRealTime();

BM_GOLD_CU_TENSOR_RANK1_MUL(byte)
BM_GOLD_CU_TENSOR_RANK1_MUL(int16_t)
BM_GOLD_CU_TENSOR_RANK1_MUL(int32_t)
BM_GOLD_CU_TENSOR_RANK1_MUL(int64_t)
BM_GOLD_CU_TENSOR_RANK1_MUL(float)
BM_GOLD_CU_TENSOR_RANK1_MUL(double)
BM_GOLD_CU_TENSOR_RANK1_MUL(point3f)
BM_GOLD_CU_TENSOR_RANK1_MUL(point4f)

#endif

#ifdef USE_HOST

template <typename _ValueType>
void bm_gold_host_tensor_rank1_mul(benchmark::State &state) {
	tensor<_ValueType, 1> ts0(state.range(0));
	tensor<_ValueType, 1> ts1(state.range(0));
	fill(ts0, zero<_ValueType>::value());
	fill(ts1, zero<_ValueType>::value());

	while (state.KeepRunning()) {
		tensor<_ValueType, 1> ts_re(ts0.shape());
		for (int_t i = 0, size = ts_re.size(); i < size; ++i) {
			ts_re[i] = ts0[i] * ts1[i];
		}

		benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(ts0.size()) * sizeof(decltype(ts0[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size * 3);
	state.SetItemsProcessed(state.iterations() * bytes_size);
}

#define BM_GOLD_HOST_TENSOR_RANK1_MUL(ValueType) \
auto bm_gold_host_tensor_##ValueType##_rank1_mul = bm_gold_host_tensor_rank1_mul<ValueType>; \
BENCHMARK(bm_gold_host_tensor_##ValueType##_rank1_mul)->RangeMultiplier(bm_config::range_multiplier<ValueType, 1, host_tag>())->Range(bm_config::min_shape<ValueType, 1, host_tag>(), bm_config::max_shape<ValueType, 1, host_tag>())->UseRealTime();

BM_GOLD_HOST_TENSOR_RANK1_MUL(byte)
BM_GOLD_HOST_TENSOR_RANK1_MUL(int16_t)
BM_GOLD_HOST_TENSOR_RANK1_MUL(int32_t)
BM_GOLD_HOST_TENSOR_RANK1_MUL(int64_t)
BM_GOLD_HOST_TENSOR_RANK1_MUL(float)
BM_GOLD_HOST_TENSOR_RANK1_MUL(double)
BM_GOLD_HOST_TENSOR_RANK1_MUL(point3f)
BM_GOLD_HOST_TENSOR_RANK1_MUL(point4f)

#endif

#define BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(Name, Op)									\
template <typename _Tensor>																\
void bm_hete_tensor_##Name(benchmark::State &state) {									\
	_Tensor ts0(pointi<_Tensor::rank>::all(state.range(0)));							\
	_Tensor ts1(ts0.shape());															\
	fill(ts0, zero<typename _Tensor::value_type>::value());								\
	fill(ts1, zero<typename _Tensor::value_type>::value());								\
	decltype((ts0 Op ts1).persist()) ts_re(ts0.shape());								\
																						\
	while (state.KeepRunning()) {														\
	  	copy(ts0 Op ts1, ts_re);														\
		HETE_SYNCHRONIZE;																\
																						\
		benchmark::ClobberMemory();														\
	}																					\
																						\
	auto bytes_size = static_cast<size_t>(ts_re.size()) * sizeof(decltype(ts_re[0]));	\
	state.SetBytesProcessed(state.iterations() * bytes_size * 3);						\
	state.SetItemsProcessed(state.iterations() * bytes_size);							\
}

//Arithmetic
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(add, +)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(sub, -)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(mul, *)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(div, /)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(mod, %)
//Bit
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(left_shift, <<)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(right_shift, >>)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(bit_or, |)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(bit_and, &)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(bit_xor, ^)
//Logic
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(or , ||)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(and, &&)
//Compapre
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(gt, >)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(lt, <)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(ge, >=)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(le, <=)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(equal, ==)
BM_HETE_TENSOR_BINARY_OPERATOR_FUNC(not_equal, !=)

#define BM_HETE_TENSOR_BINARY_OPERATOR(Name, ValueType, Rank) \
auto bm_hete_tensor_##ValueType##_rank##Rank##_##Name = bm_hete_tensor_##Name<HETE_TENSOR<ValueType, Rank>>; \
BENCHMARK(bm_hete_tensor_##ValueType##_rank##Rank##_##Name)->RangeMultiplier(bm_config::range_multiplier<ValueType, Rank, HETE_TAG>())->Range(bm_config::min_shape<ValueType, Rank, HETE_TAG>(), bm_config::max_shape<ValueType, Rank, HETE_TAG>())->UseRealTime();

#define BM_HETE_TENSOR_RANK1234_BINARY_OPERATOR(Name, ValueType) \
BM_HETE_TENSOR_BINARY_OPERATOR(Name, ValueType, 1) \
BM_HETE_TENSOR_BINARY_OPERATOR(Name, ValueType, 2) \
BM_HETE_TENSOR_BINARY_OPERATOR(Name, ValueType, 3) \
BM_HETE_TENSOR_BINARY_OPERATOR(Name, ValueType, 4)

#define BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(Name) \
BM_HETE_TENSOR_RANK1234_BINARY_OPERATOR(Name, byte) \
BM_HETE_TENSOR_RANK1234_BINARY_OPERATOR(Name, int16_t) \
BM_HETE_TENSOR_RANK1234_BINARY_OPERATOR(Name, int32_t) \
BM_HETE_TENSOR_RANK1234_BINARY_OPERATOR(Name, int64_t)

#define BM_HETE_TENSOR_TYPES_RANK1234_BINARY_OPERATOR_FLOATING(Name) \
BM_HETE_TENSOR_RANK1234_BINARY_OPERATOR(Name, float) \
BM_HETE_TENSOR_RANK1234_BINARY_OPERATOR(Name, double) \
BM_HETE_TENSOR_RANK1234_BINARY_OPERATOR(Name, point3f) \
BM_HETE_TENSOR_RANK1234_BINARY_OPERATOR(Name, point4f)

//Arithmetic
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(add)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(sub)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(mul)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(div)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(mod)
//Bit
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(left_shift)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(right_shift)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(bit_or)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(bit_and)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(bit_xor)
//Logic
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(or)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(and)
//Compapre
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(gt)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(lt)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(ge)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(le)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(equal)
BM_HETE_TENSOR_INTEGRAL_TYPES_RANK1234_BINARY_OPERATOR(not_equal)

//Arithmetic
BM_HETE_TENSOR_TYPES_RANK1234_BINARY_OPERATOR_FLOATING(add)
BM_HETE_TENSOR_TYPES_RANK1234_BINARY_OPERATOR_FLOATING(sub)
BM_HETE_TENSOR_TYPES_RANK1234_BINARY_OPERATOR_FLOATING(mul)
BM_HETE_TENSOR_TYPES_RANK1234_BINARY_OPERATOR_FLOATING(div)
//Compapre
BM_HETE_TENSOR_TYPES_RANK1234_BINARY_OPERATOR_FLOATING(gt)
BM_HETE_TENSOR_TYPES_RANK1234_BINARY_OPERATOR_FLOATING(lt)
BM_HETE_TENSOR_TYPES_RANK1234_BINARY_OPERATOR_FLOATING(ge)
BM_HETE_TENSOR_TYPES_RANK1234_BINARY_OPERATOR_FLOATING(le)
BM_HETE_TENSOR_TYPES_RANK1234_BINARY_OPERATOR_FLOATING(equal)
BM_HETE_TENSOR_TYPES_RANK1234_BINARY_OPERATOR_FLOATING(not_equal)
