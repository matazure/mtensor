#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

template <typename _ValueType>
void bm_host_stride_dim2_gold(benchmark::State &state) {
	tensor<_ValueType, 2> ts(state.range(1), state.range(1));
	int_t stride = state.range(0);
	int_t phase = stride / 2;
	auto ts_re_ext = ts.shape() / stride;
	tensor<_ValueType, 2> ts_re(ts_re_ext);

	while (state.KeepRunning()) {
		int_t pos_i = 0;
		int_t pos_j = 0;
		for (int_t j = 0; j < ts_re_ext[1]; ++j) {
			pos_i = 0;
			for (int_t i = 0; i < ts_re_ext[0]; ++i) {
				ts_re(i, j) = ts(pos_i, pos_j);
				pos_i += stride;
			}
			pos_j += stride;
		}
	}

	auto bytes_size = static_cast<size_t>(ts_re.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

template <typename _Tensor>
void bm_stride(benchmark::State &state) {
	auto ext = pointi<_Tensor::rank>::zeros();
	fill(ext, state.range(1));
	_Tensor ts(ext);
	int_t ts_stride = state.range(0);
	auto ts_re_ext = ts.shape() / ts_stride;
	_Tensor ts_re(ts_re_ext);

	while (state.KeepRunning()) {
		auto lts_re = stride(ts, ts_stride);
		copy(lts_re, ts_re);

	#ifdef USE_CUDA
		cuda::device_synchronize();
	#endif

		benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(ts_re.size()) * sizeof(decltype(ts[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

//template <typename _ValueType>
//__global__ void stride_dim1_gold_kenel(_ValueType *p_dst, _ValueType *p_src, int_t count, int_t stride){
//	for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
//		p_dst[i] = p_dst[i * stride];
//	}
//}

//template <typename _ValueType>
//void bm_gold_cu_stride(benchmark::State& state) {
//	cuda::tensor<_ValueType, 1> ts(state.range(0));
//
//	while (state.KeepRunning()) {
//		cuda::tensor<float, 1> ts_re(ts1.shape());
//		cuda::execution_policy policy;
//		cuda::assert_runtime_success(cuda::configure_grid(policy, tensor_operation_gold_kenel<_ValueType>));
//		tensor_operation_gold_kenel<<< policy.grid_size(),
//			policy.block_size(),
//			policy.shared_mem_bytes(),
//			policy.stream() >>>(ts_re.data(), ts1.data(), ts2.data(), ts_re.size());
//	}
//
//	auto bytes_size = static_cast<size_t>(ts1.size()) * sizeof(_ValueType);
//	state.SetBytesProcessed(state.iterations() * 2 * bytes_size);
//}
//
//template <typename _ValueType>
//void bm_stride_operation(benchmark::State &state) {
//	cuda::tensor<_ValueType, 1> ts1(state.range(0));
//	cuda::tensor<_ValueType, 1> ts2(state.range(0));
//	fill(ts1, _ValueType(1));
//	fill(ts2, _ValueType(1));
//
//	while (state.KeepRunning()) {
//		auto tsf_re = (ts1 * ts2 / ts1 + ts2).persist();
//	}
//
//	auto bytes_size = static_cast<size_t>(ts1.size()) * sizeof(decltype(ts1[0]));
//	state.SetBytesProcessed(2 * state.iterations() * bytes_size);
//}
//

static void custom_arguments(benchmark::internal::Benchmark* b) {
	for_index(pointi<2>::zeros(), pointi<2>{8, 8}, [=](pointi<2> idx) {
		b->Args({ idx[0] + 1, 1 << (idx[1] + 7) });
	});
}

BENCHMARK_TEMPLATE(bm_host_stride_dim2_gold, byte)->UseRealTime()->Apply(custom_arguments);
//BENCHMARK_TEMPLATE(bm_host_stride_dim2_gold, int16_t)->UseRealTime()->Apply(custom_arguments);
//BENCHMARK_TEMPLATE(bm_host_stride_dim2_gold, int32_t)->UseRealTime()->Apply(custom_arguments);
//BENCHMARK_TEMPLATE(bm_host_stride_dim2_gold, int64_t)->UseRealTime()->Apply(custom_arguments);
//BENCHMARK_TEMPLATE(bm_host_stride_dim2_gold, float)->UseRealTime()->Apply(custom_arguments);
//BENCHMARK_TEMPLATE(bm_host_stride_dim2_gold, double)->UseRealTime()->Apply(custom_arguments);

#ifdef USE_HOST
auto bm_stride_tensor_byte_dim2 = bm_stride<tensor<byte, 2>>;
BENCHMARK(bm_stride_tensor_byte_dim2)->UseRealTime()->Apply(custom_arguments);
#endif

#ifdef USE_CUDA
auto bm_stride_cu_tensor_byte_dim2 = bm_stride<cuda::tensor<byte, 2>>;
BENCHMARK(bm_stride_cu_tensor_byte_dim2)->UseRealTime()->Apply(custom_arguments);
#endif
//auto bm_stride_cu_tensor_byte_dim1 = bm_stride<cuda::tensor<byte, 1>>;
//BENCHMARK(bm_stride_cu_tensor_byte_dim1)->UseRealTime()->Apply(custom_arguments);


//BENCHMARK_TEMPLATE(bm_host_stride, float)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
//
//BENCHMARK_TEMPLATE(bm_gold_cu_stride, float)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
//BENCHMARK_TEMPLATE(bm_hcu_stride, float)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
