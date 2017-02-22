#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

//stride benchmark

//template <typename _ValueType>
//void BM_host_stride_dim2_gold(benchmark::State &state) {
//	tensor<_ValueType, 2> ts(state.range(1), state.range(1));
//	int_t stride = state.range(0);
//	int_t phase = stride / 2;
//	auto ts_re_ext = ts.extent() / stride;
//
//	while (state.KeepRunning()) {
//		tensor<_ValueType, 2> ts_re(ts_re_ext);
//		auto ts_ext = ts.extent();
//		for (int_t j = 0; j < ts_re_ext[1]; ++j) {
//			for (int_t i = 0; i < ts_re_ext[0]; ++i) {
//				pointi<2> idx = { i,j };
//				ts_re(idx) = ts(idx * stride + phase);
//			}
//		}
//	}
//
//	auto bytes_size = static_cast<size_t>(prod(ts_re_ext)) * sizeof(_ValueType);
//	state.SetBytesProcessed(state.iterations() * bytes_size);
//}

template <typename _ValueType>
void BM_host_stride_dim2_gold(benchmark::State &state) {
	tensor<_ValueType, 2> ts(state.range(1), state.range(1));
	int_t stride = state.range(0);
	int_t phase = stride / 2;
	auto ts_re_ext = ts.extent() / stride;
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

	auto bytes_size = static_cast<size_t>(prod(ts_re_ext)) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

template <typename _Tensor>
void BM_stride(benchmark::State &state) {
	auto ext = pointi<_Tensor::dim>::zeros();
	//fill(ext, state.range(1));
	for (int_t i = 0; i < ext.size(); ++i) {
		ext[i] = state.range(1);
	}
	_Tensor ts(ext);
	int_t ts_stride = state.range(0);
	auto ts_re_ext = ts.extent() / ts_stride;
	_Tensor ts_re(ts_re_ext);
	
	while (state.KeepRunning()) {
		auto lts_re = stride(ts, ts_stride);
		copy(lts_re, ts_re);

	#ifdef MATAZURE_CUDA
		cuda::barrier();
	#endif

	}

	auto bytes_size = static_cast<size_t>(prod(ts_re_ext)) * sizeof(decltype(ts[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

//template <typename _ValueType>
//__global__ void stride_dim1_gold_kenel(_ValueType *p_dst, _ValueType *p_src, int_t count, int_t stride){
//	for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
//		p_dst[i] = p_dst[i * stride];
//	}
//}

//template <typename _ValueType>
//void BM_cu_stride_gold(benchmark::State& state) {
//	cu_tensor<_ValueType, 1> ts(state.range(0));
//
//	while (state.KeepRunning()) {
//		cu_tensor<float, 1> ts_re(ts1.extent());
//		cuda::ExecutionPolicy policy;
//		cuda::throw_on_error(cuda::condigure_grid(policy, tensor_operation_gold_kenel<_ValueType>));
//		tensor_operation_gold_kenel<<< policy.getGridSize(),
//			policy.getBlockSize(),
//			policy.getSharedMemBytes(),
//			policy.getStream() >>>(ts_re.data(), ts1.data(), ts2.data(), ts_re.size());
//	}
//
//	auto bytes_size = static_cast<size_t>(ts1.size()) * sizeof(_ValueType);
//	state.SetBytesProcessed(state.iterations() * 2 * bytes_size);
//}
//
//template <typename _ValueType>
//void BM_stride_operation(benchmark::State &state) {
//	cu_tensor<_ValueType, 1> ts1(state.range(0));
//	cu_tensor<_ValueType, 1> ts2(state.range(0));
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

BENCHMARK_TEMPLATE(BM_host_stride_dim2_gold, byte)->UseRealTime()->Apply(custom_arguments);
//BENCHMARK_TEMPLATE(BM_host_stride_dim2_gold, int16_t)->UseRealTime()->Apply(custom_arguments);
//BENCHMARK_TEMPLATE(BM_host_stride_dim2_gold, int32_t)->UseRealTime()->Apply(custom_arguments);
//BENCHMARK_TEMPLATE(BM_host_stride_dim2_gold, int64_t)->UseRealTime()->Apply(custom_arguments);
//BENCHMARK_TEMPLATE(BM_host_stride_dim2_gold, float)->UseRealTime()->Apply(custom_arguments);
//BENCHMARK_TEMPLATE(BM_host_stride_dim2_gold, double)->UseRealTime()->Apply(custom_arguments);

auto BM_stride_tensor_byte_dim2 = BM_stride<tensor<byte, 2>>;
BENCHMARK(BM_stride_tensor_byte_dim2)->UseRealTime()->Apply(custom_arguments);

auto BM_stride_cu_tensor_byte_dim2 = BM_stride<cu_tensor<byte, 2>>;
BENCHMARK(BM_stride_cu_tensor_byte_dim2)->UseRealTime()->Apply(custom_arguments);

//auto BM_stride_cu_tensor_byte_dim1 = BM_stride<cu_tensor<byte, 1>>;
//BENCHMARK(BM_stride_cu_tensor_byte_dim1)->UseRealTime()->Apply(custom_arguments);


//BENCHMARK_TEMPLATE(BM_host_stride, float)->Range(1 << 10, 1 << 28)->UseRealTime();
//
//BENCHMARK_TEMPLATE(BM_cu_stride_gold, float)->Range(1 << 10, 1 << 28)->UseRealTime();
//BENCHMARK_TEMPLATE(BM_hcu_stride, float)->Range(1 << 10, 1 << 28)->UseRealTime();
