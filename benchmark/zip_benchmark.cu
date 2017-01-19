#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;


template <typename _ValueType>
void BM_host_zip_gold(benchmark::State &state) {
	tensor<_ValueType, 1> ts0(state.range(0));
	tensor<_ValueType, 1> ts1(ts0.extent());
	tensor<_ValueType, 1> ts_re0(ts0.extent());
	tensor<_ValueType, 1> ts_re1(ts0.extent());

	while (state.KeepRunning()) {
		for (int_t i = 0; i < ts0.size(); ++i) {
			ts_re0[i] = ts0[i];
			ts_re1[i] = ts1[i];
		}
	}

	auto bytes_size = static_cast<size_t>(ts0.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 4);
}

template <typename _Tensor>
void BM_zip(benchmark::State &state) {
	_Tensor ts0(state.range(0));
	_Tensor ts1(ts0.extent());
	_Tensor ts_re0(ts0.extent());
	_Tensor ts_re1(ts0.extent());

	while (state.KeepRunning()) {
		auto ts_zip = zip(ts0, ts1);
		auto ts_re_zip = zip(ts_re0, ts_re1);
		copy(ts_zip, ts_re_zip);
	}

	auto bytes_size = static_cast<size_t>(ts0.size()) * sizeof(decltype(ts0[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size * 4);
}

//template <typename _ValueType>
//__global__ void zip_dim2_gold_kenel(_ValueType *p_dst, _ValueType *p1, _ValueType *p2, int_t count){
//	for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
//		p_dst[i] = p1[i] * p2[i];
//	}
//}
//
//template <typename _ValueType>
//void BM_cu_zip_gold(benchmark::State& state) {
//	cu_tensor<_ValueType, 1> ts1(state.range(0));
//	cu_tensor<_ValueType, 1> ts2(state.range(0));
//	fill(ts1, _ValueType(1));
//	fill(ts2, _ValueType(1));
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
//void BM_zip_operation(benchmark::State &state) {
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


BENCHMARK_TEMPLATE(BM_host_zip_gold, float)->UseRealTime()->Range(1 << 16, 1 << 28);

auto BM_host_zip_byte = BM_zip<tensor<float, 1>>;
BENCHMARK(BM_host_zip_byte)->UseRealTime()->Range(1 << 16, 1 << 28);

BENCHMARK_TEMPLATE(BM_host_zip_gold, float)->UseRealTime()->Range(1 << 16, 1 << 28);
