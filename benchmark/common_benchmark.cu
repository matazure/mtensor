#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

//binary operation benchmark
template <typename _ValueType>
__global__ void tensor_operation_gold_kenel(_ValueType *p_dst, _ValueType *p1, _ValueType *p2, int_t count){
	for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
		p_dst[i] = p1[i] * p2[i] / p1[i] + p2[i];
	}
}

template <typename _ValueType>
void BM_cu_tensor_operation_gold(benchmark::State& state) {
	cu_tensor<_ValueType, 1> ts1(state.range(0));
	cu_tensor<_ValueType, 1> ts2(state.range(0));
	fill(ts1, _ValueType(1));
	fill(ts2, _ValueType(1));

	while (state.KeepRunning()) {
		cu_tensor<_ValueType, 1> ts_re(ts1.extent());
		cuda::execution_policy policy;
		cuda::configure_grid(policy, tensor_operation_gold_kenel<_ValueType>);
		tensor_operation_gold_kenel<<< policy.grid_size(),
			policy.block_size(),
			policy.shared_mem_bytes(),
			policy.stream() >>>(ts_re.data(), ts1.data(), ts2.data(), ts_re.size());

		cuda::barrier();
	}

	auto bytes_size = static_cast<size_t>(ts1.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 3);
}

template <typename _ValueType>
void BM_host_tensor_operation_gold(benchmark::State &st) {
	tensor<_ValueType, 1> ts1(st.range(0));
	tensor<_ValueType, 1> ts2(st.range(0));
	fill(ts1, _ValueType(1));
	fill(ts2, _ValueType(1));

	while (st.KeepRunning()) {
		tensor<_ValueType, 1> ts_re(ts1.extent());
		for (int_t i = 0; i < ts_re.size(); ++i) {
			ts_re[i] = ts1[i] * ts2[i] / ts1[i] + ts2[i];
		}
	}

	auto bytes_size = static_cast<size_t>(ts1.size()) * sizeof(decltype(ts1[0]));
	st.SetBytesProcessed(st.iterations() * bytes_size * 3);
}

template <typename _ValueType>
void BM_host_tensor_operation(benchmark::State &st) {
	tensor<_ValueType, 1> ts1(st.range(0));
	tensor<_ValueType, 1> ts2(st.range(0));
	fill(ts1, _ValueType(1));
	fill(ts2, _ValueType(1));

	while (st.KeepRunning()) {
		auto tsf_re = (ts1 * ts2 / ts1 + ts2).persist();
	}

	auto bytes_size = static_cast<size_t>(ts1.size()) * sizeof(decltype(ts1[0]));
	st.SetBytesProcessed(2 * st.iterations() * bytes_size * 3);
}

template <typename _ValueType>
void BM_cu_tensor_operation(benchmark::State &st) {
	cu_tensor<_ValueType, 1> ts1(st.range(0));
	cu_tensor<_ValueType, 1> ts2(st.range(0));
	fill(ts1, _ValueType(1));
	fill(ts2, _ValueType(1));

	while (st.KeepRunning()) {
		auto tsf_re = (ts1 * ts2 / ts1 + ts2).persist();
		cuda::barrier();
	}

	auto bytes_size = static_cast<size_t>(ts1.size()) * sizeof(decltype(ts1[0]));
	st.SetBytesProcessed(2 * st.iterations() * bytes_size * 3);
}

BENCHMARK_TEMPLATE(BM_host_tensor_operation_gold, float)->Range(1 << 10, 1 << 28)->UseRealTime();

BENCHMARK_TEMPLATE(BM_host_tensor_operation, float)->Range(1 << 10, 1 << 28)->UseRealTime();

BENCHMARK_TEMPLATE(BM_cu_tensor_operation_gold, float)->Range(1 << 10, 1 << 28)->UseRealTime();

BENCHMARK_TEMPLATE(BM_cu_tensor_operation, float)->Range(1 << 10, 1 << 28)->UseRealTime();
