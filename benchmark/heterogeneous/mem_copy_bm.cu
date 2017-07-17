#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

template <typename _ValueType>
__global__ void each_copy_gold_kenel(_ValueType *p_dst, _ValueType *p_src, int_t count){
	for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
		p_dst[i] = p_src[i];
	}
}

template <typename _Tensor>
void BM_each_copy_gold(benchmark::State& state) {
	_Tensor ts_src(state.range(0));
	_Tensor ts_dst(ts_src.shape());

	while (state.KeepRunning()) {
		cuda::execution_policy policy;
		cuda::configure_grid(policy, each_copy_gold_kenel<typename _Tensor::value_type>);
		each_copy_gold_kenel<<< policy.grid_size(),
			policy.block_size(),
			policy.shared_mem_bytes(),
			policy.stream() >>>(ts_dst.data(), ts_src.data(), ts_src.size());
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

template <typename _TensorSrc, typename _TensorDst>
static void BM_mem_copy(benchmark::State& state) {
	_TensorSrc ts_src(state.range(0));
	_TensorDst ts_dst(ts_src.shape());

	while (state.KeepRunning()) {
		mem_copy(ts_src, ts_dst);

	#ifdef MATAZURE_CUDA
		cuda::device_synchronize();
	#endif
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(typename _TensorSrc::value_type);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

auto BM_host2host_mem_copy = BM_mem_copy<tensor<byte, 1>, tensor<byte, 1>>;
BENCHMARK(BM_host2host_mem_copy)->Range(1 << 10, 1 << 28)->UseRealTime();

auto BM_device2host_mem_copy = BM_mem_copy<cuda::tensor<byte, 1>, tensor<byte, 1>>;
BENCHMARK(BM_device2host_mem_copy)->Range(1 << 10, 1 << 28)->UseRealTime();
auto BM_host2device_mem_copy = BM_mem_copy<tensor<byte, 1>, cuda::tensor<byte, 1>>;
BENCHMARK(BM_host2device_mem_copy)->Range(1 << 10, 1 << 28)->UseRealTime();
auto BM_device2device_mem_copy = BM_mem_copy<cuda::tensor<byte, 1>, cuda::tensor<byte, 1>>;
BENCHMARK(BM_device2device_mem_copy)->Range(1 << 10, 1 << 28)->UseRealTime();

auto BM_each_copy_gold_byte = BM_each_copy_gold<cuda::tensor<byte, 1>>;
BENCHMARK(BM_each_copy_gold_byte)->Range(1 << 10, 1 << 28)->UseRealTime();

auto BM_each_copy_gold_float = BM_each_copy_gold<cuda::tensor<float, 1>>;
BENCHMARK(BM_each_copy_gold_float)->Range(1 << 10, 1 << 28)->UseRealTime();
