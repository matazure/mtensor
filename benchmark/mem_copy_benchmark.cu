#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

template <typename _TensorSrc, typename _TensorDst>
static void BM_mem_copy(benchmark::State& state) {
	_TensorSrc ts_src(state.range(0));
	_TensorDst ts_dst(ts_src.extent());

	while (state.KeepRunning()) {
		mem_copy(ts_src, ts_dst);
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(typename _TensorSrc::value_type);
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

auto BM_host2host_mem_copy = BM_mem_copy<tensor<byte, 1>, tensor<byte, 1>>;
auto BM_device2host_mem_copy = BM_mem_copy<cu_tensor<byte, 1>, tensor<byte, 1>>;
auto BM_host2device_mem_copy = BM_mem_copy<tensor<byte, 1>, cu_tensor<byte, 1>>;
auto BM_device2device_mem_copy = BM_mem_copy<cu_tensor<byte, 1>, cu_tensor<byte, 1>>;

BENCHMARK(BM_host2host_mem_copy)->Range(1<<16, 1 << 30)->UseRealTime();
BENCHMARK(BM_device2host_mem_copy)->Range(1<<16, 1 << 30)->UseRealTime();
BENCHMARK(BM_host2device_mem_copy)->Range(1<<16, 1 << 30)->UseRealTime();
BENCHMARK(BM_device2device_mem_copy)->Range(1<<16, 1 << 30)->UseRealTime();

BENCHMARK_MAIN()



