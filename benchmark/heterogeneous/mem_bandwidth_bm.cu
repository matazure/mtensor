#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

template <typename _TensorSrc, typename _TensorDst>
static void bm_mem_copy(benchmark::State& state) {
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


#ifdef USE_CUDA

auto bm_host2host_mem_copy = bm_mem_copy<tensor<byte, 1>, tensor<byte, 1>>;
BENCHMARK(bm_host2host_mem_copy)->Range(bm_config::min_shape<byte, 1>(), bm_config::max_shape<byte, 1>())->UseRealTime();

auto bm_device2host_mem_copy = bm_mem_copy<cuda::tensor<byte, 1>, tensor<byte, 1>>;
BENCHMARK(bm_device2host_mem_copy)->Range(bm_config::min_shape<byte, 1>(), bm_config::max_shape<byte, 1>())->UseRealTime();
auto bm_host2device_mem_copy = bm_mem_copy<tensor<byte, 1>, cuda::tensor<byte, 1>>;
BENCHMARK(bm_host2device_mem_copy)->Range(bm_config::min_shape<byte, 1>(), bm_config::max_shape<byte, 1>())->UseRealTime();
auto bm_device2device_mem_copy = bm_mem_copy<cuda::tensor<byte, 1>, cuda::tensor<byte, 1>>;
BENCHMARK(bm_device2device_mem_copy)->Range(bm_config::min_shape<byte, 1>(), bm_config::max_shape<byte, 1>())->UseRealTime();

#endif

#ifdef USE_HOST

auto bm_host2host_mem_copy = bm_mem_copy<tensor<byte, 1>, tensor<byte, 1>>;
BENCHMARK(bm_host2host_mem_copy)->Range(bm_config::min_shape<byte, 1>(), bm_config::max_shape<byte, 1>())->UseRealTime();

#endif
