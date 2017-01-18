#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

void BM_cu_for_each(benchmark::State& state) {
	cu_tensor<byte, 1> ts_src(state.range(0));

	while (state.KeepRunning()) {
		for_each(ts_src, [] __matazure__(byte &e) {
			e = 1.0f;
		});
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

BENCHMARK(BM_cu_for_each)->Range(1<<10, 1 << 28)->UseRealTime();


BENCHMARK_MAIN()



