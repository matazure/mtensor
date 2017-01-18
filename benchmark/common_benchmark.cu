#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

static void BM_tensor_operation_gold(benchmark::State &st) {
	tensor<float, 1> tsf1(st.range(0));
	tensor<float, 1> tsf2(st.range(0));
	while (st.KeepRunning()) {
		tensor<float, 1> ts_re(tsf1.extent());
		for (int_t i = 0; i < ts_re.size(); ++i) {
			ts_re[i] = tsf1[i] / tsf2[i];
		}
	}

	auto bytes_size = static_cast<size_t>(tsf1.size()) * sizeof(decltype(tsf1[0]));
	st.SetBytesProcessed(st.iterations() * bytes_size);
}

static void BM_tensor_operation(benchmark::State &st) {
	tensor<float, 1> tsf1(st.range(0));
	tensor<float, 1> tsf2(st.range(0));
	while (st.KeepRunning()) {
		auto tsf_re = (tsf1 / tsf2).persist();
	}

	auto bytes_size = static_cast<size_t>(tsf1.size()) * sizeof(decltype(tsf1[0]));
	st.SetBytesProcessed(st.iterations() * bytes_size);
}

BENCHMARK(BM_tensor_operation_gold)->Range(1 << 10, 1 << 28)->UseRealTime();
BENCHMARK(BM_tensor_operation)->Range(1 << 10, 1 << 28)->UseRealTime();
BENCHMARK(BM_tensor_operation_gold)->Range(1 << 10, 1 << 28)->UseRealTime();

BENCHMARK_MAIN()



