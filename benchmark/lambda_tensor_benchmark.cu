#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

static void BM_linear_lambda_tensor_persist_gold(benchmark::State &st) {
	tensor<float, 1> tsf1(st.range(0));
	while (st.KeepRunning()) {
		tensor<float, 1> ts_re(tsf1.extent());
		for (int_t i = 0; i < ts_re.size(); ++i) {
			ts_re[i] = 2.0f * tsf1[i];
		}
	}

	auto bytes_size = static_cast<size_t>(tsf1.size()) * sizeof(decltype(tsf1[0]));
	st.SetBytesProcessed(st.iterations() * bytes_size);
}

static void BM_linear_lambda_tensor_persist(benchmark::State &st) {
	tensor<float, 1> tsf1(st.range(0));
	while (st.KeepRunning()) {
		auto tsf1_re = make_lambda(tsf1.extent(), [tsf1](int_t i) {
			return 2.0f * tsf1[i];
		}).persist();
	}

	auto bytes_size = static_cast<size_t>(tsf1.size()) * sizeof(decltype(tsf1[0]));
	st.SetBytesProcessed(st.iterations() * bytes_size);
}

BENCHMARK(BM_linear_lambda_tensor_persist_gold)->Range(1 << 10, 1 << 28)->UseRealTime();
BENCHMARK(BM_linear_lambda_tensor_persist)->Range(1 << 10, 1 << 28)->UseRealTime();
