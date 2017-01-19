#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

template <typename _ValueType>
static void BM_host_tensor_operation_gold(benchmark::State &st) {
	tensor<_ValueType, 1> tsf1(st.range(0));
	tensor<_ValueType, 1> tsf2(st.range(0));
	fill(tsf1, _ValueType(1));
	fill(tsf2, _ValueType(1));

	while (st.KeepRunning()) {
		tensor<float, 1> ts_re(tsf1.extent());
		for (int_t i = 0; i < ts_re.size(); ++i) {
			ts_re[i] = tsf1[i] / tsf2[i] + tsf1[i] * tsf2[i];
		}
	}

	auto bytes_size = static_cast<size_t>(tsf1.size()) * sizeof(decltype(tsf1[0]));
	st.SetBytesProcessed(st.iterations() * bytes_size);
}

template <typename _ValueType>
static void BM_tensor_operation(benchmark::State &st) {
	cu_tensor<_ValueType, 1> tsf1(st.range(0));
	cu_tensor<_ValueType, 1> tsf2(st.range(0));
	fill(tsf1, _ValueType(1));
	fill(tsf2, _ValueType(1));

	while (st.KeepRunning()) {
		auto tsf_re = (tsf1 * tsf2 / tsf1 + tsf2).persist();
	}

	auto bytes_size = static_cast<size_t>(tsf1.size()) * sizeof(decltype(tsf1[0]));
	st.SetBytesProcessed(2 * st.iterations() * bytes_size);
}


BENCHMARK_TEMPLATE(BM_host_tensor_operation_gold, float)->Range(1 << 10, 1 << 28)->UseRealTime();

BENCHMARK_TEMPLATE(BM_tensor_operation, float)->Range(1 << 10, 1 << 28)->UseRealTime();

BENCHMARK_MAIN()
