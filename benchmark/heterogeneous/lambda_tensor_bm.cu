#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

#ifdef USE_CUDA

static void bm_gold_linear_lambda_tensor_persist(benchmark::State &st) {
	tensor<float, 1> tsf1(st.range(0));
	while (st.KeepRunning()) {
		tensor<float, 1> ts_re(tsf1.shape());
		for (int_t i = 0; i < ts_re.size(); ++i) {
			ts_re[i] = 2.0f * tsf1[i];
		}
	}

	auto bytes_size = static_cast<size_t>(tsf1.size()) * sizeof(decltype(tsf1[0]));
	st.SetBytesProcessed(st.iterations() * bytes_size);
}

static void bm_linear_lambda_tensor_persist(benchmark::State &st) {
	tensor<float, 1> tsf1(st.range(0));
	while (st.KeepRunning()) {
		auto tsf1_re = make_lambda(tsf1.shape(), [tsf1](int_t i) {
			return 2.0f * tsf1[i];
		}).persist();
	}

	auto bytes_size = static_cast<size_t>(tsf1.size()) * sizeof(decltype(tsf1[0]));
	st.SetBytesProcessed(st.iterations() * bytes_size);
}

BENCHMARK(bm_gold_linear_lambda_tensor_persist)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();
BENCHMARK(bm_linear_lambda_tensor_persist)->Range(bm_config::min_shape<float, 1>(), bm_config::max_shape<float, 1>())->UseRealTime();

#endif
