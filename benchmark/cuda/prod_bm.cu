#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

template <typename _ValueType>
void bm_cu_prod(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));

	cuda::matrix<_ValueType> ts_lhs(ext);
	cuda::matrix<_ValueType> ts_rhs(ext);

	cuda::matrix<_ValueType> ts_re(ext);
	while (state.KeepRunning()) {
		copy(puzzle::prod_general(ts_lhs, ts_rhs), ts_re);
		cuda::device_synchronize();
	}

	state.SetBytesProcessed(ts_lhs.size() * sizeof(decltype(ts_lhs[0])) + ts_rhs.size() * sizeof(decltype(ts_rhs[0])) + ts_re.size() * sizeof(decltype(ts_re[0])));
	state.SetItemsProcessed(static_cast<size_t>(2 * ts_lhs.shape()[0] * ts_lhs.shape()[1]) * ts_rhs.shape()[1]);
}
BENCHMARK_TEMPLATE1(bm_cu_prod, float)->RangeMultiplier(2)->Range(bm_config::min_shape<float, 1>(), 1 << 12)->UseRealTime();
BENCHMARK_TEMPLATE1(bm_cu_prod, double)->RangeMultiplier(2)->Range(bm_config::min_shape<float, 1>(), 1 << 11)->UseRealTime();

template <typename _ValueType>
void bm_prod(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));

	matrix<_ValueType> ts_lhs(ext);
	matrix<_ValueType> ts_rhs(ext);

	matrix<_ValueType> ts_re(ext);
	while (state.KeepRunning()) {
		copy(puzzle::prod_general(ts_lhs, ts_rhs), ts_re);
	}

	state.SetBytesProcessed(ts_lhs.size() * sizeof(decltype(ts_lhs[0])) + ts_rhs.size() * sizeof(decltype(ts_rhs[0])) + ts_re.size() * sizeof(decltype(ts_re[0])));
	state.SetItemsProcessed(static_cast<size_t>(2 * ts_lhs.shape()[0] * ts_lhs.shape()[1]) * ts_rhs.shape()[1]);
}
BENCHMARK_TEMPLATE1(bm_prod, float)->RangeMultiplier(2)->Range(bm_config::min_shape<float, 1>(), bm_config::min_shape<float, 1>())->UseRealTime();
BENCHMARK_TEMPLATE1(bm_prod, double)->RangeMultiplier(2)->Range(bm_config::min_shape<float, 1>(), bm_config::min_shape<float, 1>())->UseRealTime();

template <typename _ValueType>
void bm_cu_prod_block(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));

	cuda::matrix<_ValueType> ts_lhs(ext);
	cuda::matrix<_ValueType> ts_rhs(ext);

	cuda::matrix<_ValueType> ts_re(ext);
	while (state.KeepRunning()) {
		cuda::puzzle::prod_block<32>(ts_lhs, ts_rhs, ts_re);
		cuda::device_synchronize();
	}

	state.SetBytesProcessed(ts_lhs.size() * sizeof(decltype(ts_lhs[0])) + ts_rhs.size() * sizeof(decltype(ts_rhs[0])) + ts_re.size() * sizeof(decltype(ts_re[0])));
	state.SetItemsProcessed(static_cast<size_t>(2 * ts_lhs.shape()[0] * ts_lhs.shape()[1]) * ts_rhs.shape()[1]);
}
BENCHMARK_TEMPLATE1(bm_cu_prod_block, float)->RangeMultiplier(2)->Range(bm_config::min_shape<float, 1>(), 1 << 12)->UseRealTime();
BENCHMARK_TEMPLATE1(bm_cu_prod_block, double)->RangeMultiplier(2)->Range(bm_config::min_shape<float, 1>(), 1 << 11)->UseRealTime();
