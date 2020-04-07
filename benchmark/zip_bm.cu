#include <benchmark/benchmark.h>
#include <matazure/bm_config.hpp>
#include <mtensor.hpp>

using namespace matazure;

template <typename _ValueType>
void bm_gold_host_zip(benchmark::State& state) {
    tensor<_ValueType, 1> ts0(state.range(0));
    tensor<_ValueType, 1> ts1(ts0.shape());
    tensor<_ValueType, 1> ts_re0(ts0.shape());
    tensor<_ValueType, 1> ts_re1(ts0.shape());

    while (state.KeepRunning()) {
        for (int_t i = 0; i < ts0.size(); ++i) {
            ts_re0[i] = ts0[i];
            ts_re1[i] = ts1[i];
        }
    }

    auto bytes_size = static_cast<size_t>(ts0.size()) * sizeof(_ValueType);
    state.SetBytesProcessed(state.iterations() * bytes_size * 4);
}

template <typename _Tensor>
void bm_zip(benchmark::State& state) {
    _Tensor ts0(state.range(0));
    _Tensor ts1(ts0.shape());
    _Tensor ts_re0(ts0.shape());
    _Tensor ts_re1(ts0.shape());

    while (state.KeepRunning()) {
        auto ts_zip = zip(ts0, ts1);
        auto ts_re_zip = zip(ts_re0, ts_re1);
        copy(ts_zip, ts_re_zip);
    }

    auto bytes_size = static_cast<size_t>(ts0.size()) * sizeof(decltype(ts0[0]));
    state.SetBytesProcessed(state.iterations() * bytes_size * 4);
}

// template <typename _ValueType>
//__global__ void zip_dim2_gold_kenel(_ValueType *p_dst, _ValueType *p1, _ValueType *p2, int_t
// count){ 	for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x *
// gridDim.x) { 		p_dst[i] = p1[i] * p2[i];
//	}
//}
//
// template <typename _ValueType>
// void bm_gold_cu_zip(benchmark::State& state) {
//	cuda::tensor<_ValueType, 1> ts1(state.range(0));
//	cuda::tensor<_ValueType, 1> ts2(state.range(0));
//	fill(ts1, _ValueType(1));
//	fill(ts2, _ValueType(1));
//
//	while (state.KeepRunning()) {
//		cuda::tensor<float, 1> ts_re(ts1.shape());
//		cuda::default_execution_policy policy;
//		cuda::assert_runtime_success(cuda::configure_grid(policy,
// tensor_operation_gold_kenel<_ValueType>)); 		tensor_operation_gold_kenel<<<
// policy.grid_dim(), 			policy.block_dim(),
//			policy.shared_mem_bytes(),
//			policy.stream() >>>(ts_re.data(), ts1.data(), ts2.data(), ts_re.size());
//	}
//
//	auto bytes_size = static_cast<size_t>(ts1.size()) * sizeof(_ValueType);
//	state.SetBytesProcessed(state.iterations() * 2 * bytes_size);
//}
//
// template <typename _ValueType>
// void bm_zip_operation(benchmark::State &state) {
//	cuda::tensor<_ValueType, 1> ts1(state.range(0));
//	cuda::tensor<_ValueType, 1> ts2(state.range(0));
//	fill(ts1, _ValueType(1));
//	fill(ts2, _ValueType(1));
//
//	while (state.KeepRunning()) {
//		auto tsf_re = (ts1 * ts2 / ts1 + ts2).persist();
//	}
//
//	auto bytes_size = static_cast<size_t>(ts1.size()) * sizeof(decltype(ts1[0]));
//	state.SetBytesProcessed(2 * state.iterations() * bytes_size);
//}

BENCHMARK_TEMPLATE(bm_gold_host_zip, float)
    ->UseRealTime()
    ->Range(1 << 16, bm_config::max_shape<float, 1>());

auto bm_host_zip_byte = bm_zip<tensor<float, 1>>;
BENCHMARK(bm_host_zip_byte)->UseRealTime()->Range(1 << 16, bm_config::max_shape<float, 1>());

BENCHMARK_TEMPLATE(bm_gold_host_zip, float)
    ->UseRealTime()
    ->Range(1 << 16, bm_config::max_shape<float, 1>());
