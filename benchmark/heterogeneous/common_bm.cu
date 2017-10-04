#include <bm_config.hpp>

template <typename _Tensor>
void bm_stride(benchmark::State &state) {
	auto ext = pointi<_Tensor::rank>::zeros();
	fill(ext, state.range(1));
	_Tensor ts(ext);
	int_t ts_stride = state.range(0);
	auto ts_re_ext = ts.shape() / ts_stride;
	_Tensor ts_re(ts_re_ext);

	while (state.KeepRunning()) {
		auto lts_re = stride(ts, ts_stride);
		copy(lts_re, ts_re);

	#ifdef USE_CUDA
		cuda::device_synchronize();
	#endif

		benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(ts_re.size()) * sizeof(decltype(ts[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size);
}

static void custom_arguments(benchmark::internal::Benchmark* b) {
	for_index(pointi<2>::zeros(), pointi<2>{8, 8}, [=](pointi<2> idx) {
		b->Args({ idx[0] + 1, 1 << (idx[1] + 7) });
	});
}

auto bm_hete_tensor_stride_byte_dim2 = bm_stride<tensor<byte, 2>>;
BENCHMARK(bm_hete_tensor_stride_byte_dim2)->UseRealTime()->Apply(custom_arguments);
