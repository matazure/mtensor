#include <benchmark/benchmark.h>
#include <matazure/tensor>
#include <matazure/cuda/puzzle/conv.hpp>

using namespace matazure;

static_tensor<float,dim<3,  3>> host_mask;
__constant__ static_tensor<float,dim<3,  3>> mask;
MATAZURE_PUZZEL_CONV_GLOBAL(conv_global, mask)
MATAZURE_PUZZEL_CONV_BLOCK(conv_block, mask)
MATAZURE_PUZZEL_CONV_BLOCK_CRACK(conv_block_crack, mask)
MATAZURE_PUZZEL_CONV_BLOCK_OVERLAP(conv_block_overlap, mask)

template <typename _ValueType>
void BM_cu_conv_global(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::tensor<_ValueType, 2> ts_src(ext);
	cuda::tensor<_ValueType, 2> ts_re(ts_src.shape());
	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		copy(cuda::puzzle::conv_global(clamp_zero(ts_src)), ts_re);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE1(BM_cu_conv_global, float)->RangeMultiplier(2)->Range(1<<10, 1<<14)->UseRealTime();

template <typename _ValueType>
void BM_cu_conv_block(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::tensor<_ValueType, 2> ts_src(ext);
	cuda::tensor<_ValueType, 2> ts_re(ts_src.shape());
	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block<dim<16,16>>(clamp_zero(ts_src), ts_re);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE1(BM_cu_conv_block, float)->RangeMultiplier(2)->Range(1<<10, 1<<14)->UseRealTime();

template <typename _ValueType>
void BM_cu_conv_block_crack(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::tensor<_ValueType, 2> ts_src(ext);
	cuda::tensor<_ValueType, 2> ts_re(ts_src.shape());
	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block_crack<dim<16,16>>(ts_src, ts_re);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE1(BM_cu_conv_block_crack, float)->RangeMultiplier(2)->Range(1<<10, 1<<14)->UseRealTime();

template <typename _ValueType>
void BM_cu_conv_block_crack_with_block_tensor(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::block_tensor<_ValueType, dim<16,16>> ts_src(ext / 16);
	cuda::block_tensor<_ValueType, dim<16,16>> ts_dst(ext / 16);

	auto ts_src_view = global_view(ts_src);
	auto ts_dst_view = global_view(ts_dst);

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block_crack<dim<16,16>>(ts_src_view, ts_dst_view);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(static_tensor<_ValueType, dim<16,16>>);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE1(BM_cu_conv_block_crack_with_block_tensor, float)->RangeMultiplier(2)->Range(1<<10, 1<<14)->UseRealTime();

template <typename _ValueType>
void BM_cu_conv_block_overlap(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::tensor<_ValueType, 2> ts_src(ext);
	cuda::tensor<_ValueType, 2> ts_re(ts_src.shape());
	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block_overlap<dim<16,16>>(clamp_zero(ts_src), ts_re);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2 * 14 * 14 / 16 / 16);
}
BENCHMARK_TEMPLATE1(BM_cu_conv_block_overlap, float)->RangeMultiplier(2)->Range(1<<10, 1<<14)->UseRealTime();

template <typename _ValueType>
void BM_cu_conv_block_overlap_with_block_tensor(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::tensor<static_tensor<_ValueType, dim<16,16>>, 2> ts_src(ext / 16);
	cuda::tensor<static_tensor<_ValueType, dim<16,16>>, 2> ts_dst(ext / 16);

	auto ts_src_view = make_lambda(ext, [=] __matazure__ (pointi<2> idx)->float &{
		auto block_idx = idx / pointi<2>{16, 16};
		auto local_idx = idx % pointi<2>{16, 16};
		return ts_src[block_idx][local_idx];
	});

	auto ts_dst_view = make_lambda(ext, [=] __matazure__ (pointi<2> idx)->float &{
		auto block_idx = idx / pointi<2>{16, 16};
		auto local_idx = idx % pointi<2>{16, 16};
		return ts_dst[block_idx][local_idx];
	});

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block_overlap<dim<16,16>>(clamp_zero(ts_src_view), ts_dst_view);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(static_tensor<_ValueType, dim<16,16>>);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2 * 14 * 14 / 16 / 16);
}
BENCHMARK_TEMPLATE1(BM_cu_conv_block_overlap_with_block_tensor, float)->RangeMultiplier(2)->Range(1<<10, 1<<14)->UseRealTime();
