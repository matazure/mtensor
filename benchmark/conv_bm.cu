#include <benchmark/benchmark.h>
#include <matazure/tensor>
#include <matazure/cuda/puzzle/conv.hpp>

using namespace matazure;

typedef dim<3, 3> mask_dim;
__constant__ static_tensor<float, mask_dim> mask;
static_tensor<float, mask_dim> host_mask;

MATAZURE_CUDA_PUZZEL_CONV_GLOBAL(conv_global, mask)
MATAZURE_CUDA_PUZZEL_CONV_BLOCK(conv_block, mask)
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_CRACK(conv_block_crack, mask)
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_OVERLAP(conv_block_overlap, mask)

MATAZURE_CUDA_PUZZEL_CONV_BLOCK_ALIGNED(conv_block_aligned, mask)
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_CRACK_ALIGNED(conv_block_crack_aligned, mask)
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_OVERLAP_ALIGNED(conv_blocK_overlap_aligned, mask)

template <typename _ValueType>
void bm_cu_conv_global(benchmark::State& state) {
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
BENCHMARK_TEMPLATE1(bm_cu_conv_global, float)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();

typedef dim<16, 16> block_16x16;
typedef dim<32, 32> block_32x32;

template <typename _ValueType, typename _BlockDim>
void bm_cu_conv_block(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::tensor<_ValueType, 2> ts_src(ext);
	cuda::tensor<_ValueType, 2> ts_re(ts_src.shape());

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block<_BlockDim>(clamp_zero(ts_src), ts_re);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE2(bm_cu_conv_block, float, block_16x16)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();

template <typename _ValueType, typename _BlockDim>
void bm_cu_conv_block_crack(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::tensor<_ValueType, 2> ts_src(ext);
	cuda::tensor<_ValueType, 2> ts_re(ts_src.shape());

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block_crack<_BlockDim>(ts_src, ts_re);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE2(bm_cu_conv_block_crack, float, block_16x16)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();
BENCHMARK_TEMPLATE2(bm_cu_conv_block_crack, float, block_32x32)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();

template <typename _ValueType, typename _BlockDim>
void bm_cu_conv_block_crack_with_block_tensor(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	auto block_dim = meta::array_to_pointi(_BlockDim{});
	cuda::block_tensor<_ValueType, _BlockDim> ts_src(ext / block_dim);
	cuda::block_tensor<_ValueType, _BlockDim> ts_dst(ext / block_dim);

	auto ts_src_view = global_view(ts_src);
	auto ts_dst_view = global_view(ts_dst);

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block_crack<_BlockDim>(ts_src_view, ts_dst_view);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE2(bm_cu_conv_block_crack_with_block_tensor, float, block_16x16)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();
BENCHMARK_TEMPLATE2(bm_cu_conv_block_crack_with_block_tensor, float, block_32x32)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();

template <typename _ValueType, typename _BlockDim>
void bm_cu_conv_block_overlap(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::tensor<_ValueType, 2> ts_src(ext);
	cuda::tensor<_ValueType, 2> ts_re(ts_src.shape());

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block_overlap<_BlockDim>(clamp_zero(ts_src), ts_re);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE2(bm_cu_conv_block_overlap, float, block_16x16)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();
BENCHMARK_TEMPLATE2(bm_cu_conv_block_overlap, float, block_32x32)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();

template <typename _ValueType, typename _BlockDim>
void bm_cu_conv_block_overlap_with_block_tensor(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	auto block_dim = meta::array_to_pointi(_BlockDim{});
	cuda::block_tensor<_ValueType, _BlockDim> ts_src(ext / block_dim);
	cuda::block_tensor<_ValueType, _BlockDim> ts_dst(ext / block_dim);

	auto ts_src_view = global_view(ts_src);
	auto ts_dst_view = global_view(ts_dst);

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block_overlap<_BlockDim>(clamp_zero(ts_src_view), ts_dst_view);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE2(bm_cu_conv_block_overlap_with_block_tensor, float, block_16x16)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();
BENCHMARK_TEMPLATE2(bm_cu_conv_block_overlap_with_block_tensor, float, block_32x32)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();

template <typename _ValueType, typename _BlockDim>
void bm_cu_conv_block_aligned(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::tensor<_ValueType, 2> ts_src(ext);
	cuda::tensor<_ValueType, 2> ts_re(ts_src.shape());

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block_aligned<_BlockDim>(clamp_zero(ts_src), ts_re);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE2(bm_cu_conv_block_aligned, float, block_16x16)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();

template <typename _ValueType, typename _BlockDim>
void bm_cu_conv_block_crack_aligned(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::tensor<_ValueType, 2> ts_src(ext);
	cuda::tensor<_ValueType, 2> ts_re(ts_src.shape());

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block_crack_aligned<_BlockDim>(ts_src, ts_re);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE2(bm_cu_conv_block_crack_aligned, float, block_16x16)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();
BENCHMARK_TEMPLATE2(bm_cu_conv_block_crack_aligned, float, block_32x32)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();

template <typename _ValueType, typename _BlockDim>
void bm_cu_conv_block_crack_aligned_with_block_tensor(benchmark::State& state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	auto block_dim = meta::array_to_pointi(_BlockDim{});
	cuda::block_tensor<_ValueType, _BlockDim> ts_src(ext / block_dim);
	cuda::block_tensor<_ValueType, _BlockDim> ts_dst(ext / block_dim);

	auto ts_src_view = global_view(ts_src);
	auto ts_dst_view = global_view(ts_dst);

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_block_crack_aligned<_BlockDim>(ts_src_view, ts_dst_view);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE2(bm_cu_conv_block_crack_aligned_with_block_tensor, float, block_16x16)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();
BENCHMARK_TEMPLATE2(bm_cu_conv_block_crack_aligned_with_block_tensor, float, block_32x32)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();

template <typename _ValueType, typename _BlockDim>
void bm_cu_conv_block_overlap_aligned(benchmark::State& state) {
	auto valid_block_dim = meta::array_to_pointi(
			meta::add_c(meta::sub_c(_BlockDim{}, decltype(mask)::meta_shape_type{}), meta::int_t_c<1>{})
	);
	auto valid_size = state.range(0) - (state.range(0) % valid_block_dim[0]);
	pointi<2> ext;
	fill(ext, valid_size);
	cuda::tensor<_ValueType, 2> ts_src(ext);
	cuda::tensor<_ValueType, 2> ts_re(ts_src.shape());

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_blocK_overlap_aligned<_BlockDim>(clamp_zero(ts_src), ts_re);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE2(bm_cu_conv_block_overlap_aligned, float, block_16x16)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();
BENCHMARK_TEMPLATE2(bm_cu_conv_block_overlap_aligned, float, block_32x32)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();

template <typename _ValueType, typename _BlockDim>
void bm_cu_conv_block_overlap_aligned_with_block_tensor(benchmark::State& state) {
	auto meta_valid_block_dim = meta::add_c(meta::sub_c(_BlockDim{}, decltype(mask)::meta_shape_type{}), meta::int_t_c<1>{});
	auto valid_block_dim = meta::array_to_pointi(meta_valid_block_dim);
	pointi<2> ext;
	fill(ext, state.range(0));
	cuda::block_tensor<_ValueType, decltype(meta_valid_block_dim)> ts_src(ext / valid_block_dim);
	cuda::block_tensor<_ValueType, decltype(meta_valid_block_dim)> ts_dst(ext / valid_block_dim);

	auto ts_src_view = global_view(ts_src);
	auto ts_dst_view = global_view(ts_dst);

	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	while (state.KeepRunning()) {
		cuda::puzzle::conv_blocK_overlap_aligned<_BlockDim>(clamp_zero(ts_src_view), ts_dst_view);
		cuda::device_synchronize();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}
BENCHMARK_TEMPLATE2(bm_cu_conv_block_overlap_aligned_with_block_tensor, float, block_16x16)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();
BENCHMARK_TEMPLATE2(bm_cu_conv_block_overlap_aligned_with_block_tensor, float, block_32x32)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->UseRealTime();