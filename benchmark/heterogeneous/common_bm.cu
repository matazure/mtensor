#include <bm_config.hpp>

template <typename _Tensor, typename _ValueType>
void bm_hete_tensor_cast(benchmark::State &state) {
	auto ext = pointi<_Tensor::rank>::zeros();
	fill(ext, state.range(1));
	_Tensor ts_src(ext);
 	decltype(cast<_ValueType>(ts_src).persist()) ts_dst(ts_src.shape());

	while (state.KeepRunning()) {
		copy(cast<_ValueType>(ts_src), ts_dst);
		HETE_SYNCHRONIZE;

		benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * ts_dst.size() * (sizeof(decltype(ts_src[0])) + sizeof(decltype(ts_dst[0]))));
}

#define BM_HETE_TENSOR_CAST(ValueTypeSrc, ValueTypeDst, Rank) \
auto bm_hete_tensor_##ValueTypeSrc##_rank##Rank##_cast_##ValueTypeDst = bm_hete_tensor_cast<HETE_TENSOR<ValueTypeSrc, Rank>, ValueTypeDst>; \
BENCHMARK(bm_hete_tensor_##ValueTypeSrc##_rank##Rank##_cast_##ValueTypeDst)->RangeMultiplier(bm_config::range_multiplier<ValueTypeDst, Rank, HETE_TAG>())->Range(bm_config::min_shape<ValueTypeDst, Rank, HETE_TAG>(), bm_config::max_shape<ValueTypeDst, Rank, HETE_TAG>())->UseRealTime();

#define BM_HETE_TENSOR_CAST_RANK1234(ValueTypeSrc, ValueTypeDst) \
BM_HETE_TENSOR_CAST(ValueTypeSrc, ValueTypeDst, 1) \
BM_HETE_TENSOR_CAST(ValueTypeSrc, ValueTypeDst, 2) \
BM_HETE_TENSOR_CAST(ValueTypeSrc, ValueTypeDst, 3) \
BM_HETE_TENSOR_CAST(ValueTypeSrc, ValueTypeDst, 4)

BM_HETE_TENSOR_CAST_RANK1234(byte, float)
BM_HETE_TENSOR_CAST_RANK1234(int16_t, float);
BM_HETE_TENSOR_CAST_RANK1234(int32_t, float);
BM_HETE_TENSOR_CAST_RANK1234(int64_t, double);
BM_HETE_TENSOR_CAST_RANK1234(float, float);
BM_HETE_TENSOR_CAST_RANK1234(double, float);
BM_HETE_TENSOR_CAST_RANK1234(point3b, point3f);
BM_HETE_TENSOR_CAST_RANK1234(point4b, point4f);
BM_HETE_TENSOR_CAST_RANK1234(point3f, point3b);
BM_HETE_TENSOR_CAST_RANK1234(point4f, point4b);

template <typename _Tensor>
void bm_hete_tensor_section(benchmark::State &state) {
	auto ext = pointi<_Tensor::rank>::zeros();
	fill(ext, state.range(1));
	_Tensor ts_src(ext);
	auto center = ts_src.shape() / 4;
	auto dst_shape = ts_src.shape() / 2;
	decltype(section(ts_src, center, dst_shape).persist()) ts_dst(ts_src.shape());

	while (state.KeepRunning()) {
		copy(section(ts_src, center, dst_shape), ts_dst);
		HETE_SYNCHRONIZE;

		benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * ts_dst.size() * sizeof(typename _Tensor::value_type) * 2);
}

#define BM_HETE_TENSOR_SECTION(ValueType, Rank) \
auto bm_hete_tensor_##ValueType##_rank##Rank##_section = bm_hete_tensor_section<HETE_TENSOR<ValueType, Rank>>; \
BENCHMARK(bm_hete_tensor_##ValueType##_rank##Rank##_section)->RangeMultiplier(bm_config::range_multiplier<ValueType, Rank, HETE_TAG>())->Range(bm_config::min_shape<ValueType, Rank, HETE_TAG>(), bm_config::max_shape<ValueType, Rank, HETE_TAG>())->UseRealTime();

#define BM_HETE_TENSOR_SECTION_RANK1234(ValueType) \
BM_HETE_TENSOR_SECTION(ValueType, 1) \
BM_HETE_TENSOR_SECTION(ValueType, 2) \
BM_HETE_TENSOR_SECTION(ValueType, 3) \
BM_HETE_TENSOR_SECTION(ValueType, 4)

BM_HETE_TENSOR_SECTION_RANK1234(byte)
BM_HETE_TENSOR_SECTION_RANK1234(int16_t)
BM_HETE_TENSOR_SECTION_RANK1234(int32_t)
BM_HETE_TENSOR_SECTION_RANK1234(int64_t)
BM_HETE_TENSOR_SECTION_RANK1234(float)
BM_HETE_TENSOR_SECTION_RANK1234(double)
BM_HETE_TENSOR_SECTION_RANK1234(point3f)
BM_HETE_TENSOR_SECTION_RANK1234(point4f)

 template <typename _Tensor>
 void bm_hete_tensor_stride_step2(benchmark::State &state) {
 	auto ext = pointi<_Tensor::rank>::zeros();
 	fill(ext, state.range(0));
 	_Tensor ts_src(ext);
 	_Tensor ts_dst(ts_src.shape() / 2);

 	while (state.KeepRunning()) {
 		copy(stride(ts_src, 2), ts_dst);
 		HETE_SYNCHRONIZE;

 		benchmark::ClobberMemory();
 	}

 	state.SetBytesProcessed(state.iterations() * ts_dst.size() * sizeof(decltype(ts_dst[0])) * 2);
 }

 #define BM_HETE_TENSOR_STRIDE_STEP2(ValueType, Rank) \
 auto bm_hete_tensor_##ValueType##_rank##Rank##_stride_step2 = bm_hete_tensor_stride_step2<HETE_TENSOR<ValueType, Rank>>; \
 BENCHMARK(bm_hete_tensor_##ValueType##_rank##Rank##_stride_step2)->RangeMultiplier(bm_config::range_multiplier<ValueType, Rank, HETE_TAG>())->Range(bm_config::min_shape<ValueType, Rank, HETE_TAG>(), bm_config::max_shape<ValueType, Rank, HETE_TAG>())->UseRealTime();

 #define BM_HETE_TENSOR_STRIDE_STEP2_RANK1234(ValueType) \
 BM_HETE_TENSOR_STRIDE_STEP2(ValueType, 1) \
 BM_HETE_TENSOR_STRIDE_STEP2(ValueType, 2) \
 BM_HETE_TENSOR_STRIDE_STEP2(ValueType, 3) \
 BM_HETE_TENSOR_STRIDE_STEP2(ValueType, 4)

 BM_HETE_TENSOR_STRIDE_STEP2_RANK1234(byte)
 BM_HETE_TENSOR_STRIDE_STEP2_RANK1234(int16_t)
 BM_HETE_TENSOR_STRIDE_STEP2_RANK1234(int32_t)
 BM_HETE_TENSOR_STRIDE_STEP2_RANK1234(int64_t)
 BM_HETE_TENSOR_STRIDE_STEP2_RANK1234(float)
 BM_HETE_TENSOR_STRIDE_STEP2_RANK1234(double)
 BM_HETE_TENSOR_STRIDE_STEP2_RANK1234(point3f)
 BM_HETE_TENSOR_STRIDE_STEP2_RANK1234(point4f)
