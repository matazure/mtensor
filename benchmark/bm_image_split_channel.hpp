#pragma once

#include "bm_config.hpp"

#include <thread>

void bm_image_split_channel(benchmark::State & state) {
	auto width = state.range(0);
	auto height = width;

	tensor<byte, 3> img_pack(3, width, height);
	tensor<float, 3> img_plane(width, height, 3);

	auto plane_size = width * height;

	while (state.KeepRunning()) {

		auto p_b = reinterpret_cast<point3b *>(img_pack.data());
		auto p_f0 = img_plane.data();
		auto p_f1 = p_f0 + width * height;
		auto p_f2 = p_f1 + width * height;
        for (int i = 0; i < width * height; ++i) {
			p_f0[i] = 0.003875f * (static_cast<float>(p_b[i][0]) - 128.0f);
			p_f1[i] = 0.003875f * (static_cast<float>(p_b[i][1]) - 128.0f);
			p_f2[i] = 0.003875f * (static_cast<float>(p_b[i][2]) - 128.0f);
        }

        // for_index(img_pack.shape(), [=](pointi<3> idx) {
        //     img_plane(idx[1], idx[2], idx[0]) = 0.003875f * (img_pack(idx) - 128.0f);
        // });
    }

    // std::cout << p_b[2] << std::endl;
    auto bytes_size = static_cast<size_t>(img_plane.size() * 4);
	state.SetBytesProcessed(state.iterations() * bytes_size);
	state.SetItemsProcessed(state.iterations() * img_plane.size());
}

template <typename _Tensor>
void bm_hete_tensor_copy(benchmark::State & state) {
	_Tensor ts_src(pointi<_Tensor::rank>::all(state.range(0)));
	_Tensor ts_dst(ts_src.shape());
	fill(ts_src, zero<typename _Tensor::value_type>::value());

	while (state.KeepRunning()) {
		copy(ts_src, ts_dst);
		// HETE_SYNCHRONIZE;

		benchmark::ClobberMemory();
	}

	auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
	state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}

#define USE_HOST

#define BM_HETE_TENSOR_COPY(ValueType, Rank)                                                                 \
	auto bm_hete_tensor_##ValueType##_rank##Rank##_copy = bm_hete_tensor_copy<tensor<ValueType, Rank>>; \
	BENCHMARK(bm_hete_tensor_##ValueType##_rank##Rank##_copy)->RangeMultiplier(bm_config::range_multiplier<ValueType, Rank, host_tag>())->Range(bm_config::min_shape<ValueType, Rank, host_tag>(), bm_config::max_shape<ValueType, Rank, host_tag>())->UseRealTime();

#define BM_HETE_TENSOR_RANK1234_COPY(ValueType) \
	BM_HETE_TENSOR_COPY(ValueType, 1)           \
	BM_HETE_TENSOR_COPY(ValueType, 2)           \
	BM_HETE_TENSOR_COPY(ValueType, 3)           \
	BM_HETE_TENSOR_COPY(ValueType, 4)

// BM_HETE_TENSOR_RANK1234_COPY(byte)
// BM_HETE_TENSOR_RANK1234_COPY(int16_t)
// BM_HETE_TENSOR_RANK1234_COPY(int32_t)
// BM_HETE_TENSOR_RANK1234_COPY(int64_t)
// BM_HETE_TENSOR_RANK1234_COPY(float)
// BM_HETE_TENSOR_RANK1234_COPY(double)
// BM_HETE_TENSOR_RANK1234_COPY(point3f)
// BM_HETE_TENSOR_RANK1234_COPY(point4f)
// BM_HETE_TENSOR_RANK1234_COPY(hete_float32x4_t)