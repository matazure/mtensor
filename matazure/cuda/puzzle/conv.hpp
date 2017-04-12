#pragma once

#include <matazure/cuda/tensor.hpp>

namespace matazure {
namespace cuda {

template <typename _Func>
__device__ void for_index(pointi<2> extent, _Func fun) {
	for (int_t j = 0; j < extent[1]; ++j) {
		for (int_t i = 0; i < extent[0]; ++i) {
			fun(pointi<2>{ { i, j } });
		}
	}
}


template <int_t _Block0, int_t _Block1, typename _Tensor, typename _Mask>
tensor<typename _Tensor::value_type, _Tensor::dim> conv(_Tensor ts, _Mask mask) {
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_Tensor, _Mask);
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_Tensor, _Mask);
	typedef typename _Tensor::value_type value_type;

	constexpr pointi<2> block_ext{ _Block0, _Block1 };
	pointi<2> grid_ext = ts.extent() / block_ext;
	//MATAZURE_ASSERT(grid_ext * block_ext == ts.extent());
	auto mask_extent = mask.extent();
	auto mask_radius = mask_extent / 2;
	tensor<value_type, _Tensor::dim> cts_re(ts.extent());

	block_for_index<_Block0, _Block1>(grid_ext, [=] MATAZURE_DEVICE(block_index<_Block0, _Block1> block_idx) {
		__shared__ static_tensor<value_type, _Block0, _Block1> sts_apron;
		sts_apron(block_idx.local) = ts(block_idx.global);
		device::barrier();

		if (inside(block_idx.local, mask_radius, block_idx.block_extent)) {
			value_type sum = 0;

			cuda::for_index(mask_extent, [&](const pointi<2> &idx) {
				sum += sts_apron(block_idx.local + idx - mask_radius) * mask(idx);
			});

			cts_re(block_idx.global) = sum;
		}
	});

	return cts_re;
}

}
}
