#pragma once

#include <matazure/cuda/tensor.hpp>

namespace matazure {
namespace cuda {

template <typename _Func>
__device__ void for_index(pointi<2> extent, _Func fun) {
#pragma unroll
	for (int_t j = 0; j < extent[1]; ++j) {
		for (int_t i = 0; i < extent[0]; ++i) {
			fun(pointi<2>{ { i, j } });
		}
	}
}

namespace puzzle {

template <int_t _Block0, int_t _Block1, typename _Tensor, typename _Mask, typename _TensorRe>
inline tensor<typename _Tensor::value_type, _Tensor::dim> conv(_Tensor ts, const _Mask &mask, _TensorRe &ts_re) {
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_Tensor, _Mask);
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_Tensor, _Mask);
	typedef typename _Tensor::value_type value_type;

	constexpr pointi<2> block_ext{ _Block0, _Block1 };
	pointi<2> grid_ext = ts.extent() / block_ext;
	MATAZURE_ASSERT(equal(grid_ext * block_ext, ts.extent()));
	MATAZURE_ASSERT(equal(ts.extent(), ts_re.extent()));

	auto mask_extent = mask.extent();
	auto mask_radius = mask_extent / 2;

	block_for_index<_Block0, _Block1>(grid_ext, [=] MATAZURE_DEVICE(block_index<_Block0, _Block1> block_idx) {
		__shared__ static_tensor<value_type, _Block0, _Block1> shared_ts_block;
		shared_ts_block(block_idx.local) = ts(block_idx.global);
		device::barrier();

		if (inside(block_idx.local, mask_radius, block_idx.block_extent)) {
			value_type sum = 0;

			/*cuda::for_index(mask_extent, [&](const pointi<2> &idx) {
				sum += shared_ts_block(block_idx.local + idx - mask_radius) * mask(idx);
			});*/

			#pragma unroll
			for (int_t j = 0; j < mask_extent[1]; ++j) {
				for (int_t i = 0; i < mask_extent[0]; ++i) {
					sum += shared_ts_block(block_idx.local + pointi<2>{i, j} -mask_radius) * mask(pointi<2>{i, j});
				}
			}
			

			ts_re(block_idx.global) = sum;
		}
	});

	return ts_re;
}


template <int_t _Block0, int_t _Block1, typename _Tensor, typename _Mask>
inline tensor<typename _Tensor::value_type, _Tensor::dim> conv(_Tensor ts, _Mask mask) {
	tensor<typename _Tensor::value_type, _Tensor::dim> ts_re(ts.extent());
	conv<_Block0, _Block1>(ts, mask, ts_re);
	return ts_re;
}

}
}
}