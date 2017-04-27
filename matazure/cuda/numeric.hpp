#pragma once

#include <matazure/cuda/tensor.hpp>

namespace matazure {
namespace cuda {

template <typename _Tensor1, typename _Tensor2>
struct __product {
	typedef typename _Tensor1::value_type result_type;

	__product(_Tensor1 cmat_lhs, _Tensor2 cmat_rhs) : cmat_lhs_(cmat_lhs), cmat_rhs_(cmat_rhs) {}

	MATAZURE_DEVICE result_type operator()(pointi<2> idx) const {
		result_type re = 0;
		for (int_t i = 0; i < cmat_rhs_.shape()[1]; ++i) {
			re += cmat_lhs_({ i, idx[1] }) * cmat_rhs_({ idx[0], i });
		}
		return re;
	}

private:
	_Tensor1 cmat_lhs_;
	_Tensor2 cmat_rhs_;
};

template <typename _Tensor1, typename _Tensor2>
device_lambda_tensor<typename _Tensor1::value_type, array_access_t, 2, __product<_Tensor1, _Tensor2>> product(_Tensor1 cmat_lhs, _Tensor2 cmat_rhs) {
	return make_lambda<typename _Tensor1::value_type, array_access_t>(pointi<2>{cmat_rhs.shape()[0], cmat_lhs.shape()[1]}, __product<_Tensor1, _Tensor2>(cmat_lhs, cmat_rhs));
}

template <int_t _BlockSize, typename _Matrix1, typename _Matrix2>
matrix<typename _Matrix1::value_type>  block_product(_Matrix1 cmat_lhs, _Matrix2 cmat_rhs) {
	typedef typename _Matrix1::value_type value_type;
	auto lhs_ext = cmat_lhs.shape();
	auto rhs_ext = cmat_rhs.shape();
	matrix<value_type> cmat_re({ lhs_ext[0], rhs_ext[1] });
 pointi<2> block_ext{ _BlockSize, _BlockSize };

	block_for_index<_BlockSize, _BlockSize>(cmat_re.shape() / block_ext, [=] MATAZURE_DEVICE(block_index<_BlockSize, _BlockSize> t_idx) {
		auto row = t_idx.local[0];
		auto col = t_idx.local[1];
		auto global_row = t_idx.global[0];
		auto global_col = t_idx.global[1];

		__shared__ static_tensor<value_type,dim< _BlockSize,  _BlockSize>> local_lhs;
		__shared__ static_tensor<value_type,dim< _BlockSize,  _BlockSize>> local_rhs;

		value_type sum = 0;
		for (int_t i = 0; i < lhs_ext[0]; i += _BlockSize) {
			local_lhs({ row, col }) = cmat_lhs({ global_row, col + i });
			local_rhs({ col, row }) = cmat_rhs({ row+i, global_col });
			block_barrier();

			for (int_t k = 0; k < _BlockSize; k++) {
				sum += local_lhs({ row, k }) *local_rhs({ k, col });
			}

			block_barrier();
		}

		cmat_re(t_idx.global) = sum;
	});

	return cmat_re;
}

template <typename _Func>
__device__ void for_index(pointi<2> extent, _Func fun) {
	for (int_t j = 0; j < extent[1]; ++j) {
		for (int_t i = 0; i < extent[0]; ++i) {
			fun(pointi<2>{ { i, j } });
		}
	}
}


template <int_t _Block0, int_t _Block1, typename _Tensor, typename _Mask>
tensor<typename _Tensor::value_type, _Tensor::rank> conv(_Tensor ts, _Mask mask) {
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_Tensor, _Mask);
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_Tensor, _Mask);
	typedef typename _Tensor::value_type value_type;

	constexpr pointi<2> block_ext{ _Block0, _Block1 };
 pointi<2> grid_ext = ts.shape() / block_ext;
	//MATAZURE_ASSERT(grid_ext * block_ext == ts.shape());
	auto mask_extent = mask.shape();
	auto mask_radius = mask_extent / 2;
	tensor<value_type, _Tensor::rank> cts_re(ts.shape());

	block_for_index<_Block0, _Block1>(grid_ext, [=] MATAZURE_DEVICE(block_index<_Block0, _Block1> block_idx) {
		__shared__ static_tensor<value_type,dim< _Block0,  _Block1>> sts_apron;
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
