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
		for (int_t i = 0; i < cmat_rhs_.extent()[1]; ++i) {
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
	return make_lambda<typename _Tensor1::value_type, array_access_t>(pointi<2>{cmat_rhs.extent()[0], cmat_lhs.extent()[1]}, __product<_Tensor1, _Tensor2>(cmat_lhs, cmat_rhs));
}

template <int_t _BlockSize, typename _Matrix1, typename _Matrix2>
matrix<typename _Matrix1::value_type>  block_product(_Matrix1 cmat_lhs, _Matrix2 cmat_rhs) {
	typedef typename _Matrix1::value_type value_type;
	auto lhs_ext = cmat_lhs.extent();
	auto rhs_ext = cmat_rhs.extent();
	matrix<value_type> cmat_re({ lhs_ext[0], rhs_ext[1] });
 pointi<2> block_ext{ _BlockSize, _BlockSize };

	block_for_index<_BlockSize, _BlockSize>(cmat_re.extent() / block_ext, [=] MATAZURE_DEVICE(block_index<_BlockSize, _BlockSize> t_idx) {
		auto row = t_idx.local[0];
		auto col = t_idx.local[1];
		auto global_row = t_idx.global[0];
		auto global_col = t_idx.global[1];

		__shared__ static_tensor<value_type, _BlockSize, _BlockSize> local_lhs;
		__shared__ static_tensor<value_type, _BlockSize, _BlockSize> local_rhs;

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

}
}
