#pragma once

#include <matazure/cuda/tensor.hpp>

namespace matazure {
namespace cuda {
namespace puzzle {

template <int_t _BlockSize, typename _MatrixLhs, typename _MatrixRhs, typename _MatrixRe>
inline void prod_block_aligned(_MatrixLhs cmat_lhs, _MatrixRhs cmat_rhs, _MatrixRe &cmat_re) {
	MATAZURE_STATIC_ASSERT_MATRIX_RANK(_MatrixLhs);
	MATAZURE_STATIC_ASSERT_MATRIX_RANK(_MatrixRhs);
	MATAZURE_STATIC_ASSERT_MATRIX_RANK(_MatrixRe);
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_MatrixLhs, _MatrixRhs);
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_MatrixRhs, _MatrixRe);
	MATAZURE_ASSERT(equal(cmat_re.shape(), pointi<2>{cmat_lhs.shape()[0], cmat_rhs.shape()[1]}), "unmatched shape");

	typedef typename _MatrixRe::value_type value_type;

	constexpr auto block_size = _BlockSize;
	auto grid_ext = cmat_re.shape() / block_size;
	MATAZURE_ASSERT(equal(grid_ext * block_size, cmat_re.shape()), "unaligned shape");

	block_for_index<dim<block_size, block_size>>(grid_ext, [=] MATAZURE_DEVICE(block_index<dim<block_size, block_size>> t_idx) {
		auto row = t_idx.local[0];
		auto col = t_idx.local[1];
		auto global_row = t_idx.global[0];
		auto global_col = t_idx.global[1];

		__shared__ static_tensor<value_type, dim<block_size, block_size>> local_lhs;
		__shared__ static_tensor<value_type, dim<block_size, block_size>> local_rhs;

		auto sum = zero<value_type>::value();
		for (int_t i = 0; i < cmat_lhs.shape()[1]; i += block_size) {
			local_lhs(row, col) = cmat_lhs(global_row, col + i);
			local_rhs(row, col) = cmat_rhs(row + i, global_col);
			device::barrier();

			for (int_t k = 0; k < block_size; k++) {
				sum += local_lhs(row, k) * local_rhs(k, col);
			}

			device::barrier();
		}

		cmat_re[t_idx.global] = sum;
	});
}

template <int_t _BlockSize, typename _MatrixLhs, typename _MatrixRhs>
inline auto prod_block_aligned(_MatrixLhs cmat_lhs, _MatrixRhs cmat_rhs)->matrix<typename _MatrixLhs::value_type> {
	matrix<typename _MatrixLhs::value_type> cmat_re(cmat_lhs.shape()[0], cmat_rhs.shape()[1]);
	prod_block_aligned<_BlockSize>(cmat_lhs, cmat_rhs, cmat_re);
	return cmat_re;
}

template <int_t _BlockSize, typename _MatrixLhs, typename _MatrixRhs, typename _MatrixRe>
inline void prod_block(_MatrixLhs cmat_lhs, _MatrixRhs cmat_rhs, _MatrixRe &cmat_re) {
	MATAZURE_STATIC_ASSERT_MATRIX_RANK(_MatrixLhs);
	MATAZURE_STATIC_ASSERT_MATRIX_RANK(_MatrixRhs);
	MATAZURE_STATIC_ASSERT_MATRIX_RANK(_MatrixRe);
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_MatrixLhs, _MatrixRhs);
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_MatrixRhs, _MatrixRe);
	MATAZURE_ASSERT(equal(cmat_re.shape(), pointi<2>{cmat_lhs.shape()[0], cmat_rhs.shape()[1]}), "unmatched shape");

	typedef typename _MatrixRe::value_type value_type;

	constexpr auto block_size = _BlockSize;
	auto grid_ext = (cmat_re.shape() + block_size - 1) / block_size;

	block_for_index<dim<_BlockSize, _BlockSize>>(grid_ext, [=] MATAZURE_DEVICE (block_index<dim<_BlockSize, _BlockSize>> block_idx) {
		auto row = block_idx.local[0];
		auto col = block_idx.local[1];
		auto global_row = block_idx.global[0];
		auto global_col = block_idx.global[1];

		__shared__ static_tensor<value_type, dim<block_size, block_size>> local_lhs;
		__shared__ static_tensor<value_type, dim<block_size, block_size>> local_rhs;

		auto is_valid = inside_range(block_idx.global, pointi<2>::zeros(), cmat_re.shape());
		auto l = cmat_lhs.shape()[0];
		auto m = cmat_lhs.shape()[1];
		auto n = cmat_rhs.shape()[1];
		auto sum = zero<value_type>::value();
		for (int_t i = 0; i < m; i += block_size) {
			if (global_row < l && col + i < m) {
				local_lhs(row, col) = cmat_lhs(global_row, col + i);
			}
			else {
				local_lhs(row, col) = zero<value_type>::value();
			}
			if (global_col < n && row + i < m) {
				local_rhs(row, col) = cmat_rhs(row + i, global_col);
			}
			else {
				local_rhs(row, col) = zero<value_type>::value();
			}
	
			device::barrier();

			if (is_valid) {
				for (int_t k = 0; k < block_size; k++) {
					sum += local_lhs(row, k) * local_rhs(k, col);
				}
			}
			device::barrier();
		}

		if (is_valid)
			cmat_re[block_idx.global] = sum;
	});
}

template <int_t _BlockSize, typename _MatrixLhs, typename _MatrixRhs>
inline auto prod_block(_MatrixLhs cmat_lhs, _MatrixRhs cmat_rhs)->matrix<typename _MatrixLhs::value_type> {
	matrix<typename _MatrixLhs::value_type> cmat_re(cmat_lhs.shape()[0], cmat_rhs.shape()[1]);
	prod_block<_BlockSize>(cmat_lhs, cmat_rhs, cmat_re);
	return cmat_re;
}

}
}
}
