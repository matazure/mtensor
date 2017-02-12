#pragma once

#include <matazure/tensor.hpp>

namespace matazure {

template <typename _T, int_t _LhsRows, int_t _LhsCols, int_t _RhsCols>
MATAZURE_GENERAL auto product(static_tensor<_T, _LhsRows, _LhsCols> ts_lhs, static_tensor<_T, _LhsCols, _RhsCols> ts_rhs)->static_tensor<_T, _LhsRows, _RhsCols> {
	auto ts_re = static_tensor<_T, _LhsRows, _RhsCols>::zeros();
	for (int_t i = 0; i < _LhsRows; ++i) {
		for (int_t j = 0; j < _RhsCols; ++j) {
			for (int k = 0; k < _LhsCols; ++k) {
				ts_re(i, j) += ts_lhs(i, k) * ts_rhs(k, j);
			}
		}
	}

	return ts_re;
}

template <typename _T, int_t _Rows, int_t _Cols>
MATAZURE_GENERAL auto product(static_tensor<_T, _Rows, _Cols> mat, point<_T, _Cols> vec)->point<_T, _Rows> {
	auto vec_re = point<_T, _Rows>::zeros();
	for (int_t i = 0; i < _Rows; ++i) {
		for (int j = 0; j < _Cols; ++j) {
			vec_re[i] += mat(i, j) * vec[j];
		}
	}

	return vec_re;
}

template <typename _T, int_t _Rows, int_t _Cols>
MATAZURE_GENERAL auto product(static_tensor<_T, _Rows, _Cols> mat, static_tensor<_T, _Cols> vec)->static_tensor<_T, _Rows> {
	auto vec_re = static_tensor<_T, _Rows>::zeros();
	for (int_t i = 0; i < _Rows; ++i) {
		for (int j = 0; j < _Cols; ++j) {
			vec_re[i] += mat(i, j) * vec[j];
		}
	}

	return vec_re;
}


// template <typename _VectorLhs, typename _VectorRhs>
// auto inner_product(_VectorLhs vec_lhs, _VectorRhs vec_rhs) {
// 	/*	assert(vec_lhs.size() == vec_rhs.size());*/
//
// 	decltype(vec_lhs[0]) re = 0;
// 	for (int_t i = 0; i < vec_lhs.size(); ++i) {
// 		re += vec_lhs[i] * vec_rhs[i];
// 	}
//
// 	return re;
// }
//
// template <typename _MatrixLhs, typename _MatrixRhs>
// auto product(_MatrixLhs mat_lhs, _MatrixRhs mat_rhs) {
// 	MATAZURE_STATIC_ASSERT_IS_MATRIX(_MatrixLhs);
// 	MATAZURE_STATIC_ASSERT_IS_MATRIX(_MatrixRhs);
//
// 	assert(mat_lhs.extent()[1] == mat_rhs.extent()[0]);
//
// 	return make_lambda(pointi<2>{ mat_lhs.extent()[0], mat_rhs.extent()[1] }, [=](pointi<2> idx) {
// 		//return inner_product(row(mat_lhs, idx[0]), column(mat_rhs, idx[1]));
// 		float re = 0;
// 		for (int_t k = 0; k < mat_lhs.extent()[1]; ++k) {
// 			re += mat_lhs({ idx[0], k }) * mat_rhs({ k, idx[1] });
// 		}
//
// 		return re;
// 	});
// }
//
// template <typename _MatrixLhs, typename _MatrixRhs>
// auto product_tmp(_MatrixLhs mat_lhs, _MatrixRhs mat_rhs) {
// 	MATAZURE_STATIC_ASSERT_IS_MATRIX(_MatrixLhs);
// 	MATAZURE_STATIC_ASSERT_IS_MATRIX(_MatrixRhs);
//
// 	assert(mat_lhs.extent()[0] == mat_rhs.extent()[0]);
//
// 	return make_lambda(pointi<2>{ mat_rhs.extent()[1], mat_lhs.extent()[1] }, [=](pointi<2> i) {
// 		float re = 0;
// 		for (int_t k = 0; k < mat_rhs.extent()[0]; ++k) {
// 			re += mat_lhs({ k, i[1] }) * mat_rhs({ k, i[0] });
// 		}
//
// 		return re;
// 	});
// }

}
