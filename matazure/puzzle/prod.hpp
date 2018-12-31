#pragma once

#include <matazure/binary_operator.hpp>

#include <matazure/simd.hpp>

#include <matazure/puzzle/transpose.hpp>

namespace matazure { namespace puzzle{

namespace internal{

template <typename _MatrixLhs, typename _MatrixRhs>
struct prod_op {
	typedef typename _MatrixLhs::value_type result_type;

	prod_op(_MatrixLhs cmat_lhs, _MatrixRhs cmat_rhs) :
		mat_lhs_(cmat_lhs), mat_rhs_(cmat_rhs)
	{}

	MATAZURE_GENERAL result_type operator()(const pointi<2> &idx) const {
		result_type re = 0;
		for (int_t i = 0; i < mat_lhs_.shape()[1]; ++i) {
			re += mat_lhs_[pointi<2>{idx[0], i}] * mat_rhs_[pointi<2>{i, idx[1]}];
		}
		return re;
	}

private:
	_MatrixLhs mat_lhs_;
	_MatrixRhs mat_rhs_;
};

}

template <typename _MatrixLhs, typename _MatrixRhs>
inline auto prod_general(_MatrixLhs mat_lhs, _MatrixRhs mat_rhs)->decltype(make_lambda(pointi<2>{mat_lhs.shape()[0], mat_rhs.shape()[1]}, internal::prod_op<_MatrixLhs, _MatrixRhs>(mat_lhs, mat_rhs), typename _MatrixLhs::memory_type{})) {
	MATAZURE_STATIC_ASSERT_MATRIX_RANK(_MatrixLhs);
	MATAZURE_STATIC_ASSERT_MATRIX_RANK(_MatrixRhs);
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_MatrixLhs, _MatrixRhs);
	MATAZURE_STATIC_ASSERT_MEMORY_TYPE_MATCHED(_MatrixLhs, _MatrixRhs);
	MATAZURE_ASSERT(mat_lhs.shape()[1] == mat_rhs.shape()[0], "unmatched size");

	return make_lambda(pointi<2>{mat_lhs.shape()[0], mat_rhs.shape()[1]}, internal::prod_op<_MatrixLhs, _MatrixRhs>(mat_lhs, mat_rhs), typename _MatrixLhs::memory_type{});
}

#ifndef ANDROID

inline simd_vector<float, 4> prod0(point<simd_vector<float, 4>, 4> lhs, simd_vector<float, 4> rhs) {	
	return hadd(hadd(lhs[0] * rhs, lhs[1] * rhs), hadd(lhs[2] * rhs, lhs[3] * rhs));
}

#endif

inline simd_vector<float, 4> prod1(point<simd_vector<float, 4>, 4> lhs, simd_vector<float, 4> rhs) {	

	simd_vector<float, 4> sv_rhs0;
	simd_vector<float, 4> sv_rhs1;
	simd_vector<float, 4> sv_rhs2;
	simd_vector<float, 4> sv_rhs3;
	fill(sv_rhs0, rhs[0]);
	fill(sv_rhs1, rhs[1]);
	fill(sv_rhs2, rhs[2]);
	fill(sv_rhs3, rhs[3]);
	return (sv_rhs0 * lhs[0] + sv_rhs1 * lhs[1]) + (sv_rhs2 * lhs[2] + sv_rhs3 * lhs[3]);
}

//inline static_tensor<float, dim<4, 4>> prod(static_tensor<float, dim<4, 4>> sts_lhs, static_tensor<float, dim<4, 4>> sts_rhs) {
//	
//}

} }
