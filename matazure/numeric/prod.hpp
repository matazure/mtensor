#pragma once

#include <matazure/common.hpp>

namespace matazure {
namespace puzzle {

namespace internal{

template <typename _MatrixLhs, typename _MatrixRhs>
struct prod_op {
	typedef typename _MatrixLhs::value_type result_type;

	prod_op(_MatrixLhs cmat_lhs, _MatrixRhs cmat_rhs) :
		mat_lhs_(cmat_lhs), mat_rhs_(cmat_rhs) 
	{}

	MATAZURE_GENERAL result_type operator()(pointi<2> idx) const {
		result_type re = 0;
		for (int_t i = 0; i < mat_lhs_.shape()[1]; ++i) {
			re += mat_lhs_(idx[0], i) * mat_rhs_(i, idx[1]);
		}
		return re;
	}

private:
	_MatrixLhs mat_lhs_;
	_MatrixRhs mat_rhs_;
};

}

template <typename _MatrixLhs, typename _MatrixRhs>
auto prod_general(_MatrixLhs mat_lhs, _MatrixRhs mat_rhs)->decltype(make_lambda(pointi<2>{mat_lhs.shape()[0], mat_rhs.shape()[1]}, internal::prod_op<_MatrixLhs, _MatrixRhs>(mat_lhs, mat_rhs), typename _MatrixLhs::memory_type{})) {
	MATAZURE_STATIC_ASSERT_MATRIX_RANK(_MatrixLhs);
	MATAZURE_STATIC_ASSERT_MATRIX_RANK(_MatrixRhs);
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_MatrixLhs, _MatrixRhs);
	MATAZURE_STATIC_ASSERT_MEMORY_TYPE_MATCHED(_MatrixLhs, _MatrixRhs);
	MATAZURE_ASSERT(mat_lhs.shape()[1] == mat_rhs.shape()[0], "unmatched size");

	return make_lambda(pointi<2>{mat_lhs.shape()[0], mat_rhs.shape()[1]}, internal::prod_op<_MatrixLhs, _MatrixRhs>(mat_lhs, mat_rhs), typename _MatrixLhs::memory_type{});
}

}
}
