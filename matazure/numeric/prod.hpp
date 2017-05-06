#pragma once

#include <matazure/common.hpp>

namespace matazure {
namespace puzzel {

namespace internal{

template <typename _Tensor1, typename _Tensor2>
struct prod_op {
	typedef typename _Tensor1::value_type result_type;

	prod_op(_Tensor1 cmat_lhs, _Tensor2 cmat_rhs) : cmat_lhs_(cmat_lhs), cmat_rhs_(cmat_rhs) {}

	MATAZURE_GENERAL result_type operator()(pointi<2> idx) const {
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

}

template <typename _TensorLhs, typename _TensorRhs>
auto product(_TensorLhs ts_lhs, _TensorRhs ts_rhs)->decltype(make_lambda(pointi<2>{ts_rhs.shape()[0], ts_lhs.shape()[1]}, internal::prod_op<_TensorLhs, _TensorRhs>(ts_lhs, ts_rhs))){
	return make_lambda(pointi<2>{ts_rhs.shape()[0], ts_lhs.shape()[1]}, internal::prod_op<_TensorLhs, _TensorRhs>(ts_lhs, ts_rhs));
}

}
}
