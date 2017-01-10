#pragma once

#include <matazure/tensor.hpp>

namespace matazure {

template <typename _ValueType, int_t _Dim>
inline MATAZURE_GENERAL bool in_apron(point<_ValueType, _Dim> input, point<_ValueType, _Dim> halo, point<_ValueType, _Dim> extent) {
	for (int_t i = 0; i < _Dim; ++i) {
		if (input[i] < halo[i] || input[i] >= extent[i] - halo[i])
			return false;
	}

	return true;
}

template <typename _TS>
typename _TS::value_type sum(_TS ts) {
	typedef typename _TS::value_type value_type;
	return reduce(ts, value_type(0), [=](value_type lhs, value_type rhs) {
		return lhs + rhs;
	});
}

template <typename _TS>
typename _TS::value_type mean(_TS ts) {
	return sum(ts) / ts.size();
}

template <typename _TS>
typename _TS::value_type max(_TS ts) {
	typedef typename _TS::value_type value_type;
	return reduce(ts, remove_const_t<decay_t<value_type>>(ts[0]), [=](value_type lhs, value_type rhs) {
		return rhs > lhs ? rhs : lhs;
	});
}

template <typename _TS>
typename _TS::value_type min(_TS ts) {
	typedef typename _TS::value_type value_type;
	return reduce(ts, numeric_limits<value_type>::max(), [=](value_type lhs, value_type rhs) {
		return lhs <= rhs ? lhs : rhs;
	});
}

template <typename _VectorLhs, typename _VectorRhs>
auto inner_product(_VectorLhs vec_lhs, _VectorRhs vec_rhs) {
	/*	assert(vec_lhs.size() == vec_rhs.size());*/

	decltype(vec_lhs[0]) re = 0;
	for (int_t i = 0; i < vec_lhs.size(); ++i) {
		re += vec_lhs[i] * vec_rhs[i];
	}

	return re;
}

template <typename _MatrixLhs, typename _MatrixRhs>
auto product(_MatrixLhs mat_lhs, _MatrixRhs mat_rhs) {
	static_assert(is_matrix_expression<_MatrixLhs>::value, "");
	static_assert(is_matrix_expression<_MatrixRhs>::value, "");

	assert(mat_lhs.extent()[1] == mat_rhs.extent()[0]);

	return make_lambda(pointi<2>{ mat_lhs.extent()[0], mat_rhs.extent()[1] }, [=](pointi<2> idx) {
		//return inner_product(row(mat_lhs, idx[0]), column(mat_rhs, idx[1]));
		float re = 0;
		for (int_t k = 0; k < mat_lhs.extent()[1]; ++k) {
			re += mat_lhs({ idx[0], k }) * mat_rhs({ k, idx[1] });
		}

		return re;
	});
}

template <typename _MatrixLhs, typename _MatrixRhs>
auto product_tmp(_MatrixLhs mat_lhs, _MatrixRhs mat_rhs) {
	static_assert(is_matrix_expression<_MatrixLhs>::value, "");
	static_assert(is_matrix_expression<_MatrixRhs>::value, "");

	assert(mat_lhs.extent()[0] == mat_rhs.extent()[0]);

	return make_lambda(pointi<2>{ mat_rhs.extent()[1], mat_lhs.extent()[1] }, [=](pointi<2> i) {
		float re = 0;
		for (int_t k = 0; k < mat_rhs.extent()[0]; ++k) {
			re += mat_lhs({ k, i[1] }) * mat_rhs({ k, i[0] });
		}

		return re;
	});
}

namespace detail {

template <typename _Tensor>
inline auto clone_int_t_tensor(_Tensor ts) {
	return matazure::tensor<int_t, _Tensor::dim>(ts.extent());
}

template <typename _T, int_t ...SArgs>
inline auto clone_int_t_tensor(const static_tensor<_T, SArgs...> &) {
	return static_tensor<int_t, SArgs...>{};
}
}

template <typename _Tensor, typename _Weight, typename _Pos>
inline auto conv(_Tensor ts_input, _Weight weights, _Pos pos) {
	return make_lambda(ts_input.extent(), [=](int_t offset) {
		typename _Tensor::value_type re = 0;

		if (offset + pos[0] < 0 || offset + pos[pos.size() - 1] >= ts_input.size())
			return decltype(re)(0.0);

		for (int_t i = 0; i < weights.size(); ++i) {
			auto tmp_offset = offset + pos[i];
			re += ts_input[tmp_offset] * weights[i];
		}

		return re;
	});
}

template <typename _Tensor, typename _Mask>
inline auto conv(_Tensor ts_input, _Mask ts_mask) {
	STATIC_ASSERT_DIM_MATCHED(_Tensor, _Mask);
	STATIC_ASSERT_VALUE_TYPE_MATCHED(_Tensor, _Mask);

	auto ts_offset_mask = detail::clone_int_t_tensor(ts_mask);
	for (int_t i = 0; i < ts_offset_mask.size(); ++i) {
		auto tmp_index = offset2index(i, ts_offset_mask.stride());
		auto tmp_offset = index2offset(tmp_index, ts_input.stride());
		ts_offset_mask[i] = tmp_offset;
	}

	return conv(ts_input, ts_mask, ts_offset_mask);
}

}
