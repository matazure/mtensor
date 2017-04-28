
template <typename _ValueType, int_t _Rank>
inline MATAZURE_GENERAL bool outside(point<_ValueType, _Rank> input, point<_ValueType, _Rank> halo, point<_ValueType, _Rank> extent) {
	for (int_t i = 0; i < _Rank; ++i) {
		if (input[i] < halo[i] || input[i] >= extent[i] - halo[i])
			return false;
	}

	return true;
}

namespace detail {

template <typename _Tensor>
inline auto clone_int_t_tensor(_Tensor ts) {
	return matazure::tensor<int_t, _Tensor::rank>(ts.shape());
}

template <typename _T, int_t ...SArgs>
inline auto clone_int_t_tensor(const static_tensor<_T, SArgs...> &) {
	return static_tensor<int_t, SArgs...>{};
}

}

template <typename _Tensor, typename _Weight, typename _Pos>
inline auto conv(_Tensor ts_input, _Weight weights, _Pos pos) {
	return make_lambda(ts_input.shape(), [=](int_t offset) {
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
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_Tensor, _Mask);
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_Tensor, _Mask);

	auto ts_offset_mask = detail::clone_int_t_tensor(ts_mask);
	for (int_t i = 0; i < ts_offset_mask.size(); ++i) {
		auto tmp_index = offset2index(i, ts_offset_mask.stride());
		auto tmp_offset = index2offset(tmp_index, ts_input.stride());
		ts_offset_mask[i] = tmp_offset;
	}

	return conv(ts_input, ts_mask, ts_offset_mask);
}
