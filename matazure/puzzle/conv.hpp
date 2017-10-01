#pragma once

#include <matazure/tensor.hpp>

namespace matazure { namespace puzzle {

namespace internal{

	template <typename _Tensor, typename _Kenel>
	struct conv_op {
		_Tensor ts_;
		_Kenel kenel_;

		conv_op(_Tensor ts, _Kenel kenel): ts_(ts), kenel_(kenel) {}

		auto operator()(const pointi<_Tensor::rank> &idx) const ->typename _Tensor::value_type {
			auto kenel_radius = (kenel_.shape() / 2);
			auto sum = zero<typename _Tensor::value_type>::value();
			for_index(pointi<_Tensor::rank>::zeros(), kenel_.shape(), [&] (const pointi<2> &kenel_idx) {
				sum += ts_[idx + kenel_idx - kenel_radius] * kenel_[kenel_idx];
			});
			return sum;
		}
	};

}


template <typename _Tensor, typename _Kenel>
inline auto conv_lazy_array_none_check(_Tensor ts, _Kenel kenel)->decltype(make_lambda(ts.shape(), internal::conv_op<_Tensor, _Kenel>(ts, kenel))){
	return make_lambda(ts.shape(), internal::conv_op<_Tensor, _Kenel>(ts, kenel));
}

//
// template <typename _Tensor, typename _Kenel>
// inline auto conv_lazy_array_inside_check(_Tensor ts, _Kenel kenel)->decltype
//
// template <typename _Tensor, typename _Kenel>
// inline auto conv_lazy_linear_none_check;
//
// template <typename _Tensor, typename

// template <typename _Tensor

// template <typename _Tensor, typename _Kenel, typename _TensorOut>
// inline void conv_direct(_Tensor ts, _Kenel kenel, _TensorOut ts_out) {
// 	auto kenel_radius = kenel.shape() / 2;
// 	for_index(pointi<_Tensor::rank>::zeros(), )
// 			float sum = 0.0f;
// 			for (int_t n = 0; n < kenel.shape()[1]; ++n) {
// 				for (int_t m = 0; m < kenel.shape()[0]; ++m) {
// 					sum += ts[pointi<2>{i, j} +pointi<2>{m, n} -kenel_radius] * kenel[pointi<2>{m, n}];
// 				}
// 			}
//
// 			ts_out[pointi<2>{i, }] = sum;
// 		}
// 	}
// }

// template <typename _Tensor, typename

} }
