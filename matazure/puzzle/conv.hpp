#pragma once

#include <matazure/tensor.hpp>

namespace matazure {
namespace puzzle {

namespace internal{

	template <typename _TensorSrc, typename _TensorKernel>
	struct conv_array_index_op {
		_TensorSrc ts_src_;
		_TensorKernel ts_kernel_;

		conv_array_index_op(_TensorSrc ts_src, _TensorKernel ts_kernel) : ts_src_(ts_src), ts_kernel_(ts_kernel) {}

		MATAZURE_GENERAL auto operator()(const pointi<_TensorSrc::rank> &idx) const ->typename _TensorSrc::value_type {
			auto kernel_radius = (ts_kernel_.shape() / 2);
			auto sum = zero<typename _TensorSrc::value_type>::value();
			for_index(pointi<_TensorSrc::rank>::zeros(), ts_kernel_.shape(), [&](const pointi<2> &kenel_idx) {
				sum = sum + ts_src_[idx + kenel_idx - kernel_radius] * ts_kernel_[kenel_idx];
			});
			return sum;
		}
	};

}

//
///
///
template <typename _TensorSrc, typename _TensorKernel>
inline MATAZURE_GENERAL auto conv_lazy_array_index_unclamp(_TensorSrc ts_src, _TensorKernel ts_kernel)->decltype(
	make_lambda(
	ts_src.shape(),
	internal::conv_array_index_op<_TensorSrc, _TensorKernel>(ts_src, ts_kernel),
	typename _TensorSrc::memory_type{}
	)
	) {
	return make_lambda(
		ts_src.shape(),
		internal::conv_array_index_op<_TensorSrc, _TensorKernel>(ts_src, ts_kernel),
		typename _TensorSrc::memory_type{}
	);
}


template <typename _TensorSrc, typename _TensorKernel>
inline auto conv_lazy_array_index_inside_clamp_zero(_TensorSrc ts_src, _TensorKernel ts_kernel) {
	auto lts_conv_unclamp = conv_lazy_array_index_unclamp(ts_src, ts_kernel);
	auto lts_conv_clamp = conv_lazy_array_index_unclamp(clamp_zero(ts_src), ts_kernel);

	auto kernel_radius = ts_kernel.shape() / 2;
	auto safe_shape = ts_src.shape() - (ts_kernel.shape() - 1) / 2;
	auto start = kernel_radius;
	auto end = kernel_radius + safe_shape;
	return make_lambda(ts_src.shape(), [=](pointi<_TensorSrc::rank> a_idx) {
		if (MATAZURE_LIKELY(inside_range(a_idx, start, end))) {
			return lts_conv_unclamp[a_idx];
		}
		else {
			return lts_conv_clamp[a_idx];
		}
	});
}

// template <typename _ExecutionPolicy, typename _TensorSrc, typename _TensorKernel, typename _TensorDst>
// inlne void conv_direct_array_index_outside_clamp(_ExectutionPolicy policy, _TensorSrc ts_src, _TensorKernel ts_kernel, _TensorDst ts_dst){
// 	auto lts_conv_unclamp = conv_lazy_array_index_unclamp(ts_src, ts_kernel);
// 	auto kernel_radius = ts_kernel.shape() / 2;
// 	auto safe_shape = lts_conv_unclamp.shape() - ts_kernel.shape() + 1;
// 	auto lts_conv_without_boundary = section(lts_conv_unclamp, kernel_radius, safe_shape);
// 	auto lts_dst_without_boundary = section(ts_dst, kernel_radius, safe_shape);
// 	copy(policy, lts_conv_without_boundary, lts_dst_without_boundary);
// }

// decltype(lts_conv_unclamp.persist()) ts_conv_re(lts_conv_unclamp.shape());


//
// template <typename _TensorSrc, typename _TensorKernel>
// inline auto conv_lazy_array_inside_check(_TensorSrc ts_src, _TensorKernel ts_kernel)->decltype
//
// template <typename _TensorSrc, typename _TensorKernel>
// inline auto conv_lazy_linear_none_check;
//
// template <typename _TensorSrc, typename

// template <typename _TensorSrc

// template <typename _TensorSrc, typename _TensorKernel, typename _TensorOut>
// inline void conv_direct(_TensorSrc ts_src, _TensorKernel ts_kernel, _TensorOut ts_out) {
// 	auto kernel_radius = ts_kernel.shape() / 2;
// 	for_index(pointi<_TensorSrc::rank>::zeros(), )
// 			float sum = 0.0f;
// 			for (int_t n = 0; n < ts_kernel.shape()[1]; ++n) {
// 				for (int_t m = 0; m < ts_kernel.shape()[0]; ++m) {
// 					sum += ts_src[pointi<2>{i, j} +pointi<2>{m, n} -kernel_radius] * ts_kernel[pointi<2>{m, n}];
// 				}
// 			}
//
// 			ts_out[pointi<2>{i, }] = sum;
// 		}
// 	}
// }

// template <typename _TensorSrc, typename

}
}
