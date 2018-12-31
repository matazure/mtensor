#pragma once

#include <matazure/tensor.hpp>
#include <matazure/algorithm.hpp>
#include <limits>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
///	unary is the namespace for unary functor
inline namespace unary {

// /**
// * @brief saturate convert a value to a narrow type value
// *
// * for example, saturate_convertor<byte>{}(256) == byte(255), but byte(256) == byte(0)
// *
// * @tparam _Type dest value type
// */
// template <typename _Type>
// struct saturate_convertor {
// 	template <typename _TypeSrc>
// 	MATAZURE_GENERAL _Type operator()(_TypeSrc v) const {
// 		if (v < std::numeric_limits<_Type>::min())	return std::numeric_limits<_Type>::min();
// 		if (v > std::numeric_limits<_Type>::max())	return std::numeric_limits<_Type>::max();
// 		return static_cast<_Type>(v);
// 	}
// };
//
// template <typename _ValueType, int_t _Rank>
// struct saturate_convertor<point<_ValueType, _Rank>>{
//
// 	template <typename _TypeSrc>
// 	MATAZURE_GENERAL point<_ValueType, _Rank> operator()(point<_TypeSrc, _Rank> v) const {
// 		point<_ValueType, _Rank> re;
//
// 		transform(v, re, [](const _TypeSrc &e){
// 			return saturate_convertor<_ValueType>{}(e);
// 		});
//
// 		return re;
// 	}
//
// };

template <typename _TypeDst, typename _TypeSrc>
MATAZURE_GENERAL _TypeDst saturate_cast(_TypeSrc v){
	if (v < std::numeric_limits<_TypeDst>::min())	return std::numeric_limits<_TypeDst>::min();
	if (v > std::numeric_limits<_TypeDst>::max())	return std::numeric_limits<_TypeDst>::max();
	return static_cast<_TypeDst>(v);
}

template <typename _ValueTypeDst, typename _ValueTypeSrc, int_t _Rank>
MATAZURE_GENERAL auto saturate_cast(const point<_ValueTypeSrc, _Rank> &v)->point<_ValueTypeDst, _Rank>{
	point<_ValueTypeDst, _Rank> re;

	transform(v, re, [](const _ValueTypeSrc &e){
		return saturate_cast<_ValueTypeDst>(e);
	});

	return re;
}

}
}
