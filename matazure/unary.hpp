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

/**
* @brief saturate convert a value to a narrow type value
*
* for example, saturate_convertor<byte>{}(256) == byte(255), but byte(256) == byte(0)
*
* @tparam _Type dest value type
*/
template <typename _Type>
struct saturate_convertor {

	template <typename _SrcType>
	MATAZURE_GENERAL _Type operator()(_SrcType v) const {
		if (v < std::numeric_limits<_Type>::min())	return std::numeric_limits<_Type>::min();
		if (v > std::numeric_limits<_Type>::max())	return std::numeric_limits<_Type>::max();
		return static_cast<_Type>(v);
	}
};

}
}
