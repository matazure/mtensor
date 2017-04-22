#pragma once

#include <matazure/tensor.hpp>
#include <matazure/algorithm.hpp>
#include <limits>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace op {

template <typename _Type>
struct saturate_convert {
	template <typename _SrcType>
	MATAZURE_GENERAL _Type operator()(_SrcType v) const {
		if (v < std::numeric_limits<_SrcType>::min())	return std::numeric_limits<_SrcType>::min();
		if (v > std::numeric_limits<_SrcType>::max())	return std::numeric_limits<_SrcType>::max();
		return static_cast<_Type>(v);
	}
};

}
}