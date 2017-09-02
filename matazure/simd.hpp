#pragma once

#include <matazure/config.hpp>
#include <emmintrin.h>

#define MATAZURE_SSE_FLOAT_BINARY_OPERATOR(op, tag) \
inline __m128 operator op(const __m128 &lhs, const __m128 &rhs){ \
	return _mm_##tag##_ps(lhs, rhs); \
}

#define MATAZURE_SSE_FLOAT_ASSIGNMENT_OPERATOR(op, tag) \
inline __m128 operator op (__m128 &lhs, const __m128 &rhs){ \
	lhs = _mm_##tag##_ps(lhs, rhs); \
	return lhs; \
}

namespace matazure{
inline namespace sse{

#ifdef _MSC_VER
MATAZURE_SSE_FLOAT_BINARY_OPERATOR(+, add)
MATAZURE_SSE_FLOAT_BINARY_OPERATOR(-, sub)
MATAZURE_SSE_FLOAT_BINARY_OPERATOR(*, mul)
MATAZURE_SSE_FLOAT_BINARY_OPERATOR(/, div)
MATAZURE_SSE_FLOAT_ASSIGNMENT_OPERATOR(+=, add)
MATAZURE_SSE_FLOAT_ASSIGNMENT_OPERATOR(-=, sub)
MATAZURE_SSE_FLOAT_ASSIGNMENT_OPERATOR(*=, mul)
MATAZURE_SSE_FLOAT_ASSIGNMENT_OPERATOR(/=, div)
#endif

}

template <>
struct zero<__m128>{
	static __m128 value() {
		return _mm_setzero_ps();
	}
};

}
