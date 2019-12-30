#pragma once

#include <xmmintrin.h>
#include <matazure/config.hpp>
#include <matazure/point.hpp>

namespace matazure { namespace simd {

	template <typename _ElementType>
	class sse_wrapper;

	template <typename _ValueType, int_t _Rank>
	class alignas(16) sse_wrapper<point<_ValueType, _Rank>> : public point<_ValueType, _Rank> {
	public:
		//_ElementType is not supported
		static_assert(sizeof(_ValueType) * _Rank * 8 == 128, "");

		typedef point<_ValueType, _Rank> element_type;
	};

	template <typename _ValueType, int_t _Rank>
	sse_wrapper<point<_ValueType, _Rank>> operator+(const sse_wrapper<point<_ValueType, _Rank>> &lhs, const sse_wrapper<point<_ValueType, _Rank>> &rhs) {
		auto tmp = _mm_add_ps(*reinterpret_cast<const __m128 *>(&lhs), *reinterpret_cast<const __m128 *>(&rhs));
		return *reinterpret_cast<sse_wrapper<point<_ValueType, _Rank>> *>(&tmp);
	}

	//template <typename _ValueType, int_t _Rank>
	//sse_wrapper<point<_ValueType, _Rank>> operator+(const sse_wrapper<point<_ValueType, _Rank>> &lhs, const sse_wrapper<point<_ValueType, _Rank>> &rhs) {

	//}

	//template <typename _ValueType, int_t _Rank>
	//sse_wrapper<point<_ValueType, _Rank>> operator+(const sse_wrapper<point<_ValueType, _Rank>> &lhs, const sse_wrapper<point<_ValueType, _Rank>> &rhs) {

	//}

	//template <typename _ValueType, int_t _Rank>
	//sse_wrapper<point<_ValueType, _Rank>> operator+(const sse_wrapper<point<_ValueType, _Rank>> &lhs, const sse_wrapper<point<_ValueType, _Rank>> &rhs) {

	//}

	//template <typename _ValueType, int_t _Rank>
	//sse_wrapper<point<_ValueType, _Rank>> operator+(const sse_wrapper<point<_ValueType, _Rank>> &lhs, const sse_wrapper<point<_ValueType, _Rank>> &rhs) {

	//}

} }
