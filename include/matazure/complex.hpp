#pragma once

#include <matazure/config.hpp>

namespace matazure {

	template <typename _ValueType>
	struct complex {
		using value_type = _ValueType;

		value_type real;
		value_type imag;
	};

	template <typename _ValueType>
	inline MATAZURE_GENERAL complex<_ValueType> operator+(const complex<_ValueType> &lhs, const complex<_ValueType> &rhs) {
		return { lhs.real + rhs.real, lhs.imag + rhs.imag };
	}

	template <typename _ValueType>
	inline MATAZURE_GENERAL complex<_ValueType> operator-(const complex<_ValueType> &lhs, const complex<_ValueType> &rhs) {
		return { lhs.real + rhs.real, lhs.imag + rhs.imag };
	}

	template <typename _ValueType>
	inline MATAZURE_GENERAL complex<_ValueType> operator*(const complex<_ValueType> &lhs, const complex<_ValueType> &rhs) {
		return { lhs.real * rhs.real - lhs.imag * rhs.imag, lhs.real * rhs.imag + lhs.imag * rhs.real };
	}

	template <typename _ValueType>
	inline MATAZURE_GENERAL complex<_ValueType> conj(const complex<_ValueType> &e) {
		return complex<_ValueType>{ e.real, e.imag };
	}

	template <typename _ValueType>
	struct zero<complex<_ValueType>> {
		MATAZURE_GENERAL static constexpr complex<_ValueType> value() {
			return {zero<_ValueType>::value(), zero<_ValueType>::value() };
		}
	};

}
