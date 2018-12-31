#pragma once

#include <arm_neon.h>
#include <matazure/config.hpp>
#include <matazure/meta.hpp>

namespace matazure {
	inline namespace simd {

		template <typename _ValueType, int_t Rank>
		class neon_vector;

		template <>
		class neon_vector<float, 4> {
		public:
			/// neon_vector length
			static const int_t				rank = 4;
			typedef float					value_type;
			typedef value_type &			reference;
			typedef linear_index			index_type;
			typedef local_tag				memory_type;

		#ifdef __GNUC__

			/**
			* @brief accesses element by index
			* @param i element index
			* @return element const referece
			*/
			MATAZURE_GENERAL const float & operator[](int_t i) const {
				return data()[i];
			}

			/**
			* @brief accesses element by index
			* @param i element index
			* @return element referece
			*/
			MATAZURE_GENERAL float & operator[](int_t i) {
				return data()[i];
			}

		#endif

			/// return the length of neon_vector
			MATAZURE_GENERAL constexpr int_t size() const {
				return rank;
			}

			MATAZURE_GENERAL  const value_type * data() const {
				return reinterpret_cast<const value_type *>(&elements_);
			}

			MATAZURE_GENERAL  value_type * data() {
				return reinterpret_cast<value_type *>(&elements_);
			}

		public:

			float32x4_t elements_;
		};

	#define MATAZURE_NEON_FLOAT_BINARY_OPERATOR(tag, op) \
	inline neon_vector<float, 4> operator op(const neon_vector<float, 4> &lhs, const neon_vector<float, 4> &rhs){ \
		return neon_vector<float, 4>{ v##tag##q_f32(lhs.elements_, rhs.elements_) }; \
	}

	#define MATAZURE_NEON_FLOAT_ASSIGNMENT_OPERATOR(tag, op) \
	inline neon_vector<float, 4> operator op (neon_vector<float, 4> &lhs, const neon_vector<float, 4> &rhs){ \
		lhs = neon_vector<float, 4> { v##tag##q_f32(lhs.elements_, rhs.elements_) }; \
		return lhs; \
	}

#ifndef __aarch64__

	inline float32x4_t vdivq_f32(float32x4_t a, float32x4_t b) {
		float32x4_t recip0 = vrecpeq_f32(b);
		float32x4_t recip1 = vmulq_f32(recip0, vrecpsq_f32(recip0, b));
		return vmulq_f32(a, recip1);
	}

#endif

	inline float32x4_t vcneqq_f32(float32x4_t a, float32x4_t b) {
		return vmvnq_u32(vceqq_f32(a, b));
	}

	MATAZURE_NEON_FLOAT_BINARY_OPERATOR(add, +)
	MATAZURE_NEON_FLOAT_BINARY_OPERATOR(sub, -)
	MATAZURE_NEON_FLOAT_BINARY_OPERATOR(mul, *)
	MATAZURE_NEON_FLOAT_BINARY_OPERATOR(div, /)

	MATAZURE_NEON_FLOAT_BINARY_OPERATOR(cgt, >)
	MATAZURE_NEON_FLOAT_BINARY_OPERATOR(clt, <)
	MATAZURE_NEON_FLOAT_BINARY_OPERATOR(cge, >=)
	MATAZURE_NEON_FLOAT_BINARY_OPERATOR(cle, <=)
	MATAZURE_NEON_FLOAT_BINARY_OPERATOR(ceq, ==)
	MATAZURE_NEON_FLOAT_BINARY_OPERATOR(cneq, !=)

	template <typename _T, int_t _Rank>
	inline MATAZURE_GENERAL neon_vector<_T, _Rank> operator+(const neon_vector<_T, _Rank> &p) {
		return p;
	}

	template <typename _T, int_t _Rank>
	inline MATAZURE_GENERAL neon_vector<_T, _Rank> operator-(const neon_vector<_T, _Rank> &p) {
		neon_vector<_T, _Rank> temp;
		for (int_t i = 0; i < _Rank; ++i) {
			temp[i] = -p[i];
		}

		return temp;
	}

	template <typename _T, int_t _Rank>
	inline MATAZURE_GENERAL neon_vector<_T, _Rank>& operator++(neon_vector<_T, _Rank> &p) {
		for (int_t i = 0; i < _Rank; ++i) {
			++p[i];
		}

		return p;
	}

	template <typename _T, int_t _Rank>
	inline MATAZURE_GENERAL neon_vector<_T, _Rank>& operator--(neon_vector<_T, _Rank> &p) {
		for (int_t i = 0; i < _Rank; ++i) {
			--p[i];
		}

		return p;
	}

#ifdef __GNUC__

	template <>
	class neon_vector<float16_t, 8> {
	public:
		using value_type = float16_t;
		const static int_t rank = 8;

		float16x8_t elements_;
	};

#define MATAZURE_NEON_FLOAT16_BINARY_OPERATOR(tag, op) \
	inline neon_vector<float16_t, 8> operator op(const neon_vector<float16_t, 8> &lhs, const neon_vector<float16_t, 8> &rhs){ \
		return neon_vector<float16_t, 8>{ lhs.elements_ op rhs.elements_ }; \
	}

#define MATAZURE_NEON_FLOAT16_ASSIGNMENT_OPERATOR(tag, op) \
	inline neon_vector<float16_t, 8> operator op (neon_vector<float16_t, 8> &lhs, const neon_vector<float16_t, 8> &rhs){ \
		lhs = neon_vector<float16_t, 8> { lhs.elements_ op rhs.elements_ }; \
		return lhs; \
	}

	MATAZURE_NEON_FLOAT16_BINARY_OPERATOR(add, +)
	MATAZURE_NEON_FLOAT16_BINARY_OPERATOR(sub, -)
	MATAZURE_NEON_FLOAT16_BINARY_OPERATOR(mul, *)
	//MATAZURE_NEON_FLOAT16_BINARY_OPERATOR(div, /)

	//MATAZURE_NEON_FLOAT16_BINARY_OPERATOR(cmpgt, >)
	//MATAZURE_NEON_FLOAT16_BINARY_OPERATOR(cmplt, <)
	//MATAZURE_NEON_FLOAT16_BINARY_OPERATOR(cmpge, >=)
	//MATAZURE_NEON_FLOAT16_BINARY_OPERATOR(cmple, <=)
	//MATAZURE_NEON_FLOAT16_BINARY_OPERATOR(cmpeq, ==)
	//MATAZURE_NEON_FLOAT16_BINARY_OPERATOR(cmpneq, !=)

	MATAZURE_NEON_FLOAT16_ASSIGNMENT_OPERATOR(add, +=)
	MATAZURE_NEON_FLOAT16_ASSIGNMENT_OPERATOR(sub, -=)
	MATAZURE_NEON_FLOAT16_ASSIGNMENT_OPERATOR(mul, *=)
	//MATAZURE_NEON_FLOAT16_ASSIGNMENT_OPERATOR(div, /=)

#endif

	inline void fill(neon_vector<float, 4> & vec, float value) {
		vec.elements_ = vdupq_n_f32(value);
	}

} }

namespace matazure {

	using namespace simd;

	template <>
	struct zero<simd::neon_vector<float, 4>> {
		static simd::neon_vector<float, 4> value() {
			return simd::neon_vector<float, 4>{ vdupq_n_f32(0) };
		}
	};

#ifdef __GNUC__
	template <>
	struct zero<simd::neon_vector<float16_t, 8>> {
		static simd::neon_vector<float16_t, 8> value() {
			return simd::neon_vector<float16_t, 8>{ vdupq_n_p16(0) };
		}
	};
#endif


}

