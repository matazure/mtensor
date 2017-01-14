#pragma once

#include <matazure/type_traits.hpp>

namespace matazure {

template <typename _Type, int_t _Dim>
class point {
public:
	static const int_t				dim = _Dim;
	typedef _Type					value_type;
	typedef value_type &			reference;
	typedef const value_type &		const_reference;
	typedef linear_access_t			access_type;
	typedef local_t					memory_type;

	MATAZURE_GENERAL constexpr const_reference operator[](int_t index) const { return elements_[index]; }

	MATAZURE_GENERAL reference operator[](int_t index) { return elements_[index]; }

	constexpr int_t size() const { return dim; }

	static constexpr point zeros() {
		return{ 0 };
	}

	static constexpr point ones() {
		return{ 1 };
	}

public:
	value_type elements_[dim];
};

#define POINT_BINARY_OPERATOR(op) \
template <typename _T, int_t _Dim> \
inline MATAZURE_GENERAL auto operator op(const point<_T, _Dim> &lhs, const point<_T, _Dim> &rhs)->point<decltype(lhs[0] op rhs[0]), _Dim> { \
	point<decltype(lhs[0] op rhs[0]), _Dim> re; \
	for (int_t i = 0; i < _Dim; ++i) { \
		re[i] = lhs[i] op rhs[i]; \
	} \
	return re; \
}

#define POINT_WITH_VALUE_BINARY_OPERATOR(op) \
template <typename _T, int_t _Dim>  \
inline MATAZURE_GENERAL auto operator op(const point<_T, _Dim> &container, typename point<_T, _Dim>::value_type value)->point<decltype(container[0] op value), _Dim> { \
	point<decltype(container[0] op value), _Dim> re; \
	for (int_t i = 0; i < _Dim; ++i) { \
		re[i] = container[i] op value; \
	} \
\
return re; \
}\
\
template <typename _T, int_t _Dim>  \
inline MATAZURE_GENERAL auto operator op(typename point<_T, _Dim>::value_type value, const point<_T, _Dim> &container)->point<decltype(value op container[0]), _Dim> { \
	point<decltype(value op container[0]), _Dim> re; \
	for (int_t i = 0; i < _Dim; ++i) { \
		re[i] = container[i] op value; \
	} \
\
return re; \
}

//Arithmetic
POINT_BINARY_OPERATOR(+)
POINT_BINARY_OPERATOR(-)
POINT_BINARY_OPERATOR(*)
POINT_BINARY_OPERATOR(/)
POINT_BINARY_OPERATOR(%)
POINT_WITH_VALUE_BINARY_OPERATOR(+)
POINT_WITH_VALUE_BINARY_OPERATOR(-)
POINT_WITH_VALUE_BINARY_OPERATOR(*)
POINT_WITH_VALUE_BINARY_OPERATOR(/)
POINT_WITH_VALUE_BINARY_OPERATOR(%)
//Bit
POINT_BINARY_OPERATOR(<<)
POINT_BINARY_OPERATOR(>>)
POINT_BINARY_OPERATOR(|)
POINT_BINARY_OPERATOR(&)
POINT_BINARY_OPERATOR(^)
POINT_WITH_VALUE_BINARY_OPERATOR(<<)
POINT_WITH_VALUE_BINARY_OPERATOR(>>)
POINT_WITH_VALUE_BINARY_OPERATOR(|)
POINT_WITH_VALUE_BINARY_OPERATOR(&)
POINT_WITH_VALUE_BINARY_OPERATOR(^)
//Logic
POINT_BINARY_OPERATOR(||)
POINT_BINARY_OPERATOR(&&)
POINT_WITH_VALUE_BINARY_OPERATOR(||)
POINT_WITH_VALUE_BINARY_OPERATOR(&&)
//Compapre
POINT_BINARY_OPERATOR(>)
POINT_BINARY_OPERATOR(<)
POINT_BINARY_OPERATOR(>=)
POINT_BINARY_OPERATOR(<=)
POINT_BINARY_OPERATOR(==)
POINT_BINARY_OPERATOR(!=)
POINT_WITH_VALUE_BINARY_OPERATOR(>)
POINT_WITH_VALUE_BINARY_OPERATOR(<)
POINT_WITH_VALUE_BINARY_OPERATOR(>=)
POINT_WITH_VALUE_BINARY_OPERATOR(<=)
POINT_WITH_VALUE_BINARY_OPERATOR(==)
POINT_WITH_VALUE_BINARY_OPERATOR(!=)

template <typename _T, int_t _Dim>
point<_T, _Dim> operator+(const point<_T, _Dim> &p) {
	point<_T, _Dim> temp;
	for (int_t i = 0; i < _Dim; ++i) {
		temp[i] = +p[i];
	}

	return temp;
}

template <typename _T, int_t _Dim>
point<_T, _Dim> operator-(const point<_T, _Dim> &p) {
	point<_T, _Dim> temp;
	for (int_t i = 0; i < _Dim; ++i) {
		temp[i] = -p[i];
	}

	return temp;
}

template <typename _T, int_t _Dim>
point<_T, _Dim>& operator++(point<_T, _Dim> &p) {
	for (int_t i = 0; i < _Dim; ++i) {
		++p[i];
	}

	return p;
}

template <typename _T, int_t _Dim>
point<_T, _Dim>& operator--(point<_T, _Dim> &p) {
	for (int_t i = 0; i < _Dim; ++i) {
		--p[i];
	}

	return p;
}

template <int_t _Dim>
using pointi = point<int_t, _Dim>;

template <int_t _Dim>
using pointf = point<float, _Dim>;

template <int_t _Dim>
inline MATAZURE_GENERAL pointi<_Dim> get_stride(pointi<_Dim> ex) {
 pointi<_Dim>  stride;
	stride[0] = ex[0];
	for (int_t i = 1; i < _Dim; ++i) {
		stride[i] = ex[i] * stride[i - 1];
	}
	return stride;
}

template <int_t _Dim>
inline MATAZURE_GENERAL typename pointi<_Dim>::value_type index2offset(const pointi<_Dim> &id, const pointi<_Dim> &stride, first_major_t) {
	typename pointi<_Dim>::value_type offset = id[0];
	for (int_t i = 1; i < _Dim; ++i) {
		offset += id[i] * stride[i - 1];
	}

	return offset;
};

template <int_t _Dim>
inline MATAZURE_GENERAL pointi<_Dim> offset2index(typename pointi<_Dim>::value_type offset, const pointi<_Dim> &stride, first_major_t) {
 pointi<_Dim> id;
	for (int_t i = _Dim - 1; i > 0; --i) {
		id[i] = offset / stride[i - 1];
		offset = offset % stride[i - 1];
	}
	id[0] = offset;

	return id;
}

template <int_t _Dim>
inline MATAZURE_GENERAL typename pointi<_Dim>::value_type index2offset(const pointi<_Dim> &id, const pointi<_Dim> &stride, last_major_t) {
	typename pointi<_Dim>::value_type offset = id[_Dim - 1];
	for (int_t i = 1; i < _Dim; ++i) {
		offset += id[_Dim - 1 - i] * stride[i - 1];
	}

	return offset;
};

template <int_t _Dim>
inline MATAZURE_GENERAL pointi<_Dim> offset2index(typename pointi<_Dim>::value_type offset, const pointi<_Dim> &stride, last_major_t) {
 pointi<_Dim> id;
	for (int_t i = _Dim - 1; i > 0; --i) {
		id[_Dim - 1 - i] = offset / stride[i - 1];
		offset = offset % stride[i - 1];
	}
	id[_Dim - 1] = offset;

	return id;
}

namespace detail {

	//template <int_t _DimI, typename _ValueType, int_t _Dim>
	//auto slice(const point<_ValueType, _Dim> &pt) {
	//	point<_ValueType, _Dim - 1> re;
	//	for ()
	//}

}

}
