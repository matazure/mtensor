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

	MATAZURE_GENERAL constexpr int_t size() const { return dim; }

	MATAZURE_GENERAL static constexpr point zeros() {
		return{ 0 };
	}

	//MATAZURE_GENERAL static constexpr point ones() {
	//	return{ 1 };
	//}

public:
	value_type elements_[dim];
};

static_assert(std::is_pod<point<byte, 1>>::value, "point should be pod");

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
MATAZURE_GENERAL point<_T, _Dim> operator+(const point<_T, _Dim> &p) {
	point<_T, _Dim> temp;
	for (int_t i = 0; i < _Dim; ++i) {
		temp[i] = +p[i];
	}

	return temp;
}

template <typename _T, int_t _Dim>
MATAZURE_GENERAL point<_T, _Dim> operator-(const point<_T, _Dim> &p) {
	point<_T, _Dim> temp;
	for (int_t i = 0; i < _Dim; ++i) {
		temp[i] = -p[i];
	}

	return temp;
}

template <typename _T, int_t _Dim>
MATAZURE_GENERAL point<_T, _Dim>& operator++(point<_T, _Dim> &p) {
	for (int_t i = 0; i < _Dim; ++i) {
		++p[i];
	}

	return p;
}

template <typename _T, int_t _Dim>
MATAZURE_GENERAL point<_T, _Dim>& operator--(point<_T, _Dim> &p) {
	for (int_t i = 0; i < _Dim; ++i) {
		--p[i];
	}

	return p;
}

template <typename _DstType, typename _T, int_t _Dim>
MATAZURE_GENERAL point<_DstType, _Dim> point_cast(const point<_T, _Dim> &p) {
	point<_DstType, _Dim> re;
	for (int_t i = 0; i < _Dim; ++i) {
		re[i] = static_cast<_DstType>(p[i]);
	}

	return re;
}

template<int_t _Idx, class _Ty, int_t _Dim>
MATAZURE_GENERAL constexpr _Ty& get(point<_Ty, _Dim>& pt) {
	// return element at _Idx in point pt
	static_assert(_Idx < _Dim, "point index out of bounds");
	return (pt.elements_[_Idx]);
}

template<int_t _Idx, class _Ty, int_t _Dim>
MATAZURE_GENERAL constexpr const _Ty& get(const point<_Ty, _Dim>& pt){
	// return element at _Idx in point pt
	static_assert(_Idx < _Dim, "point index out of bounds");
	return (pt.elements_[_Idx]);
}

template<int_t _Idx, class _Ty, int_t _Dim>
MATAZURE_GENERAL constexpr _Ty&& get(point<_Ty, _Dim>&& pt) {
	// return element at _Idx in point pt
	static_assert(_Idx < _Dim, "point index out of bounds");
	return (move(pt.elements_[_Idx]));
}

template <int_t _Dim> using pointb = point<byte, _Dim>;
template <int_t _Dim> using points = point<short, _Dim>;
template <int_t _Dim> using pointi = point<int_t, _Dim>;
template <int_t _Dim> using pointf = point<float, _Dim>;

template <int_t _Dim>
inline MATAZURE_GENERAL pointi<_Dim> get_stride(pointi<_Dim> ex) {
	pointi<_Dim>  stride;
	stride[0] = ex[0];
	for (int_t i = 1; i < _Dim; ++i) {
		stride[i] = ex[i] * stride[i - 1];
	}
	return stride;
}

template <typename _Tuple, int_t dim = tuple_size<_Tuple>::value>
class point_viewer;

template <typename _Tuple>
class point_viewer<_Tuple, 3> : public _Tuple{
public:
	static_assert(tuple_size<_Tuple>::value == 3, "");
	const static int_t dim = 3;
	typedef decay_t<typename tuple_element<0, _Tuple>::type> value_type;
	typedef point<value_type, dim> point_type;

	point_viewer(const _Tuple &tp): _Tuple(tp){}

	point_viewer &operator=(const point_type &tp) {
		get<0>(*static_cast<_Tuple *>(this)) = tp[0];
		get<1>(*static_cast<_Tuple *>(this)) = tp[1];
		get<2>(*static_cast<_Tuple *>(this)) = tp[2];

		return *this;
	}

	operator point_type() const {
		point_type re;
		re[0] = get<0>(*static_cast<_Tuple *>(this));
		re[1] = get<1>(*static_cast<_Tuple *>(this));
		re[2] = get<2>(*static_cast<_Tuple *>(this));
	}
};



}
