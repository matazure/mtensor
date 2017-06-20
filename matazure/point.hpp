#pragma once

#include <matazure/config.hpp>

namespace matazure {

template <typename _Type, int_t _Rank>
class point {
public:
	static const int_t				rank = _Rank;
	typedef _Type					value_type;
	typedef value_type &			reference;
	typedef const value_type &		const_reference;
	typedef linear_access_t			access_type;
	typedef local_t					memory_type;

	MATAZURE_GENERAL constexpr const_reference operator[](int_t index) const { return elements_[index]; }

	MATAZURE_GENERAL reference operator[](int_t index) { return elements_[index]; }

	MATAZURE_GENERAL constexpr int_t size() const { return rank; }

	MATAZURE_GENERAL static constexpr point zeros() {
		return { 0 };
	}

	MATAZURE_GENERAL static point all(value_type v) {
		point re{};
		fill(re, v);
		return re;
	}

public:
	value_type elements_[rank];
};

static_assert(std::is_pod<point<byte, 1>>::value, "point should be pod");

#define POINT_BINARY_OPERATOR(op) \
template <typename _T, int_t _Rank> \
inline MATAZURE_GENERAL auto operator op(const point<_T, _Rank> &lhs, const point<_T, _Rank> &rhs)->point<decltype(lhs[0] op rhs[0]), _Rank> { \
	point<decltype(lhs[0] op rhs[0]), _Rank> re; \
	for (int_t i = 0; i < _Rank; ++i) { \
		re[i] = lhs[i] op rhs[i]; \
	} \
	return re; \
}

#define POINT_WITH_VALUE_BINARY_OPERATOR(op) \
template <typename _T, int_t _Rank>  \
inline MATAZURE_GENERAL auto operator op(const point<_T, _Rank> &container, typename point<_T, _Rank>::value_type value)->point<decltype(container[0] op value), _Rank> { \
	point<decltype(container[0] op value), _Rank> re; \
	for (int_t i = 0; i < _Rank; ++i) { \
		re[i] = container[i] op value; \
	} \
\
return re; \
}\
\
template <typename _T, int_t _Rank>  \
inline MATAZURE_GENERAL auto operator op(typename point<_T, _Rank>::value_type value, const point<_T, _Rank> &container)->point<decltype(value op container[0]), _Rank> { \
	point<decltype(value op container[0]), _Rank> re; \
	for (int_t i = 0; i < _Rank; ++i) { \
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

template <typename _T, int_t _Rank>
MATAZURE_GENERAL point<_T, _Rank> operator+(const point<_T, _Rank> &p) {
	point<_T, _Rank> temp;
	for (int_t i = 0; i < _Rank; ++i) {
		temp[i] = +p[i];
	}

	return temp;
}

template <typename _T, int_t _Rank>
MATAZURE_GENERAL point<_T, _Rank> operator-(const point<_T, _Rank> &p) {
	point<_T, _Rank> temp;
	for (int_t i = 0; i < _Rank; ++i) {
		temp[i] = -p[i];
	}

	return temp;
}

template <typename _T, int_t _Rank>
MATAZURE_GENERAL point<_T, _Rank>& operator++(point<_T, _Rank> &p) {
	for (int_t i = 0; i < _Rank; ++i) {
		++p[i];
	}

	return p;
}

template <typename _T, int_t _Rank>
MATAZURE_GENERAL point<_T, _Rank>& operator--(point<_T, _Rank> &p) {
	for (int_t i = 0; i < _Rank; ++i) {
		--p[i];
	}

	return p;
}

template <typename _DstType, typename _T, int_t _Rank>
MATAZURE_GENERAL point<_DstType, _Rank> point_cast(const point<_T, _Rank> &p) {
	point<_DstType, _Rank> re;
	for (int_t i = 0; i < _Rank; ++i) {
		re[i] = static_cast<_DstType>(p[i]);
	}

	return re;
}

template<int_t _Idx, class _Ty, int_t _Rank>
MATAZURE_GENERAL constexpr _Ty& get(point<_Ty, _Rank>& pt) {
	// return element at _Idx in point pt
	static_assert(_Idx < _Rank, "point index out of bounds");
	return (pt.elements_[_Idx]);
}

template<int_t _Idx, class _Ty, int_t _Rank>
MATAZURE_GENERAL constexpr const _Ty& get(const point<_Ty, _Rank>& pt){
	// return element at _Idx in point pt
	static_assert(_Idx < _Rank, "point index out of bounds");
	return (pt.elements_[_Idx]);
}

template<int_t _Idx, class _Ty, int_t _Rank>
MATAZURE_GENERAL constexpr _Ty&& get(point<_Ty, _Rank>&& pt) {
	// return element at _Idx in point pt
	static_assert(_Idx < _Rank, "point index out of bounds");
	return (move(pt.elements_[_Idx]));
}

template <int_t _Rank> using pointb = point<byte, _Rank>;
template <int_t _Rank> using points = point<short, _Rank>;
template <int_t _Rank> using pointi = point<int_t, _Rank>;
template <int_t _Rank> using pointf = point<float, _Rank>;

template <typename _Ty, int_t _Rank>
inline MATAZURE_GENERAL bool equal(point<_Ty, _Rank> lhs, point<_Ty, _Rank> rhs) {
	for (int_t i = 0; i < lhs.size(); ++i) {
		if ((lhs[i] != rhs[i])) return false;
	}
	return true;
}

template <typename _T, int_t _Rank>
struct zero<point<_T, _Rank>>{
	static constexpr point<_T, _Rank> value(){
		return {0};
	};
};

template <int_t _Rank>
inline MATAZURE_GENERAL pointi<_Rank> accumulate_stride(pointi<_Rank> ex) {
	pointi<_Rank>  stride;
	stride[0] = ex[0];
	for (int_t i = 1; i < _Rank; ++i) {
		stride[i] = ex[i] * stride[i - 1];
	}
	return stride;
}

template <typename _ValueType, int_t _Rank>
inline MATAZURE_GENERAL bool inside(point<_ValueType, _Rank> idx, point<_ValueType, _Rank> origin, point<_ValueType, _Rank> extent) {
	for (int_t i = 0; i < _Rank; ++i) {
		if (MATAZURE_LIKELY(idx[i] < origin[i] || idx[i] >= extent[i] ))
			return false;
	}

	return true;
}

template <typename _ValueType, int_t _Rank>
inline MATAZURE_GENERAL bool outside(point<_ValueType, _Rank> idx, point<_ValueType, _Rank> origin, point<_ValueType, _Rank> extent) {
	for (int_t i = 0; i < _Rank; ++i) {
		if (MATAZURE_LIKELY(idx[i] < origin[i] || idx[i] >= extent[i] ))
			return true;
	}

	return false;
}

template <typename _Tuple, int_t rank = tuple_size<_Tuple>::value>
class point_viewer;

template <typename _Tuple>
class point_viewer<_Tuple, 3> : public _Tuple{
public:
	static_assert(tuple_size<_Tuple>::value == 3, "");
	const static int_t rank = 3;
	typedef decay_t<typename tuple_element<0, _Tuple>::type> value_type;
	typedef point<value_type, rank> point_type;

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
