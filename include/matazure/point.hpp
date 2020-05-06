#pragma once

#include <matazure/type_traits.hpp>

namespace matazure {

/**
 * @brief a point class, like as std::array.
 *
 * uses point<int_t, rank> to represent an array index, a tensor shape, stride etc.
 * we clould look point as fixed length vector or array
 *
 * @tparam _ValueType the point element value type, must be pod
 * @tparam _Rank the rank of point, it equals to point length/size
 */
template <typename _ValueType, int_t _Rank>
class point {
   public:
    /// point length
    static const int_t rank = _Rank;
    typedef _ValueType value_type;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef linear_index index_type;
    typedef local_tag memory_type;

    /**
     * @brief accesses element by index
     * @param i element index
     * @return element const referece
     */
    MATAZURE_GENERAL constexpr const_reference operator[](int_t i) const { return elements_[i]; }

    /**
     * @brief accesses element by index
     * @param i element index
     * @return element referece
     */
    MATAZURE_GENERAL reference operator[](int_t i) { return elements_[i]; }

    /// return the length of point
    MATAZURE_GENERAL constexpr int_t size() const { return rank; }

    /// return a point whose elements are v
    MATAZURE_GENERAL static point all(value_type v) {
        point re{};
        for (int_t i = 0; i < re.size(); ++i) {
            re[i] = v;
        }
        return re;
    }

    MATAZURE_GENERAL value_type* data() { return elements_; }

    MATAZURE_GENERAL const value_type* data() const { return elements_; }

   public:
    value_type elements_[rank];
};

static_assert(std::is_pod<point<byte, 1>>::value, "point should be pod");

// binary opertor
#define MATAZURE_POINT_BINARY_OPERATOR(op)                                                \
    template <typename _T, int_t _Rank>                                                   \
    inline MATAZURE_GENERAL auto operator op(const point<_T, _Rank>& lhs,                 \
                                             const point<_T, _Rank>& rhs)                 \
        ->point<decltype(lhs[0] op rhs[0]), _Rank> {                                      \
        point<decltype(lhs[0] op rhs[0]), _Rank> re;                                      \
        for (int_t i = 0; i < _Rank; ++i) {                                               \
            re[i] = lhs[i] op rhs[i];                                                     \
        }                                                                                 \
        return re;                                                                        \
    }                                                                                     \
                                                                                          \
    template <typename _T, int_t _Rank>                                                   \
    inline MATAZURE_GENERAL auto operator op(const point<_T, _Rank>& container,           \
                                             typename point<_T, _Rank>::value_type value) \
        ->point<decltype(container[0] op value), _Rank> {                                 \
        point<decltype(container[0] op value), _Rank> re;                                 \
        for (int_t i = 0; i < _Rank; ++i) {                                               \
            re[i] = container[i] op value;                                                \
        }                                                                                 \
                                                                                          \
        return re;                                                                        \
    }                                                                                     \
                                                                                          \
    template <typename _T, int_t _Rank>                                                   \
    inline MATAZURE_GENERAL auto operator op(typename point<_T, _Rank>::value_type value, \
                                             const point<_T, _Rank>& container)           \
        ->point<decltype(value op container[0]), _Rank> {                                 \
        point<decltype(value op container[0]), _Rank> re;                                 \
        for (int_t i = 0; i < _Rank; ++i) {                                               \
            re[i] = container[i] op value;                                                \
        }                                                                                 \
                                                                                          \
        return re;                                                                        \
    }

// assignment operators
#define MATAZURE_POINT_ASSIGNMENT_OPERATOR(op)                                                   \
    template <typename _T, int_t _Rank>                                                          \
    inline MATAZURE_GENERAL auto operator op(point<_T, _Rank>& lhs, const point<_T, _Rank>& rhs) \
        ->point<_T, _Rank> {                                                                     \
        for (int_t i = 0; i < _Rank; ++i) {                                                      \
            lhs[i] op rhs[i];                                                                    \
        }                                                                                        \
        return lhs;                                                                              \
    }                                                                                            \
                                                                                                 \
    template <typename _T, int_t _Rank>                                                          \
    inline MATAZURE_GENERAL auto operator op(point<_T, _Rank>& container, _T value)              \
        ->point<_T, _Rank> {                                                                     \
        for (int_t i = 0; i < _Rank; ++i) {                                                      \
            container[i] op value;                                                               \
        }                                                                                        \
                                                                                                 \
        return container;                                                                        \
    }

// Arithmetic
MATAZURE_POINT_BINARY_OPERATOR(+)
MATAZURE_POINT_BINARY_OPERATOR(-)
MATAZURE_POINT_BINARY_OPERATOR(*)
MATAZURE_POINT_BINARY_OPERATOR(/)
MATAZURE_POINT_BINARY_OPERATOR(%)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(+=)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(-=)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(*=)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(/=)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(%=)
// Bit
MATAZURE_POINT_BINARY_OPERATOR(<<)
MATAZURE_POINT_BINARY_OPERATOR(>>)
MATAZURE_POINT_BINARY_OPERATOR(|)
MATAZURE_POINT_BINARY_OPERATOR(&)
MATAZURE_POINT_BINARY_OPERATOR (^)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(<<=)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(>>=)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(|=)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(&=)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(^=)
// Logic
MATAZURE_POINT_BINARY_OPERATOR(||)
MATAZURE_POINT_BINARY_OPERATOR(&&)
// Compapre
MATAZURE_POINT_BINARY_OPERATOR(>)
MATAZURE_POINT_BINARY_OPERATOR(<)
MATAZURE_POINT_BINARY_OPERATOR(>=)
MATAZURE_POINT_BINARY_OPERATOR(<=)
MATAZURE_POINT_BINARY_OPERATOR(==)
MATAZURE_POINT_BINARY_OPERATOR(!=)

template <typename _T, int_t _Rank>
inline MATAZURE_GENERAL point<_T, _Rank> operator+(const point<_T, _Rank>& p) {
    return p;
}

template <typename _T, int_t _Rank>
inline MATAZURE_GENERAL point<_T, _Rank> operator-(const point<_T, _Rank>& p) {
    point<_T, _Rank> temp;
    for (int_t i = 0; i < _Rank; ++i) {
        temp[i] = -p[i];
    }

    return temp;
}

template <typename _T, int_t _Rank>
inline MATAZURE_GENERAL point<_T, _Rank>& operator++(point<_T, _Rank>& p) {
    for (int_t i = 0; i < _Rank; ++i) {
        ++p[i];
    }

    return p;
}

template <typename _T, int_t _Rank>
inline MATAZURE_GENERAL point<_T, _Rank>& operator--(point<_T, _Rank>& p) {
    for (int_t i = 0; i < _Rank; ++i) {
        --p[i];
    }

    return p;
}

template <typename _ValueTypeDst, typename _ValueTypeSrc, int_t _Rank>
inline MATAZURE_GENERAL point<_ValueTypeDst, _Rank> point_cast(
    const point<_ValueTypeSrc, _Rank>& p) {
    point<_ValueTypeDst, _Rank> re;
    for (int_t i = 0; i < _Rank; ++i) {
        re[i] = static_cast<_ValueTypeDst>(p[i]);
    }

    return re;
}

/// get point element like as std::get
template <int_t _Idx, class _Ty, int_t _Rank>
inline MATAZURE_GENERAL constexpr _Ty& get(point<_Ty, _Rank>& pt) {
    // return element at _Idx in point pt
    static_assert(_Idx < _Rank, "point index out of bounds");
    return (pt.elements_[_Idx]);
}

template <int_t _Idx, class _Ty, int_t _Rank>
inline MATAZURE_GENERAL constexpr const _Ty& get(const point<_Ty, _Rank>& pt) {
    // return element at _Idx in point pt
    static_assert(_Idx < _Rank, "point index out of bounds");
    return (pt.elements_[_Idx]);
}

template <int_t _Idx, class _Ty, int_t _Rank>
inline MATAZURE_GENERAL constexpr _Ty&& get(point<_Ty, _Rank>&& pt) {
    // return element at _Idx in point pt
    static_assert(_Idx < _Rank, "point index out of bounds");
    return (move(pt.elements_[_Idx]));
}

template <int_t _Rank>
using pointb = point<byte, _Rank>;
template <int_t _Rank>
using points = point<short, _Rank>;
template <int_t _Rank>
using pointi = point<int_t, _Rank>;
template <int_t _Rank>
using pointf = point<float, _Rank>;
template <int_t _Rank>
using pointd = point<float, _Rank>;

using point1b = pointb<1>;
using point2b = pointb<2>;
using point3b = pointb<3>;
using point4b = pointb<4>;
using point1s = points<1>;
using point2s = points<2>;
using point3s = points<3>;
using point4s = points<4>;
using point1i = pointi<1>;
using point2i = pointi<2>;
using point3i = pointi<3>;
using point4i = pointi<4>;
using point1f = pointf<1>;
using point2f = pointf<2>;
using point3f = pointf<3>;
using point4f = pointf<4>;
using point1d = pointd<1>;
using point2d = pointd<2>;
using point3d = pointd<3>;
using point4d = pointd<4>;

/// if two points are equal elementwise, return true, others false.
template <typename _Ty, int_t _Rank>
inline MATAZURE_GENERAL bool equal(const point<_Ty, _Rank>& lhs, const point<_Ty, _Rank>& rhs) {
    for (int_t i = 0; i < lhs.size(); ++i) {
        if ((lhs[i] != rhs[i])) return false;
    }
    return true;
}

/// special zero for point
template <typename _T, int_t _Rank>
struct zero<point<_T, _Rank>> {
    MATAZURE_GENERAL static constexpr point<_T, _Rank> value() { return {0}; };
};

/// return cumulative prod of point elements
template <int_t _Rank>
inline MATAZURE_GENERAL pointi<_Rank> cumulative_prod(pointi<_Rank> ex) {
    pointi<_Rank> stride;
    stride[0] = ex[0];
    for (int_t i = 1; i < _Rank; ++i) {
        stride[i] = ex[i] * stride[i - 1];
    }
    return stride;
}

template <typename _T, int_t _Rank>
inline constexpr point<_T, _Rank> reverse(point<_T, _Rank> pt);

template <typename _T>
inline constexpr point<_T, 1> reverse(point<_T, 1> pt) {
    return pt;
}

template <typename _T>
inline constexpr point<_T, 2> reverse(point<_T, 2> pt) {
    return {pt[1], pt[0]};
}

template <typename _T>
inline constexpr point<_T, 3> reverse(point<_T, 3> pt) {
    return {pt[2], pt[1], pt[0]};
}

template <typename _T>
inline constexpr point<_T, 4> reverse(point<_T, 4> pt) {
    return {pt[3], pt[2], pt[1], pt[0]};
}

// pointi<3>
template <int_t _SliceDimIdx>
inline pointi<2> slice_point(pointi<3> pt);

template <>
inline pointi<2> slice_point<0>(pointi<3> pt) {
    return pointi<2>{get<1>(pt), get<2>(pt)};
}

template <>
inline pointi<2> slice_point<1>(pointi<3> pt) {
    return pointi<2>{get<0>(pt), get<2>(pt)};
}

template <>
inline pointi<2> slice_point<2>(pointi<3> pt) {
    return pointi<2>{get<0>(pt), get<1>(pt)};
}

// pointi<2>
template <int_t _SliceDimIdx>
inline pointi<1> slice_point(pointi<2> pt);

template <>
inline pointi<1> slice_point<0>(pointi<2> pt) {
    return pointi<1>{get<1>(pt)};
}

template <>
inline pointi<1> slice_point<1>(pointi<2> pt) {
    return pointi<1>{get<0>(pt)};
}

template <int_t _CatDimIdx>
inline pointi<2> cat_point(pointi<1> pt, int_t cat_i);

template <>
inline pointi<2> cat_point<0>(pointi<1> pt, int_t cat_i) {
    return pointi<2>{cat_i, get<0>(pt)};
}

template <>
inline pointi<2> cat_point<1>(pointi<1> pt, int_t cat_i) {
    return pointi<2>{get<0>(pt), cat_i};
}

template <int_t _CatDimIdx>
inline pointi<3> cat_point(pointi<2> pt, int_t cat_i);

template <>
inline pointi<3> cat_point<0>(pointi<2> pt, int_t cat_i) {
    return pointi<3>{cat_i, get<0>(pt), get<1>(pt)};
}

template <>
inline pointi<3> cat_point<1>(pointi<2> pt, int_t cat_i) {
    return pointi<3>{get<0>(pt), cat_i, get<1>(pt)};
}

template <>
inline pointi<3> cat_point<2>(pointi<2> pt, int_t cat_i) {
    return pointi<3>{get<0>(pt), get<1>(pt), cat_i};
}

}  // namespace matazure
