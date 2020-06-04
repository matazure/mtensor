#pragma once

#include <matazure/view/map.hpp>

namespace matazure {
namespace view {
template <typename _OutValueType>
struct cast_functor {
    template <typename _InValueType>
    MATAZURE_GENERAL _OutValueType operator()(_InValueType v) const {
        return static_cast<_OutValueType>(v);
    }
};

template <typename _OutPointValueType, int_t _Rank>
struct cast_functor<point<_OutPointValueType, _Rank>> {
    template <typename _InPointValueType>
    MATAZURE_GENERAL point<_OutPointValueType, _Rank> operator()(
        const point<_InPointValueType, _Rank>& p) const {
        return point_cast<_OutPointValueType>(p);
    }
};

/**
 * @brief cast a tensor to another value_type lambda_tensor
 *
 * support primitive type static_cast and point_cast.
 *
 * @param tensor the source tensor
 * @tparam _ValueType the dest tensor value type
 * @return a lambda_tensor whose value_type is _ValueType
 */
template <typename _ValueType, typename _Tensor>
inline auto cast(_Tensor tensor, enable_if_t<is_tensor<_Tensor>::value>* = 0)
    -> decltype(map(tensor, cast_functor<_ValueType>())) {
    return map(tensor, cast_functor<_ValueType>());
}

}  // namespace view
}  // namespace matazure
