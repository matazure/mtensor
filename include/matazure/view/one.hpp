#pragma once

#include <matazure/view/map.hpp>

namespace matazure {
namespace view {

template <typename _ValueType, int_t _Rank>
struct one_functor {
    MATAZURE_GENERAL _ValueType operator()(pointi<_Rank> idx) const { return _ValueType{1}; }
};

/**
 * @brief one a tensor to another value_type lambda_tensor
 *
 * support primitive type static_cast and point_cast.
 *
 * @param tensor the source tensor
 * @tparam _ValueType the dest tensor value type
 * @return a lambda_tensor whose value_type is _ValueType
 */
template <typename _ValueType, int_t _Rank, typename _Memory_type>
inline auto one(pointi<_Rank> shape, _Memory_type)
    -> decltype(make_lambda(shape, one_functor<_ValueType, _Rank>{}, _Memory_type{})) {
    return make_lambda(shape, one_functor<_ValueType, _Rank>{}, _Memory_type{});
}

}  // namespace view
}  // namespace matazure