#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace view {

template <typename _ValueType, int_t _Rank>
struct zero_functor {
    MATAZURE_GENERAL _ValueType operator()(pointi<_Rank> idx) const { return _ValueType{0}; }
};

/**
 * @brief zero a tensor to another value_type lambda_tensor
 *
 * support primitive type static_cast and point_cast.
 *
 * @param tensor the source tensor
 * @tparam _ValueType the dest tensor value type
 * @return a lambda_tensor whose value_type is _ValueType
 */
template <typename _ValueType, int_t _Rank, typename _RuntimeType>
inline auto zero(pointi<_Rank> shape, _RuntimeType)
    -> decltype(make_lambda(shape, zero_functor<_ValueType, _Rank>{}, _RuntimeType{})) {
    return make_lambda(shape, zero_functor<_ValueType, _Rank>{}, _RuntimeType{});
}

}  // namespace view
}  // namespace matazure
