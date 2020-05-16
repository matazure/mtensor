#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace view {

template <typename _ValueType>
struct eye_functor {
    MATAZURE_GENERAL _ValueType operator()(point2i idx) const {
        if (idx[0] == idx[1]) {
            return _ValueType{1};
        } else {
            return _ValueType{0};
        }
    }
};

/**
 * @brief eye a tensor to another value_type lambda_tensor
 *
 * support primitive type static_cast and point_cast.
 *
 * @param tensor the source tensor
 * @tparam _ValueType the dest tensor value type
 * @return a lambda_tensor whose value_type is _ValueType
 */
template <typename _ValueType, typename _Memory_type>
inline auto eye(point2i shape, _Memory_type)
    -> decltype(make_lambda(shape, eye_functor<_ValueType>{}, _Memory_type{})) {
    return make_lambda(shape, eye_functor<_ValueType>{}, _Memory_type{});
}

}  // namespace view
}  // namespace matazure
