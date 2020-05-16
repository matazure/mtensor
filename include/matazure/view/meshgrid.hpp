#pragma once

#include <matazure/view/map.hpp>

namespace matazure {
namespace view {

template <typename _ValueType>
struct meshgrid_functor {
    meshgrid_functor(tensor<_ValueType, 1> x, tensor<_ValueType, 1> y) : x_(x), y_(y) {}

    MATAZURE_GENERAL point<_ValueType, 2> operator()(pointi<2> idx) const {
        return point<_ValueType, 2>{x_[idx[0]], y_[idx[1]]};
    }

   private:
    tensor<_ValueType, 1> x_;
    tensor<_ValueType, 1> y_;
};

/**
 * @brief meshgrid a tensor to another value_type lambda_tensor
 *
 * support primitive type static_cast and point_cast.
 *
 * @param tensor the source tensor
 * @tparam _ValueType the dest tensor value type
 * @return a lambda_tensor whose value_type is _ValueType
 */
template <typename _T0, typename _T1>
inline auto meshgrid(_T0 x, _T1 y)
    -> decltype(make_lambda(pointi<2>{x.size(), y.size()},
                            meshgrid_functor<typename _T0::value_type>(x, y),
                            typename _T0::runtime_type{})) {
    return make_lambda(pointi<2>{x.size(), y.size()},
                       meshgrid_functor<typename _T0::value_type>(x, y),
                       typename _T0::runtime_type{});
}

}  // namespace view
}  // namespace matazure
