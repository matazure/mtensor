#pragma once

#include <matazure/view/map.hpp>

namespace matazure {
namespace view {

template <typename _ValueType, typename _Fun>
struct binary_functor {
    _Fun fun;

    MATAZURE_GENERAL bool operator()(_ValueType v) const { return fun(v); }
};

template <typename _Tensor, typename _Fun>
inline auto binary(_Tensor tensor, _Fun fun)
    -> decltype(map(tensor, binary_functor<typename _Tensor::value_type, _Fun>{fun})) {
    return map(tensor, binary_functor<typename _Tensor::value_type, _Fun>{fun});
}

}  // namespace view
}  // namespace matazure