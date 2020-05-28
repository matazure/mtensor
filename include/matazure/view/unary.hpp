#pragma once

#include <matazure/view/map.hpp>

namespace matazure {
namespace view {

// clang-format off
#define MTENSOR_UNARY_VIEW(fun)                                                       \
    template <typename _ValueType>                                                    \
    struct fun##_functor {                                                            \
        MATAZURE_GENERAL _ValueType operator()(_ValueType v) const { return ::fun (v); } \
    };                                                                                \
                                                                                      \
    template <typename _Tensor>                                                       \
    inline auto fun (_Tensor ts)                                                       \
        ->decltype(map(ts, fun##_functor<typename _Tensor::value_type>())) {          \
        return map(ts, fun##_functor<typename _Tensor::value_type>());                \
    }
// clang-format on

// basice
MTENSOR_UNARY_VIEW(abs)

// exponential
MTENSOR_UNARY_VIEW(exp)
MTENSOR_UNARY_VIEW(log)

// power
MTENSOR_UNARY_VIEW(pow)
MTENSOR_UNARY_VIEW(sqrt)

// nearest
MTENSOR_UNARY_VIEW(round)
MTENSOR_UNARY_VIEW(floor)
MTENSOR_UNARY_VIEW(ceil)

// trigonometric
MTENSOR_UNARY_VIEW(sin)
MTENSOR_UNARY_VIEW(cos)
MTENSOR_UNARY_VIEW(tan)
MTENSOR_UNARY_VIEW(asin)
MTENSOR_UNARY_VIEW(acos)
MTENSOR_UNARY_VIEW(atan)

}  // namespace view
}  // namespace matazure