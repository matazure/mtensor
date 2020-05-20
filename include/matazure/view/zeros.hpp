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
struct zeros_functor {
    MATAZURE_GENERAL _ValueType operator()(pointi<_Rank> idx) const { return _ValueType{0}; }
};

template <typename _ValueType, int_t _Rank, typename _RuntimeType>
inline auto zeros(pointi<_Rank> shape, _RuntimeType)
    -> decltype(make_lambda(shape, zeros_functor<_ValueType, _Rank>{}, _RuntimeType{})) {
    return make_lambda(shape, zeros_functor<_ValueType, _Rank>{}, _RuntimeType{});
}

}  // namespace view
}  // namespace matazure
