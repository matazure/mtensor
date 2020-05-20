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
struct ones_functor {
    MATAZURE_GENERAL _ValueType operator()(pointi<_Rank> idx) const { return _ValueType{1}; }
};

template <typename _ValueType, int_t _Rank, typename _RuntimeType>
inline auto ones(pointi<_Rank> shape, _RuntimeType)
    -> decltype(make_lambda(shape, ones_functor<_ValueType, _Rank>{}, _RuntimeType{})) {
    return make_lambda(shape, ones_functor<_ValueType, _Rank>{}, _RuntimeType{});
}

}  // namespace view
}  // namespace matazure
