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
struct eye_functor {
    MATAZURE_GENERAL _ValueType operator()(pointi<_Rank> idx) const {
        return access_imp(idx, make_integer_sequence<int_t, _Rank>{});
    }

    template <int_t... _Indices>
    MATAZURE_GENERAL _ValueType access_imp(pointi<_Rank> idx,
                                           integer_sequence<int_t, _Indices...>) const {
        if (all((idx[0] == idx[_Indices])...)) {
            return _ValueType(1);
        } else {
            return _ValueType(0);
        }
    }
};

template <typename _ValueType, int_t _Rank, typename _RuntimeType = host_t,
          typename _Layout = default_layout<global_t, 2>::type>
inline auto eye(pointi<_Rank> shape, _RuntimeType runtime = host_t{},
                _Layout layout = row_major_layout<_Rank>{})
    -> decltype(make_lambda(shape, eye_functor<_ValueType, _Rank>{}, _RuntimeType{}, _Layout{})) {
    return make_lambda(shape, eye_functor<_ValueType, _Rank>{}, _RuntimeType{}, _Layout{});
}

}  // namespace view
}  // namespace matazure
