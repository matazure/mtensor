#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace view {

template <typename... _Axes>
struct meshgrid_functor {
    typedef typename tuple_element<0, tuple<_Axes...>>::type first_tensor_type;
    typedef typename first_tensor_type::value_type value_type;
    const static int_t rank = sizeof...(_Axes);

    meshgrid_functor(tuple<_Axes...> axes) : axes_(axes) {}

    MATAZURE_GENERAL point<value_type, rank> operator()(pointi<rank> idx) const {
        return access_imp(idx, make_integer_sequence<int_t, sizeof...(_Axes)>{});
    }

   private:
    template <int_t... _Indices>
    MATAZURE_GENERAL point<value_type, rank> access_imp(
        pointi<rank> idx, integer_sequence<int_t, _Indices...>) const {
        return point<value_type, rank>{get<_Indices>(axes_)[idx[_Indices]]...};
    }

   private:
    tuple<_Axes...> axes_;
};

namespace internal {

template <typename... _Axes, int_t... _Indices>
inline auto get_meshgrid_shape_imp(tuple<_Axes...> axes, integer_sequence<int_t, _Indices...>)
    -> pointi<sizeof...(_Axes)> {
    return pointi<sizeof...(_Axes)>{get<_Indices>(axes).size()...};
}

template <typename... _Axes>
inline auto get_meshgrid_shape(tuple<_Axes...> axes) -> pointi<sizeof...(_Axes)> {
    return get_meshgrid_shape_imp(axes, make_integer_sequence<int_t, sizeof...(_Axes)>{});
}

}  // namespace internal

template <typename... _Axes>
inline auto meshgrid_imp(tuple<_Axes...> axes) -> decltype(make_lambda(
    internal::get_meshgrid_shape(axes), meshgrid_functor<_Axes...>(axes),
    typename tuple_element<0, tuple<_Axes...>>::type::runtime_type{},
    typename layout_getter<typename tuple_element<0, tuple<_Axes...>>::type::layout_type,
                           sizeof...(_Axes)>::type{})) {
    return make_lambda(
        internal::get_meshgrid_shape(axes), meshgrid_functor<_Axes...>(axes),
        typename tuple_element<0, tuple<_Axes...>>::type::runtime_type{},
        typename layout_getter<typename tuple_element<0, tuple<_Axes...>>::type::layout_type,
                               sizeof...(_Axes)>::type{});
}

template <typename... _Axes>
inline auto meshgrid(_Axes... axes) -> decltype(meshgrid_imp(tuple<_Axes...>(axes...))) {
    return meshgrid_imp(tuple<_Axes...>(axes...));
}

}  // namespace view
}  // namespace matazure
