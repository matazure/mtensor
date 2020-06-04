#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>
#include <matazure/type_traits.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace view {

template <typename... _Tensors>
struct zip_functor {
    typedef typename tuple_element<0, tuple<_Tensors...>>::type first_tensor_type;
    const static int_t rank = first_tensor_type::rank;

    zip_functor(tuple<_Tensors...> tensors) : tensors_(tensors) {}

    MATAZURE_GENERAL tuple<reference_t<_Tensors>...> operator()(pointi<rank> idx) const {
        return access_imp(idx, make_integer_sequence<int_t, sizeof...(_Tensors)>{});
    }

   private:
    template <int_t... _Indices>
    MATAZURE_GENERAL tuple<reference_t<_Tensors>...> access_imp(
        pointi<rank> idx, integer_sequence<int_t, _Indices...>) const {
        return tuple<reference_t<_Tensors>...>{get<_Indices>(tensors_)(idx)...};
    }

   private:
    tuple<_Tensors...> tensors_;
};

template <typename... _Tensors>
inline auto zip_imp(tuple<_Tensors...> tensors)
    -> decltype(make_lambda(get<0>(tensors).shape(), zip_functor<_Tensors...>(tensors),
                            typename tuple_element<0, tuple<_Tensors...>>::type::runtime_type{},
                            typename tuple_element<0, tuple<_Tensors...>>::type::layout_type{})) {
    return make_lambda(get<0>(tensors).shape(), zip_functor<_Tensors...>(tensors),
                       typename tuple_element<0, tuple<_Tensors...>>::type::runtime_type{},
                       typename tuple_element<0, tuple<_Tensors...>>::type::layout_type{});
}

template <typename... _Tensors>
inline auto zip(_Tensors... tensors) -> decltype(zip_imp(tuple<_Tensors...>(tensors...))) {
    return zip_imp(tuple<_Tensors...>(tensors...));
}

}  // namespace view
}  // namespace matazure
