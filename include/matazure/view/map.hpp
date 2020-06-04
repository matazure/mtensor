
#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace view {

template <typename _Tensor, typename _Fun>
struct linear_map_functor {
   private:
    const _Tensor ts_;
    const _Fun functor_;

   public:
    linear_map_functor(_Tensor ts, _Fun fun) : ts_(ts), functor_(fun) {}

    MATAZURE_GENERAL auto operator()(int_t i) const -> decltype((functor_(ts_[i]))) {
        return functor_(ts_[i]);
    }
};

template <typename _Tensor, typename _Fun>
struct array_map_functor {
   private:
    const _Tensor ts_;
    const _Fun functor_;

   public:
    array_map_functor(_Tensor ts, _Fun fun) : ts_(ts), functor_(fun) {}

    MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const
        -> decltype((functor_(ts_(idx)))) {
        return functor_(ts_(idx));
    }
};

/**
 * @brief map the functor for each element of a linear indexing tensor
 * @param ts the source tensor
 * @param fun the functor, element -> value  pattern
 */
template <typename _Tensor, typename _Fun>
inline auto map(_Tensor ts, _Fun fun,
                enable_if_t<is_same<linear_index, typename _Tensor::index_type>::value>* = 0)
    -> decltype(make_lambda(ts.shape(), linear_map_functor<_Tensor, _Fun>(ts, fun),
                            typename _Tensor::runtime_type{}, typename _Tensor::layout_type{})) {
    // function_trais has bug
    // typedef function_traits<_Fun> trais_t;
    // static_assert(trais_t::arguments_size == 1, "_Fun arguments size must be 1");
    // static_assert(
    //     is_convertible<typename _Tensor::reference,
    //                    typename function_traits<_Fun>::template arguments<0>::type>::value,
    //     "_Fun arguments size must be 1");

    return make_lambda(ts.shape(), linear_map_functor<_Tensor, _Fun>(ts, fun),
                       typename _Tensor::runtime_type{}, typename _Tensor::layout_type{});
}

/**
 * @brief map the functor for each element of a array indexing tensor
 * @param ts the source tensor
 * @param fun the functor, element -> value  pattern
 */
template <typename _Tensor, typename _Fun>
inline auto map(_Tensor ts, _Fun fun,
                enable_if_t<is_same<array_index, typename _Tensor::index_type>::value>* = 0)
    -> decltype(make_lambda(ts.shape(), array_map_functor<_Tensor, _Fun>(ts, fun),
                            typename _Tensor::runtime_type{}, typename _Tensor::layout_type{})) {
    return make_lambda(ts.shape(), array_map_functor<_Tensor, _Fun>(ts, fun),
                       typename _Tensor::runtime_type{}, typename _Tensor::layout_type{});
}

}  // namespace view
}  // namespace matazure
