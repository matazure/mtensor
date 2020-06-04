#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace view {

template <typename _Tensor>
struct clamp_zero_functor {
   private:
    _Tensor ts_;

   public:
    clamp_zero_functor(_Tensor ts) : ts_(ts) {}

    MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const
        -> decltype(zero<decay_t<typename _Tensor::value_type>>::value()) {
        if (MATAZURE_LIKELY(inside_rect(idx, zero<pointi<_Tensor::rank>>::value(), ts_.shape()))) {
            return ts_(idx);
        } else {
            return zero<typename _Tensor::value_type>::value();
        }
    }
};

/**
 * @brief procudes a clamped indexing lambda_tensor from the source tensor.
 * @param ts the source tensor
 * @return a clamped indexing lambda_tensor
 */
template <typename _Tensor>
inline auto clamp_zero(_Tensor ts)
    -> decltype(make_lambda(ts.shape(), clamp_zero_functor<decay_t<_Tensor>>(ts),
                            runtime_t<_Tensor>{}, layout_t<_Tensor>{})) {
    return make_lambda(ts.shape(), clamp_zero_functor<decay_t<_Tensor>>(ts), runtime_t<_Tensor>{},
                       layout_t<_Tensor>{});
}

}  // namespace view
}  // namespace matazure
