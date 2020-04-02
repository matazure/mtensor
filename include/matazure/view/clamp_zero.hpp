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
struct clamp_zero_op {
   private:
    _Tensor ts_;

   public:
    clamp_zero_op(_Tensor ts) : ts_(ts) {}

    MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const
        -> decltype(zero<decay_t<typename _Tensor::value_type>>::value()) {
        if (MATAZURE_LIKELY(inside_rect(idx, pointi<_Tensor::rank>::zeros(), ts_.shape()))) {
            return ts_[idx];
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
    -> decltype(make_lambda(ts.shape(), clamp_zero_op<decay_t<_Tensor>>(ts),
                            typename _Tensor::memory_type{})) {
    return make_lambda(ts.shape(), clamp_zero_op<decay_t<_Tensor>>(ts),
                       typename _Tensor::memory_type{});
}

}  // namespace view
}  // namespace matazure
