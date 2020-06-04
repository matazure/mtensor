#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace view {

template <typename _Tensor, typename _StrideType>
struct stride_functor {
   private:
    _Tensor ts_;
    _StrideType stride_;

   public:
    stride_functor(_Tensor ts, _StrideType stride) : ts_(ts), stride_(stride) {}

    MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const
        -> decltype((ts_(idx * stride_))) {
        return ts_(idx * stride_);
    }
};

/**
 * @brief produces a stride indexing lambda_tensor of the source tensor
 * @param ts the source tensor
 * @param stride the stride of the indexing
 * @return a stride indexing lambda_tensor
 */
template <typename _Tensor, typename _StrideType>
inline auto stride(_Tensor ts, _StrideType stride)
    -> decltype(make_lambda(ts.shape() / stride, stride_functor<_Tensor, _StrideType>(ts, stride),
                            runtime_t<_Tensor>{}, layout_t<_Tensor>{})) {
    return make_lambda(ts.shape() / stride, stride_functor<_Tensor, _StrideType>(ts, stride),
                       runtime_t<_Tensor>{}, layout_t<_Tensor>{});
}

}  // namespace view
}  // namespace matazure
