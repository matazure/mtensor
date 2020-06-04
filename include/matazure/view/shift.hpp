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
struct shift_functor {
   private:
    _Tensor ts_;
    pointi<_Tensor::rank> offset_;

   public:
    shift_functor(_Tensor ts, pointi<_Tensor::rank> offset) : ts_(ts), offset_(offset) {}

    MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const
        -> decltype((ts_(idx + offset_))) {
        return ts_(idx + offset_);
    }
};

template <typename _Tensor>
inline auto shift(_Tensor ts, pointi<_Tensor::rank> offset)
    -> decltype(make_lambda(ts.shape(), shift_functor<decay_t<_Tensor>>(ts, offset),
                            runtime_t<_Tensor>{}, layout_t<_Tensor>{})) {
    return make_lambda(ts.shape(), shift_functor<decay_t<_Tensor>>(ts, offset),
                       runtime_t<_Tensor>{}, layout_t<_Tensor>{});
}

}  // namespace view
}  // namespace matazure
