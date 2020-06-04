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
struct slice_functor {
   private:
    _Tensor ts_;
    pointi<_Tensor::rank> offset_;

   public:
    slice_functor(_Tensor ts, pointi<_Tensor::rank> offset) : ts_(ts), offset_(offset) {}

    MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const
        -> decltype((ts_(idx + offset_))) {
        return ts_(idx + offset_);
    }
};

/**
 * @brief produces a subsection lambda_tensor of the source tensor
 * @param ts the source tensor
 * @param origin the origin of the slice
 * @param shape the shape of the slice
 * @return a subsection lambda_tensor
 */
template <typename _Tensor>
inline auto slice(_Tensor ts, pointi<_Tensor::rank> origin, pointi<_Tensor::rank> shape)
    -> decltype(make_lambda(shape, slice_functor<decay_t<_Tensor>>(ts, origin),
                            runtime_t<_Tensor>{}, layout_t<_Tensor>{})) {
    return make_lambda(shape, slice_functor<decay_t<_Tensor>>(ts, origin), runtime_t<_Tensor>{},
                       layout_t<_Tensor>{});
}

}  // namespace view
}  // namespace matazure
