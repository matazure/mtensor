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
struct crop_functor {
   private:
    _Tensor ts_;
    pointi<_Tensor::rank> offset_;

   public:
    crop_functor(_Tensor ts, pointi<_Tensor::rank> offset) : ts_(ts), offset_(offset) {}

    MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const
        -> decltype((ts_[idx + offset_])) {
        return ts_[idx + offset_];
    }
};

/**
 * @brief produces a subsection lambda_tensor of the source tensor
 * @param ts the source tensor
 * @param origin the origin of the crop
 * @param ext the extent of the crop
 * @return a subsection lambda_tensor
 */
template <typename _Tensor>
inline auto crop(_Tensor ts, pointi<_Tensor::rank> origin, pointi<_Tensor::rank> ext)
    -> decltype(make_lambda(ext, crop_functor<decay_t<_Tensor>>(ts, origin),
                            typename _Tensor::memory_type{})) {
    return make_lambda(ext, crop_functor<decay_t<_Tensor>>(ts, origin),
                       typename _Tensor::memory_type{});
}

}  // namespace view
}  // namespace matazure
