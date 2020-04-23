#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace view {

template <typename _Tensor, int_t _SliceDimIdx>
struct slice_functor {
   private:
    _Tensor ts_;
    int_t slice_i_;

   public:
    slice_functor(_Tensor ts, int_t slice_i) : ts_(ts), slice_i_(slice_i) {}

    MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank - 1> idx) const
        -> decltype((ts_(cat_point<_SliceDimIdx>(idx, slice_i_)))) {
        return ts_(cat_point<_SliceDimIdx>(idx, slice_i_));
    }
};

/**
 * @brief produces a sub-dim lambda_tensor of the source tensor
 * @param ts the source tensor
 * @tparam _DimIdx the sliced dim(orientation) index
 * @param i the slice position index on the sliced dim(orientation)
 * @return a sub-dim lambda_tensor
 */
template <int_t _DimIdx, typename _Tensor>
inline auto slice(_Tensor ts, int_t positon_index)
    -> decltype(make_lambda(slice_point<_DimIdx>(ts.shape()),
                            slice_functor<_Tensor, _DimIdx>(ts, positon_index),
                            typename _Tensor::memory_type{})) {
    return make_lambda(slice_point<_DimIdx>(ts.shape()),
                       slice_functor<_Tensor, _DimIdx>(ts, positon_index),
                       typename _Tensor::memory_type{});
}

}  // namespace view
}  // namespace matazure
