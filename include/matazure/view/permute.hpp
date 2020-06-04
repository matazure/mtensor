
#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace view {

template <typename _Tensor, int_t... _Idx>
struct permute_functor {
   private:
    const _Tensor ts_;

   public:
    permute_functor(_Tensor ts) : ts_(ts) {}

    MATAZURE_GENERAL typename _Tensor::value_type operator()(pointi<_Tensor::rank> idx) const {
        return ts_(permute_point<_Idx...>(idx));
    }
};

template <int_t... _Idx, typename _Tensor>
inline auto permute(_Tensor ts)
    -> decltype(make_lambda(ts.shape(), permute_functor<_Tensor, _Idx...>(ts), runtime_t<_Tensor>{},
                            layout_t<_Tensor>{})) {
    return make_lambda(ts.shape(), permute_functor<_Tensor, _Idx...>(ts), runtime_t<_Tensor>{},
                       layout_t<_Tensor>{});
}

}  // namespace view
}  // namespace matazure
