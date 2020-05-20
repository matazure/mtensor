#pragma once
#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace view {

template <typename _Tensor, typename _Kernel>
struct conv_functor {
   private:
    typedef typename _Tensor::value_type value_type;
    static const int_t rank = _Tensor::rank;

    const _Tensor ts_;
    const _Kernel kernel_;
    const pointi<rank> kernel_shape_;
    const pointi<rank> kernel_radius_;

   public:
    conv_functor(_Tensor ts, _Kernel kernel)
        : ts_(ts),
          kernel_(kernel),
          kernel_shape_(kernel.shape()),
          kernel_radius_(kernel.shape() / 2) {}

    MATAZURE_GENERAL value_type operator()(pointi<_Tensor::rank> idx) const {
        auto re = matazure::zero<value_type>::value();
        for_index(kernel_shape_, [&](pointi<rank> neigbor_idx) {
            re += kernel_(neigbor_idx) * ts_(idx + neigbor_idx - kernel_radius_);
        });

        return re;
    }
};

template <typename _Tensor>
struct conv_neighbors_weights_functor {
   private:
    typedef typename _Tensor::value_type value_type;
    static const int_t rank = _Tensor::rank;
    typedef tensor<tuple<pointi<_Tensor::rank>, typename _Tensor::value_type>, 1> weights_type;

    _Tensor ts_;
    weights_type neighbors_weights_;

   public:
    conv_neighbors_weights_functor(_Tensor ts, weights_type neighbors_weights)
        : ts_(ts), neighbors_weights_(neighbors_weights) {}

    MATAZURE_GENERAL value_type operator()(pointi<_Tensor::rank> idx) const {
        auto re = matazure::zero<value_type>::value();
        for (int_t i = 0; i < neighbors_weights_.size(); ++i) {
            auto nw = neighbors_weights_[i];
            auto local_idx = get<0>(nw);
            auto weights = get<1>(nw);
            re += ts_(idx + local_idx) * weights;
        }

        return re;
    }
};

template <typename _Tensor, typename _Kernel>
inline auto conv(_Tensor ts, _Kernel kernel)
    -> decltype(make_lambda(ts.shape(), conv_functor<_Tensor, _Kernel>(ts, kernel),
                            typename _Tensor::runtime_type{}, typename _Tensor::layout_type{})) {
    return make_lambda(ts.shape(), conv_functor<_Tensor, _Kernel>(ts, kernel),
                       typename _Tensor::runtime_type{}, typename _Tensor::layout_type{});
}

template <typename _Tensor>
[[deprecated]] inline auto conv(
    _Tensor ts,
    tensor<tuple<pointi<_Tensor::rank>, typename _Tensor::value_type>, 1> neighbors_weights)
    -> decltype(make_lambda(ts.shape(),
                            conv_neighbors_weights_functor<_Tensor>(ts, neighbors_weights),
                            typename _Tensor::runtime_type{}, typename _Tensor::layout_type{})) {
    return make_lambda(ts.shape(), conv_neighbors_weights_functor<_Tensor>(ts, neighbors_weights),
                       typename _Tensor::runtime_type{}, typename _Tensor::layout_type{});
}

}  // namespace view
}  // namespace matazure
