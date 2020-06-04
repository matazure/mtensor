#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

namespace matazure {
namespace view {

template <typename _Tensor, int_t _Axis>
struct gather_scalar_functor {
   private:
    _Tensor ts_;
    int_t axis_i_;

   public:
    gather_scalar_functor(_Tensor ts, int_t axis_i) : ts_(ts), axis_i_(axis_i) {}

    MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank - 1> idx) const
        -> decltype((ts_(scatter_point<_Axis>(idx, axis_i_)))) {
        return ts_(scatter_point<_Axis>(idx, axis_i_));
    }
};

template <typename _Tensor, typename _Vector, int_t _Axis>
struct gather_vector_functor {
   private:
    _Tensor ts_;
    _Vector indices_;

   public:
    gather_vector_functor(_Tensor ts, _Vector indices) : ts_(ts), indices_(indices) {}

    MATAZURE_GENERAL reference_t<_Tensor> operator()(pointi<_Tensor::rank> idx) const {
        idx[_Axis] = indices_[idx[_Axis]];
        return ts_(idx);
    }
};

namespace internal {
template <int_t _Axis, int_t _Rank>
inline pointi<_Rank> get_gather_vector_shape(const pointi<_Rank>& shape, int_t size) {
    auto re = shape;
    re[_Axis] = size;
    return re;
}
}  // namespace internal

template <int_t _Axis, typename _Tensor, typename _Vector>
inline auto gather(_Tensor ts, _Vector indices)
    -> decltype(make_lambda(internal::get_gather_vector_shape<_Axis>(ts.shape(), indices.size()),
                            gather_vector_functor<_Tensor, _Vector, _Axis>(ts, indices),
                            runtime_t<_Tensor>{}, layout_t<_Tensor>{})) {
    static_assert(_Axis >= 0 && _Axis < _Tensor::rank, "_Axis must be >=0 or < _Tensor::rank");
    return make_lambda(internal::get_gather_vector_shape<_Axis>(ts.shape(), indices.size()),
                       gather_vector_functor<_Tensor, _Vector, _Axis>(ts, indices),
                       runtime_t<_Tensor>{}, layout_t<_Tensor>{});
}

template <int_t _Axis, typename _Tensor>
inline auto gather(_Tensor ts, int_t positon_index)
    -> decltype(make_lambda(gather_point<_Axis>(ts.shape()),
                            gather_scalar_functor<_Tensor, _Axis>(ts, positon_index),
                            runtime_t<_Tensor>{},
                            typename layout_getter<layout_t<_Tensor>, _Tensor::rank - 1>::type{})) {
    static_assert(_Axis >= 0 && _Axis < _Tensor::rank, "_Axis must be >=0 or < _Tensor::rank");
    return make_lambda(
        gather_point<_Axis>(ts.shape()), gather_scalar_functor<_Tensor, _Axis>(ts, positon_index),
        runtime_t<_Tensor>{}, typename layout_getter<layout_t<_Tensor>, _Tensor::rank - 1>::type{});
}

}  // namespace view
}  // namespace matazure
