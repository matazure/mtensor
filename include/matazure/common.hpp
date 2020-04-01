#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/lambda_tensor.hpp>
#include <matazure/cuda/tensor.hpp>
#endif

#include <matazure/geometry.hpp>

namespace matazure {

#ifndef MATAZURE_CUDA

/**
 * @brief make a lambda_tensor
 * @param the shape
 * @param the functor, a index -> value pattern
 */
template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> extent, _Func fun) -> lambda_tensor<_Rank, _Func> {
    return lambda_tensor<_Rank, _Func>(extent, fun);
}

/**
 * @brief make a lambda_tensor
 * @param the shape
 * @param the functor, a index -> value pattern
 */
template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> extent, _Func fun, host_tag) -> lambda_tensor<_Rank, _Func> {
    return lambda_tensor<_Rank, _Func>(extent, fun);
}

/**
 * @brief make a lambda_tensor
 * @param the shape
 * @param the functor, a index -> value pattern
 */
template <int_t _Rank, typename _Func>
inline auto make_host_lambda(pointi<_Rank> extent, _Func fun) -> lambda_tensor<_Rank, _Func> {
    return lambda_tensor<_Rank, _Func>(extent, fun);
}

#else

/**
 * @brief make a lambda_tensor
 * @param the shape
 * @param the functor, a index -> value pattern
 */
template <int_t _Rank, typename _Func>
inline auto make_host_lambda(pointi<_Rank> extent, _Func fun) -> lambda_tensor<_Rank, _Func> {
    return lambda_tensor<_Rank, _Func>(extent, fun);
}

/**
 * @brief make a lambda_tensor
 * @param the shape
 * @param the functor, should be a host functor a index -> value pattern
 */
template <int_t _Rank, typename _Func>
inline auto make_lambda(
    pointi<_Rank> ext, _Func fun,
    enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func) && !MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)
    -> decltype(make_host_lambda(ext, fun)) {
    return make_host_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(
    pointi<_Rank> ext, _Func fun, host_tag,
    enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func) && !MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)
    -> decltype(make_host_lambda(ext, fun)) {
    return make_host_lambda(ext, fun);
}

/**
 * @todo: not support device struct operator, it's diffcult to do this.
 */
template <int_t _Rank, typename _Func>
inline auto make_lambda(
    pointi<_Rank> ext, _Func fun, device_tag,
    enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func) && !MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)
    -> decltype(cuda::make_lambda(ext, fun)) {
    return cuda::make_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun,
                        enable_if_t<MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)
    -> decltype(cuda::make_lambda(ext, fun)) {
    return cuda::make_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, device_tag,
                        enable_if_t<MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)
    -> decltype(cuda::make_lambda(ext, fun)) {
    return cuda::make_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, host_tag,
                        enable_if_t<MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)
    -> decltype(make_host_lambda(ext, fun)) {
    return make_host_lambda(ext, fun);
}

#endif

template <typename _Tensor, typename _Func>
struct linear_map_op {
   private:
    const _Tensor ts_;
    const _Func functor_;

   public:
    linear_map_op(_Tensor ts, _Func fun) : ts_(ts), functor_(fun) {}

    MATAZURE_GENERAL auto operator()(int_t i) const -> decltype(this->functor_(this->ts_[i])) {
        return functor_(ts_[i]);
    }
};

template <typename _Tensor, typename _Func>
struct array_map_op {
   private:
    const _Tensor ts_;
    const _Func functor_;

   public:
    array_map_op(_Tensor ts, _Func fun) : ts_(ts), functor_(fun) {}

    MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const
        -> decltype(this->functor_(this->ts_[idx])) {
        return functor_(ts_[idx]);
    }
};

template <typename _Tensor, typename _Func>
struct device_linear_map_op {
   private:
    const _Tensor ts_;
    const _Func functor_;

   public:
    device_linear_map_op(_Tensor ts, _Func fun) : ts_(ts), functor_(fun) {}

    MATAZURE_DEVICE auto operator()(int_t i) const -> decltype(this->functor_(this->ts_[i])) {
        return functor_(ts_[i]);
    }
};

template <typename _Tensor, typename _Func>
struct device_array_map_op {
   private:
    const _Tensor ts_;
    const _Func functor_;

   public:
    device_array_map_op(_Tensor ts, _Func fun) : ts_(ts), functor_(fun) {}

    MATAZURE_DEVICE auto operator()(pointi<_Tensor::rank> idx) const
        -> decltype(this->functor_(this->ts_[idx])) {
        return functor_(ts_[idx]);
    }
};

template <typename _OutValueType>
struct cast_functor {
    template <typename _InValueType>
    MATAZURE_GENERAL _OutValueType operator()(_InValueType v) const {
        return static_cast<_OutValueType>(v);
    }
};

template <typename _OutPointValueType, int_t _Rank>
struct cast_functor<point<_OutPointValueType, _Rank>> {
    template <typename _InPointValueType>
    MATAZURE_GENERAL point<_OutPointValueType, _Rank> operator()(
        const point<_InPointValueType, _Rank>& p) const {
        return point_cast<_OutPointValueType>(p);
    }
};

// pointi<2>
template <int_t _SliceDimIdx>
inline pointi<1> slice_point(pointi<2> pt);

template <>
inline pointi<1> slice_point<0>(pointi<2> pt) {
    return pointi<1>{get<1>(pt)};
}

template <>
inline pointi<1> slice_point<1>(pointi<2> pt) {
    return pointi<1>{get<0>(pt)};
}

template <int_t _CatDimIdx>
inline pointi<2> cat_point(pointi<1> pt, int_t cat_i);

template <>
inline pointi<2> cat_point<0>(pointi<1> pt, int_t cat_i) {
    return pointi<2>{cat_i, get<0>(pt)};
}

template <>
inline pointi<2> cat_point<1>(pointi<1> pt, int_t cat_i) {
    return pointi<2>{get<0>(pt), cat_i};
}

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

#ifndef MATAZURE_CUDA

/**
 * @brief apply the functor for each element of a linear indexing tensor
 * @param ts the source tensor
 * @param fun the functor, element -> value  pattern
 */
template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun,
                  enable_if_t<is_same<linear_index, typename _Tensor::index_type>::value>* = 0)
    -> decltype(make_lambda(ts.shape(), linear_map_op<_Tensor, _Func>(ts, fun),
                            typename _Tensor::memory_type{})) {
    return make_lambda(ts.shape(), linear_map_op<_Tensor, _Func>(ts, fun),
                       typename _Tensor::memory_type{});
}

/**
 * @brief apply the functor for each element of a array indexing tensor
 * @param ts the source tensor
 * @param fun the functor, element -> value  pattern
 */
template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun,
                  enable_if_t<is_same<array_index, typename _Tensor::index_type>::value>* = 0)
    -> decltype(make_lambda(ts.shape(), array_map_op<_Tensor, _Func>(ts, fun),
                            typename _Tensor::memory_type{})) {
    return make_lambda(ts.shape(), array_map_op<_Tensor, _Func>(ts, fun),
                       typename _Tensor::memory_type{});
}

#else

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun,
                  enable_if_t<is_same<linear_index, typename _Tensor::index_type>::value>* = 0,
                  enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)
    -> decltype(make_lambda(ts.shape(), linear_map_op<_Tensor, _Func>(ts, fun),
                            typename _Tensor::memory_type{})) {
    return make_lambda(ts.shape(), linear_map_op<_Tensor, _Func>(ts, fun),
                       typename _Tensor::memory_type{});
}

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun,
                  enable_if_t<is_same<array_index, typename _Tensor::index_type>::value>* = 0,
                  enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)
    -> decltype(make_lambda(ts.shape(), array_map_op<_Tensor, _Func>(ts, fun),
                            typename _Tensor::memory_type{})) {
    return make_lambda(ts.shape(), array_map_op<_Tensor, _Func>(ts, fun),
                       typename _Tensor::memory_type{});
}

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun,
                  enable_if_t<is_same<linear_index, typename _Tensor::index_type>::value>* = 0,
                  enable_if_t<MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)
    -> decltype(make_lambda(ts.shape(), device_linear_map_op<_Tensor, _Func>(ts, fun),
                            typename _Tensor::memory_type{})) {
    return make_lambda(ts.shape(), device_linear_map_op<_Tensor, _Func>(ts, fun),
                       typename _Tensor::memory_type{});
}

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun,
                  enable_if_t<is_same<array_index, typename _Tensor::index_type>::value>* = 0,
                  enable_if_t<MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)
    -> decltype(make_lambda(ts.shape(), device_array_map_op<_Tensor, _Func>(ts, fun),
                            typename _Tensor::memory_type{})) {
    return make_lambda(ts.shape(), device_array_map_op<_Tensor, _Func>(ts, fun),
                       typename _Tensor::memory_type{});
}

#endif

/**
 * @brief cast a tensor to another value_type lambda_tensor
 *
 * support primitive type static_cast and point_cast.
 *
 * @param tensor the source tensor
 * @tparam _ValueType the dest tensor value type
 * @return a lambda_tensor whose value_type is _ValueType
 */
template <typename _ValueType, typename _Tensor>
inline auto cast(_Tensor tensor, enable_if_t<is_tensor<_Tensor>::value>* = 0)
    -> decltype(apply(tensor, cast_functor<_ValueType>())) {
    return apply(tensor, cast_functor<_ValueType>());
}

///**
//* @brief saturate cast a tensor to another value_type lambda_tensor
//* @param tensor the source tensor
//* @tparam _ValueType the dest tensor value type
//* @return a lambda_tensor whose value_type is _ValueType
//*/
// template <typename _ValueType, typename _Tensor>
// inline auto saturate_cast(_Tensor tensor, enable_if_t<is_tensor<_Tensor>>* =
// 0)->decltype(apply(tensor,  cast_functor<_ValueType>())) {
//	// typedef point<byte, 3> (* sature_cast_op)(const point<float, 3> &);
//	// sature_cast_op pointf3_to_pointb3 = &unary::saturate_cast<byte, float, 3>;
//	return apply(tensor,  cast_functor<_ValueType>());
//}

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
}
