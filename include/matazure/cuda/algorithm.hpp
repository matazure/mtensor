#pragma once

#include <matazure/cuda/block_for_index.hpp>
#include <matazure/cuda/for_index.hpp>

namespace matazure {
namespace cuda {

template <typename _Tensor, typename _Fun>
struct for_each_linear_functor {
    for_each_linear_functor(_Tensor ts, _Fun fun) : ts_(ts), fun_(fun) {}

    void MATAZURE_DEVICE operator()(int_t i) const { fun_(ts_[i]); }

   private:
    _Tensor ts_;
    _Fun fun_;
};

template <typename _ExecutionPolicy, typename _Tensor, typename _Fun>
inline void for_each(
    _ExecutionPolicy policy, _Tensor ts, _Fun fun,
    enable_if_t<are_device_memory<_Tensor>::value && are_linear_index<_Tensor>::value>* = 0) {
    cuda::for_index(policy, 0, ts.size(), for_each_linear_functor<_Tensor, _Fun>(ts, fun));
}

template <typename _Tensor, typename _Fun>
struct for_each_array_functor {
    for_each_array_functor(_Tensor ts, _Fun fun) : ts_(ts), fun_(fun) {}

    void MATAZURE_DEVICE operator()(pointi<_Tensor::rank> idx) const { fun_(ts_(idx)); }

   private:
    _Tensor ts_;
    _Fun fun_;
};

template <typename _ExecutionPolicy, typename _Tensor, typename _Fun>
inline void for_each(_ExecutionPolicy policy, _Tensor ts, _Fun fun,
                     enable_if_t<are_device_memory<decay_t<_Tensor>>::value &&
                                 !are_linear_index<decay_t<_Tensor>>::value>* = 0) {
    cuda::for_index(policy, zero<pointi<_Tensor::rank>>::value(), ts.shape(),
                    for_each_array_functor<_Tensor, _Fun>(ts, fun));
}

template <typename _Tensor, typename _Fun>
inline void for_each(
    _Tensor ts, _Fun fun,
    enable_if_t<
        are_device_memory<enable_if_t<is_tensor<decay_t<_Tensor>>::value, _Tensor>>::value>* = 0) {
    for_index_execution_policy policy;
    policy.total_size(ts.size());
    for_each(policy, ts, fun);
}

template <typename _Tensor>
struct fill_functor {
    typedef typename _Tensor::value_type value_type;

    MATAZURE_DEVICE void operator()(reference_t<_Tensor> e) const { e = v_; }

    value_type v_;
};

template <typename _ExecutionPolicy, typename _Tensor>
inline void fill(
    _ExecutionPolicy policy, _Tensor ts, typename _Tensor::value_type v,
    enable_if_t<
        are_device_memory<enable_if_t<is_tensor<decay_t<_Tensor>>::value, _Tensor>>::value>* = 0) {
    for_each(policy, ts, fill_functor<_Tensor>{v});
}

template <typename _Tensor>
inline void fill(
    _Tensor ts, typename _Tensor::value_type v,
    enable_if_t<
        are_device_memory<enable_if_t<is_tensor<decay_t<_Tensor>>::value, _Tensor>>::value>* = 0) {
    for_index_execution_policy policy;
    policy.total_size(ts.size());
    fill(policy, ts, v);
}

template <typename _T1, typename _T2>
struct copy_linear_functor {
    _T1 ts1;
    _T2 ts2;

    MATAZURE_DEVICE void operator()(int_t i) const { ts2[i] = ts1[i]; }
};

template <typename _ExecutionPolicy, typename _T1, typename _T2>
void copy(
    _ExecutionPolicy policy, _T1 ts1, _T2 ts2,
    enable_if_t<are_linear_index<_T1, _T2>::value && are_device_memory<_T1, _T2>::value>* = 0) {
    cuda::for_index(policy, 0, ts1.size(), copy_linear_functor<_T1, _T2>{ts1, ts2});
}

template <typename _T1, typename _T2>
struct copy_array_functor {
    _T1 ts1;
    _T2 ts2;

    MATAZURE_DEVICE void operator()(pointi<_T1::rank> idx) const { ts2(idx) = ts1(idx); }
};

template <typename _ExecutionPolicy, typename _T1, typename _T2>
void copy(
    _ExecutionPolicy policy, _T1 ts1, _T2 ts2,
    enable_if_t<!are_linear_index<_T1, _T2>::value && are_device_memory<_T1, _T2>::value>* = 0) {
    cuda::for_index(policy, zero<pointi<_T1::rank>>::value(), ts1.shape(),
                    copy_array_functor<_T1, _T2>{ts1, ts2});
}

template <typename _T1, typename _T2>
void copy(
    _T1 ts1, _T2 ts2,
    enable_if_t<are_device_memory<enable_if_t<is_tensor<_T1>::value, _T1>, _T2>::value>* = 0) {
    for_index_execution_policy policy;
    policy.total_size(ts1.size());
    copy(policy, ts1, ts2);
}

template <typename _T1, typename _T2, typename _Fun>
struct transform_linear_functor {
    _T1 ts1;
    _T2 ts2;
    _Fun fun;

    MATAZURE_DEVICE void operator()(int_t i) const { ts2[i] = fun(ts1[i]); }
};

/**
 * @brief transform a linear indexing cuda tensor to another by the fun
 * @param policy the execution policy
 * @param ts1 the source tensor
 * @param ts2 the destination tensor
 * @param fun the functor, (e_src) -> e_dst pattern
 */
template <typename _ExectutionPolicy, typename _T1, typename _T2, typename _Fun>
inline void transform(_ExectutionPolicy policy, _T1 ts1, _T2 ts2, _Fun fun,
                      enable_if_t<!are_linear_index<decay_t<_T1>, decay_t<_T2>>::value &&
                                  are_device_memory<decay_t<_T1>, decay_t<_T2>>::value>* = 0) {
    cuda::for_index(policy, 0, ts1.size(), transform_linear_functor<_T1, _T2, _Fun>(ts1, ts2, fun));
}

template <typename _T1, typename _T2, typename _Fun>
struct transform_array_functor {
    _T1 ts1;
    _T2 ts2;
    _Fun fun;

    MATAZURE_DEVICE void operator()(pointi<_T1::rank> idx) const { ts2(idx) = fun(ts1(idx)); }
};

// template

/**
 * @brief transform a array indexing cuda tensor to another by the fun
 * @param policy the execution policy
 * @param ts1 the source tensor
 * @param ts2 the destination tensor
 * @param fun the functor, (e_src) -> e_dst pattern
 */
template <typename _ExectutionPolicy, typename _T1, typename _T2, typename _Fun>
inline void transform(_ExectutionPolicy policy, _T1 ts1, _T2 ts2, _Fun fun,
                      enable_if_t<are_linear_index<decay_t<_T1>>::value &&
                                  are_device_memory<decay_t<_T1>>::value>* = 0) {
    cuda::for_index(policy, zero<pointi<_T1::rank>>::value(), ts1.shape(),
                    transform_array_functor<_T1, _T2, _Fun>{ts1, ts2, fun});
}

/**
 * @brief transform a cuda tensor to another by the fun
 * @param ts1 the source tensor
 * @param ts2 the destination tensor
 * @param fun the functor, (e_src) -> e_dst pattern
 */
template <typename _T1, typename _T2, typename _Fun>
inline void transform(
    const _T1 ts1, _T2 ts2, _Fun fun,
    enable_if_t<are_device_memory<enable_if_t<is_tensor<decay_t<_T1>>::value, decay_t<_T1>>,
                                  decay_t<_T2>>::value>* = 0) {
    for_index_execution_policy policy;
    policy.total_size(ts1.size());
    transform(policy, ts1, ts2, fun);
}

}  // namespace cuda

// use in matazure
using cuda::copy;
using cuda::fill;
using cuda::for_each;
using cuda::transform;

}  // namespace matazure
