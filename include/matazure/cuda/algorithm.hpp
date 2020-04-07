#pragma once

#include <matazure/cuda/block_for_index.hpp>
#include <matazure/cuda/for_index.hpp>

namespace matazure {
namespace cuda {

template <typename _ExecutionPolicy, typename _Tensor, typename _Fun>
inline void for_each(
    _ExecutionPolicy policy, _Tensor ts, _Fun fun,
    enable_if_t<are_device_memory<_Tensor>::value && are_linear_index<_Tensor>::value>* = 0) {
    cuda::for_index(policy, 0, ts.size(), [=] MATAZURE_DEVICE(int_t i) { fun(ts[i]); });
}

template <typename _ExecutionPolicy, typename _Tensor, typename _Fun>
inline void for_each(_ExecutionPolicy policy, _Tensor ts, _Fun fun,
                     enable_if_t<are_device_memory<decay_t<_Tensor>>::value &&
                                 !are_linear_index<decay_t<_Tensor>>::value>* = 0) {
    cuda::for_index(policy, zero<pointi<_Tensor::rank>>::value(), ts.shape(),
                    [=] MATAZURE_DEVICE(pointi<_Tensor::rank> idx) { fun(ts[idx]); });
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

template <typename _ExecutionPolicy, typename _Tensor>
inline void fill(
    _ExecutionPolicy policy, _Tensor ts, typename _Tensor::value_type v,
    enable_if_t<
        are_device_memory<enable_if_t<is_tensor<decay_t<_Tensor>>::value, _Tensor>>::value>* = 0) {
    for_each(policy, ts,
             [v] MATAZURE_DEVICE(typename _Tensor::value_type & element) { element = v; });
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

template <typename _ExecutionPolicy, typename _T1, typename _T2>
void copy(
    _ExecutionPolicy policy, _T1 lhs, _T2 rhs,
    enable_if_t<are_linear_index<_T1, _T2>::value && are_device_memory<_T1, _T2>::value>* = 0) {
    cuda::for_index(policy, 0, lhs.size(), [=] MATAZURE_DEVICE(int_t i) { rhs[i] = lhs[i]; });
}

template <typename _ExecutionPolicy, typename _T1, typename _T2>
void copy(
    _ExecutionPolicy policy, _T1 lhs, _T2 rhs,
    enable_if_t<!are_linear_index<_T1, _T2>::value && are_device_memory<_T1, _T2>::value>* = 0) {
    cuda::for_index(policy, zero<pointi<_T1::rank>>::value(), lhs.shape(),
                    [=] MATAZURE_DEVICE(pointi<_T1::rank> idx) { rhs(idx) = lhs(idx); });
}

template <typename _T1, typename _T2>
void copy(
    _T1 lhs, _T2 rhs,
    enable_if_t<are_device_memory<enable_if_t<is_tensor<_T1>::value, _T1>, _T2>::value>* = 0) {
    for_index_execution_policy policy;
    policy.total_size(lhs.size());
    copy(policy, lhs, rhs);
}

/**
 * @brief transform a linear indexing cuda tensor to another by the fun
 * @param policy the execution policy
 * @param ts_src the source tensor
 * @param ts_dst the destination tensor
 * @param fun the functor, (e_src) -> e_dst pattern
 */
template <typename _ExectutionPolicy, typename _TensorSrc, typename _TensorDst, typename _Fun>
inline void transform(
    _ExectutionPolicy policy, _TensorSrc ts_src, _TensorDst ts_dst, _Fun fun,
    enable_if_t<!are_linear_index<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value &&
                are_device_memory<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value>* = 0) {
    cuda::for_index(policy, 0, ts_src.size(),
                    [=] MATAZURE_DEVICE(int_t i) { ts_dst[i] = fun(ts_src[i]); });
}

/**
 * @brief transform a array indexing cuda tensor to another by the fun
 * @param policy the execution policy
 * @param ts_src the source tensor
 * @param ts_dst the destination tensor
 * @param fun the functor, (e_src) -> e_dst pattern
 */
template <typename _ExectutionPolicy, typename _TensorSrc, typename _TensorDst, typename _Fun>
inline void transform(_ExectutionPolicy policy, _TensorSrc ts_src, _TensorDst ts_dst, _Fun fun,
                      enable_if_t<are_linear_index<decay_t<_TensorSrc>>::value &&
                                  are_device_memory<decay_t<_TensorSrc>>::value>* = 0) {
    cuda::for_index(
        policy, pointi<_TensorSrc::rank>::zeros(), ts_src.shape(),
        [=] MATAZURE_DEVICE(pointi<_TensorSrc::rank> idx) { ts_dst[idx] = fun(ts_src[idx]); });
}

/**
 * @brief transform a cuda tensor to another by the fun
 * @param ts_src the source tensor
 * @param ts_dst the destination tensor
 * @param fun the functor, (e_src) -> e_dst pattern
 */
template <typename _TensorSrc, typename _TensorDst, typename _Fun>
inline void transform(
    const _TensorSrc ts_src, _TensorDst ts_dst, _Fun fun,
    enable_if_t<
        are_device_memory<enable_if_t<is_tensor<decay_t<_TensorSrc>>::value, decay_t<_TensorSrc>>,
                          decay_t<_TensorDst>>::value>* = 0) {
    for_index_execution_policy policy;
    policy.total_size(ts_src.size());
    transform(policy, ts_src, ts_dst, fun);
}

}  // namespace cuda

// use in matazure
using cuda::copy;
using cuda::fill;
using cuda::for_each;
using cuda::transform;

}  // namespace matazure
