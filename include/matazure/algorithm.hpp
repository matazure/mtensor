#pragma once

#include <matazure/for_index.hpp>
#include <matazure/layout.hpp>

namespace matazure {

/**
 * @brief for each element of a linear indexing tensor, apply fun
 * @param policy the execution policy
 * @param ts the source tensor
 * @param fun the functor, (element &) -> none pattern
 */
template <typename _ExectutionPolicy, typename _Tensor, typename _Fun>
inline void for_each(_ExectutionPolicy policy, _Tensor&& ts, _Fun fun,
                     enable_if_t<are_linear_index<decay_t<_Tensor>>::value &&
                                 none_device_memory<decay_t<_Tensor>>::value>* = 0) {
    for_index(policy, 0, ts.size(), [&](int_t i) { fun(ts[i]); });
}

/**
 * @brief for each element of an array indexing tensor, apply fun
 * @param policy the execution policy
 * @param ts the source tensor
 * @param fun the functor, (element &) -> none pattern
 */
template <typename _ExectutionPolicy, typename _Tensor, typename _Fun>
inline void for_each(_ExectutionPolicy policy, _Tensor&& ts, _Fun fun,
                     enable_if_t<!are_linear_index<decay_t<_Tensor>>::value &&
                                 none_device_memory<decay_t<_Tensor>>::value>* = 0) {
    for_index(policy, zero<pointi<decay_t<_Tensor>::rank>>::value(), ts.shape(),
              [&](pointi<decay_t<_Tensor>::rank> idx) {
                  fun(ts(internal::get_array_index_by_layout(idx, ts.layout())));
              });
}

/**
 * @brief for each element of a tensor, apply fun by the sequence policy
 * @param ts the source tensor
 * @param fun the functor, (element &) -> none pattern
 */
template <typename _Tensor, typename _Fun>
inline void for_each(
    _Tensor&& ts, _Fun fun,
    enable_if_t<none_device_memory<
        enable_if_t<is_linear_array<decay_t<_Tensor>>::value, decay_t<_Tensor>>>::value>* = 0) {
    sequence_policy policy{};
    for_each(policy, std::forward<_Tensor>(ts), fun);
}

/**
 * @brief fill a tensor value elementwise
 * @param policy the execution policy
 * @param ts the source tensor
 * @param v the filled value
 */
template <typename _ExectutionPolicy, typename _Tensor>
inline void fill(_ExectutionPolicy policy, _Tensor&& ts, typename decay_t<_Tensor>::value_type v,
                 enable_if_t<none_device_memory<decay_t<_Tensor>>::value>* = 0) {
    for_each(policy, std::forward<_Tensor>(ts),
             [v](typename decay_t<_Tensor>::value_type& x) { x = v; });
}

/**
 * @brief fill a tensor value elementwise by the sequence policy
 * @param ts the source tensor
 * @param v the filled value
 */
template <typename _Tensor>
inline void fill(
    _Tensor&& ts, typename decay_t<_Tensor>::value_type v,
    enable_if_t<none_device_memory<
        enable_if_t<is_linear_array<decay_t<_Tensor>>::value, decay_t<_Tensor>>>::value>* = 0) {
    sequence_policy policy{};
    fill(policy, std::forward<_Tensor>(ts), v);
}

/**
 * @brief elementwisely copy a linear indexing tensor to another one
 * @param policy the execution policy
 * @param ts_src the source tensor
 * @param ts_dst the dest tensor
 */
template <typename _ExectutionPolicy, typename _TensorSrc, typename _TensorDst>
inline void copy(
    _ExectutionPolicy policy, const _TensorSrc& ts_src, _TensorDst&& ts_dst,
    enable_if_t<are_linear_index<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value &&
                none_device_memory<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value>* = 0) {
    for_index(policy, 0, ts_src.size(), [&](int_t i) { ts_dst[i] = ts_src[i]; });
}

/**
 * @brief elementwisely copy a array indexing tensor to another one
 * @param policy the execution policy
 * @param ts_src the source tensor
 * @param ts_dst the dest tensor
 */
template <typename _ExectutionPolicy, typename _TensorSrc, typename _TensorDst>
inline void copy(
    _ExectutionPolicy policy, const _TensorSrc& ts_src, _TensorDst&& ts_dst,
    enable_if_t<!are_linear_index<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value &&
                none_device_memory<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value>* = 0) {
    for_index(policy, zero<pointi<_TensorSrc::rank>>::value(), ts_src.shape(),
              [&](pointi<_TensorSrc::rank> idx) {
                  ts_dst(idx) = ts_src(internal::get_array_index_by_layout(idx, ts_src.layout()));
              });
}

/**
 * @brief elementwisely copy a tensor to another one by the sequence policy
 * @param ts_src the source tensor
 * @param ts_dst the dest tensor
 */
template <typename _TensorSrc, typename _TensorDst>
inline void copy(const _TensorSrc& ts_src, _TensorDst&& ts_dst,
                 enable_if_t<none_device_memory<
                     enable_if_t<is_linear_array<decay_t<_TensorSrc>>::value, decay_t<_TensorSrc>>,
                     decay_t<_TensorDst>>::value>* = 0) {
    sequence_policy policy;
    copy(policy, ts_src, std::forward<_TensorDst>(ts_dst));
}

/**
 * @brief transform a linear indexing tensor to another by the fun
 * @param policy the execution policy
 * @param ts_src the source tensor
 * @param ts_dst the destination tensor
 * @param fun the functor, (e_src) -> e_dst pattern
 */
template <typename _ExectutionPolicy, typename _TensorSrc, typename _TensorDst, typename _Fun>
inline void transform(
    _ExectutionPolicy policy, const _TensorSrc& ts_src, _TensorDst&& ts_dst, _Fun fun,
    enable_if_t<are_linear_index<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value &&
                none_device_memory<decay_t<_TensorSrc>, decay_t<_TensorDst>>::value>* = 0) {
    for_index(policy, 0, ts_src.size(), [&](int_t i) { ts_dst[i] = fun(ts_src[i]); });
}

/**
 * @brief transform a array indexing tensor to another by the fun
 * @param policy the execution policy
 * @param ts_src the source tensor
 * @param ts_dst the destination tensor
 * @param fun the functor, (e_src) -> e_dst pattern
 */
template <typename _ExectutionPolicy, typename _TensorSrc, typename _TensorDst, typename _Fun>
inline void transform(_ExectutionPolicy policy, const _TensorSrc& ts_src, _TensorDst&& ts_dst,
                      _Fun fun,
                      enable_if_t<!are_linear_index<decay_t<_TensorSrc>>::value &&
                                  none_device_memory<decay_t<_TensorSrc>>::value>* = 0) {
    for_index(policy, zero<pointi<_TensorSrc::rank>>::value(), ts_src.shape(),
              [&](pointi<_TensorSrc::rank> idx) {
                  ts_dst(idx) =
                      fun(ts_src(internal::get_array_index_by_layout(idx, ts_src.layout())));
              });
}

/**
 * @brief transform a tensor to another by the fun
 * @param ts_src the source tensor
 * @param ts_dst the destination tensor
 * @param fun the functor, (e_src) -> e_dst pattern
 */
template <typename _TensorSrc, typename _TensorDst, typename _Fun>
inline void transform(
    const _TensorSrc& ts_src, _TensorDst&& ts_dst, _Fun fun,
    enable_if_t<none_device_memory<
        enable_if_t<is_linear_array<decay_t<_TensorSrc>>::value, decay_t<_TensorSrc>>,
        decay_t<_TensorDst>>::value>* = 0) {
    sequence_policy policy{};
    transform(policy, ts_src, std::forward<_TensorDst>(ts_dst), fun);
}

/**
 * @brief reduce the elements of a tensor
 * @param the execution policy
 * @param ts the source tensor
 * @param init the initial value
 * @param binary_fun the reduce functor, must be (element, element)-> value pattern
 */
///@todo
// template <typename _ExectutionPolicy, typename _Tensor, typename _VT, typename _BinaryFunc>
// inline   _VT reduce(_ExectutionPolicy policy, _Tensor ts, _VT init, _BinaryFunc binary_fun,
// enable_if_t<none_device_memory<decay_t<_Tensor>>::value>* = 0) { 	auto re = init;
// 	for_each(policy, ts, [&re, binary_fun](decltype(ts[0]) x) {
// 		re = binary_fun(re, x);
// 	});
//
// 	return re;
// }

/**
 * @brief reduce the elements of a tensor by the sequence policy
 * @param ts the source tensor
 * @param init the initial value
 * @param binary_fun the reduce functor, must be (element, element)-> value pattern
 */
template <typename _Tensor, typename _VT, typename _BinaryFunc>
inline _VT reduce(const _Tensor& ts, _VT init, _BinaryFunc binary_fun) {
    sequence_policy policy{};
    auto re = init;
    for_each(policy, ts, [&re, binary_fun](decltype(ts[0]) x) { re = binary_fun(re, x); });

    return re;
}

// template <typename _Tensor>
// inline   auto sum(const _Tensor &ts) {
// 	return reduce(ts, zero<typename _Tensor::value_type>::value(), [](auto lhs, auto rhs) {
// 		return lhs + rhs;
// 	});
// }

}  // namespace matazure
