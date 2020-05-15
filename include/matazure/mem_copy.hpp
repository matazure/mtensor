#pragma once

#include <matazure/tensor.hpp>

namespace matazure {

/**
 * @brief memcpy a dense tensor to another dense tensor
 * @param ts_src source tensor
 * @param ts_dst dest tensor
 */
template <typename _TensorSrc, typename _TensorDst>
inline void mem_copy(_TensorSrc ts_src, _TensorDst ts_dst,
                     enable_if_t<are_host_memory<_TensorSrc, _TensorDst>::value &&
                                 is_same<typename _TensorSrc::layout_type,
                                         typename _TensorDst::layout_type>::value>* = nullptr) {
    MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_TensorSrc, _TensorDst);
    memcpy(ts_dst.data(), ts_src.data(), sizeof(typename _TensorDst::value_type) * ts_src.size());
}

/**
 * @brief deeply clone a tensor
 * @param ts source tensor
 * @return a new tensor which clones source tensor
 */
template <typename _ValueType, int_t _Rank, typename _Layout>
inline tensor<_ValueType, _Rank, _Layout> mem_clone(tensor<_ValueType, _Rank, _Layout> ts, host_t) {
    tensor<decay_t<_ValueType>, _Rank, _Layout> ts_re(ts.shape());
    mem_copy(ts, ts_re);
    return ts_re;
}

/**
 * @brief deeply clone a tensor
 * @param ts source tensor
 * @return a new tensor which clones source tensor
 */
template <typename _ValueType, int_t _Rank, typename _Layout>
inline auto mem_clone(tensor<_ValueType, _Rank, _Layout> ts) -> decltype(mem_clone(ts, host_t{})) {
    return mem_clone(ts, host_t{});
}
}  // namespace matazure
