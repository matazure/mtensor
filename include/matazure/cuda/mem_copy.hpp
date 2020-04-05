#pragma once

#include <matazure/cuda/tensor.hpp>

namespace matazure {
namespace cuda {

template <typename _TensorSrc, typename _TensorDst>
inline void mem_copy(_TensorSrc ts_src, _TensorDst cts_dst,
                     enable_if_t<!are_host_memory<_TensorSrc, _TensorDst>::value &&
                                 is_same<typename _TensorSrc::layout_type,
                                         typename _TensorDst::layout_type>::value>* = nullptr) {
    MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_TensorSrc, _TensorDst);

    assert_runtime_success(cudaMemcpy(cts_dst.data(), ts_src.data(),
                                      sizeof(typename _TensorDst::value_type) * ts_src.size(),
                                      cudaMemcpyDefault));
}

template <typename _TensorSrc, typename _TensorSymbol>
inline void copy_symbol(_TensorSrc src, _TensorSymbol& symbol_dst) {
    assert_runtime_success(cudaMemcpyToSymbol(
        symbol_dst, src.data(), src.size() * sizeof(typename _TensorSrc::value_type)));
}

inline void device_synchronize() { assert_runtime_success(cudaDeviceSynchronize()); }

template <typename _ValueType, int_t _Rank>
inline void memset(tensor<_ValueType, _Rank> ts, int v) {
    assert_runtime_success(cudaMemset(ts.shared_data().get(), v, ts.size() * sizeof(_ValueType)));
}

inline void MATAZURE_DEVICE sync_threads() { __syncthreads(); }

template <typename _Type, int_t _Rank, typename _Layout>
inline tensor<_Type, _Rank, _Layout> mem_clone(tensor<_Type, _Rank, _Layout> ts, device_tag) {
    tensor<decay_t<_Type>, _Rank, _Layout> ts_re(ts.shape());
    mem_copy(ts, ts_re);
    return ts_re;
}

template <typename _Type, int_t _Rank, typename _Layout>
inline tensor<_Type, _Rank, _Layout> mem_clone(tensor<_Type, _Rank, _Layout> ts) {
    return mem_clone(ts, device_tag{});
}

template <typename _Type, int_t _Rank, typename _Layout>
inline tensor<_Type, _Rank, _Layout> mem_clone(matazure::tensor<_Type, _Rank, _Layout> ts,
                                               device_tag) {
    tensor<decay_t<_Type>, _Rank, _Layout> ts_re(ts.shape());
    mem_copy(ts, ts_re);
    return ts_re;
}

template <typename _Type, int_t _Rank, typename _Layout>
inline matazure::tensor<_Type, _Rank, _Layout> mem_clone(tensor<_Type, _Rank, _Layout> ts,
                                                         host_tag) {
    matazure::tensor<decay_t<_Type>, _Rank, _Layout> ts_re(ts.shape());
    mem_copy(ts, ts_re);
    return ts_re;
}

}  // namespace cuda

using cuda::mem_clone;
using cuda::mem_copy;

}  // namespace matazure
