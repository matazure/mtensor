#pragma once

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#else
#include <matazure/tensor.hpp>
#endif

namespace matazure {

template <typename _Runtime, typename _ValueType, int_t _Rank>
struct tensor_selector;

template <typename _ValueType, int_t _Rank>
struct tensor_selector<host_t, _ValueType, _Rank> {
    typedef tensor<_ValueType, _Rank> type;
};

#ifdef MATAZURE_CUDA

template <typename _ValueType, int_t _Rank>
struct tensor_selector<device_t, _ValueType, _Rank> {
    typedef cuda::tensor<_ValueType, _Rank> type;
};

#endif

}  // namespace matazure