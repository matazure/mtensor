#pragma once

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#else
#include <matazure/tensor.hpp>
#endif

namespace matazure {

#ifndef MATAZURE_CUDA

template <typename _Runtime, typename _ValueType, int_t _Rank,
          typename _Layout = row_major_layout<_Rank>,
          typename _Allocator = aligned_allocator<_ValueType, 32>>
struct utensor;

#else

template <typename _Runtime, typename _ValueType, int_t _Rank,
          typename _Layout = row_major_layout<_Rank>,
          typename _Allocator = typename std::conditional<is_same<_Runtime, host_t>::value,
                                                          aligned_allocator<_ValueType, 32>,
                                                          cuda::allocator<_ValueType>>::type>
struct utensor;

#endif

template <typename _ValueType, int_t _Rank, typename _Layout, typename _Allocator>
struct utensor<host_t, _ValueType, _Rank, _Layout, _Allocator>
    : public tensor<_ValueType, _Rank, _Layout, _Allocator> {
    template <typename... _Args>
    utensor(_Args... args) : tensor<_ValueType, _Rank, _Layout, _Allocator>(args...) {}
};

#ifdef MATAZURE_CUDA

template <typename _ValueType, int_t _Rank, typename _Layout, typename _Allocator>
struct utensor<device_t, _ValueType, _Rank, _Layout, _Allocator>
    : public cuda::tensor<_ValueType, _Rank, _Layout, _Allocator> {
    template <typename... _Args>
    utensor(_Args... args) : cuda::tensor<_ValueType, _Rank, _Layout, _Allocator>(args...) {}
};

#endif

}  // namespace matazure
