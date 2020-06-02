#pragma once

#include <matazure/cuda/execution_policy.hpp>
#include <matazure/cuda/launch.hpp>
#include <matazure/cuda/tensor.hpp>
#include <matazure/point.hpp>

namespace matazure {
namespace cuda {

namespace internal {

template <typename _Fun>
struct for_index_functor {
    int first;
    int last;
    _Fun fun;

    __device__ void operator()() {
        for (int_t i = first + threadIdx.x + blockIdx.x * blockDim.x; i < last;
             i += blockDim.x * gridDim.x) {
            fun(i);
        };
    }
};

}  // namespace internal

template <typename _ExecutionPolicy, typename _Fun>
inline void for_index(_ExecutionPolicy policy, int_t first, int_t last, _Fun fun) {
    internal::for_index_functor<_Fun> func{first, last, fun};
    launch(policy, func);
}

template <typename _Fun>
inline void for_index(int_t first, int_t last, _Fun fun) {
    for_index_execution_policy policy;
    policy.total_size(last - first);
    cuda::for_index(policy, first, last, fun);
}

template <typename _Fun>
inline void for_index(int_t last, _Fun fun) {
    for_index(0, last, fun);
}

namespace internal {

template <typename _Fun, int_t _Rank>
struct for_index_array_access_functor {
    __device__ void operator()(int_t i) { fun(layout.offset2index(i) + origin); }

    _Fun fun;
    row_major_layout<_Rank> layout;
    pointi<_Rank> origin;
};

}  // namespace internal

template <typename _ExecutionPolicy, int_t _Rank, typename _Fun>
inline void for_index(_ExecutionPolicy policy, pointi<_Rank> origin, pointi<_Rank> end, _Fun fun) {
    auto extent = end - origin;
    auto stride = matazure::scan_multiply(extent);

    row_major_layout<_Rank> layout(extent);
    auto max_size = layout.index2offset(end - 1) + 1;  //要包含最后一个元素

    internal::for_index_array_access_functor<_Fun, _Rank> functor{fun, layout, origin};

    cuda::for_index(policy, 0, max_size, functor);
}

template <int_t _Rank, typename _Fun>
inline void for_index(pointi<_Rank> origin, pointi<_Rank> end, _Fun fun) {
    default_execution_policy p;
    cuda::for_index(p, origin, end, fun);
}

template <int_t _Rank, typename _Fun>
inline void for_index(pointi<_Rank> end, _Fun fun) {
    cuda::for_index(zero<pointi<_Rank>>::value(), end, fun);
}
}  // namespace cuda

template <typename... _Args>
inline void for_index(host_t, _Args... args) {
    matazure::for_index(args...);
}

template <typename... _Args>
inline void for_index(device_t, _Args... args) {
    matazure::cuda::for_index(args...);
}

}  // namespace matazure
