#pragma once

#include <matazure/cuda/execution.hpp>
#include <matazure/cuda/tensor.hpp>
#include <matazure/point.hpp>

namespace matazure {
namespace cuda {

#pragma nv_exec_check_disable
template <typename Function, typename... Arguments>
MATAZURE_GLOBAL void kenel(Function f, Arguments... args) {
    f(args...);
}

template <typename _ExecutionPolicy, typename _Fun, typename... _Args>
inline void launch(_ExecutionPolicy exe_policy, _Fun f, _Args... args) {
    configure_grid(exe_policy, kenel<_Fun, _Args...>);
    kenel<<<exe_policy.grid_size(), exe_policy.block_dim(), exe_policy.shared_mem_bytes(),
            exe_policy.stream()>>>(f, args...);
    assert_runtime_success(cudaGetLastError());
}

template <typename _Fun, typename... _Args>
inline void launch(_Fun f, _Args... args) {
    execution_policy exe_policy;
    launch(exe_policy, f, args...);
}

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
    parallel_execution_policy policy;
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
    column_major_layout<_Rank> layout;
    pointi<_Rank> origin;
};

}  // namespace internal

template <typename _ExecutionPolicy, int_t _Rank, typename _Fun>
inline void for_index(_ExecutionPolicy policy, pointi<_Rank> origin, pointi<_Rank> end, _Fun fun) {
    auto extent = end - origin;
    auto stride = matazure::cumulative_prod(extent);

    column_major_layout<_Rank> layout(extent);
    auto max_size = layout.index2offset(end - 1) + 1;  //要包含最后一个元素

    internal::for_index_array_access_functor<_Fun, _Rank> functor{fun, layout, origin};

    cuda::for_index(policy, 0, max_size, functor);
}

template <int_t _Rank, typename _Fun>
inline void for_index(pointi<_Rank> origin, pointi<_Rank> end, _Fun fun) {
    execution_policy p;
    cuda::for_index(p, origin, end, fun);
}

template <int_t _Rank, typename _Fun>
inline void for_index(pointi<_Rank> end, _Fun fun) {
    cuda::for_index(zero<pointi<_Rank>>::value(), end, fun);
}
}
}  // namespace matazure
