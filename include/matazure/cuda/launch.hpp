#pragma once

#include <matazure/cuda/execution_policy.hpp>
#include <matazure/cuda/launch.hpp>
#include <matazure/cuda/tensor.hpp>
#include <matazure/point.hpp>

namespace matazure {
namespace cuda {

template <typename Function, typename... Arguments>
MATAZURE_GLOBAL void kenel(Function f, Arguments... args) {
    f(args...);
}

template <typename _ExecutionPolicy, typename _Fun, typename... _Args>
inline void launch(_ExecutionPolicy exe_policy, _Fun f, _Args... args) {
    configure_grid(exe_policy, kenel<_Fun, _Args...>);
    kenel<<<exe_policy.grid_dim(), exe_policy.block_dim(), exe_policy.shared_mem_bytes(),
            exe_policy.stream()>>>(f, args...);
    assert_runtime_success(cudaGetLastError());
}

template <typename _Fun, typename... _Args>
inline void launch(_Fun f, _Args... args) {
    execution_policy exe_policy;
    launch(exe_policy, f, args...);
}

}  // namespace cuda
}  // namespace matazure
