#pragma once

#include <matazure/cuda/execution_policy.hpp>
#include <matazure/cuda/launch.hpp>
#include <matazure/cuda/tensor.hpp>
#include <matazure/point.hpp>

namespace matazure {
namespace cuda {

namespace internal {

inline MATAZURE_GENERAL uint3 pointi_to_uint3(pointi<1> p) {
    return {static_cast<unsigned int>(p[0]), 0, 0};
}

inline MATAZURE_GENERAL uint3 pointi_to_uint3(pointi<2> p) {
    return {static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]), 0};
}

inline MATAZURE_GENERAL uint3 pointi_to_uint3(pointi<3> p) {
    return {static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]),
            static_cast<unsigned int>(p[2])};
}

template <int_t _Rank>
inline MATAZURE_GENERAL pointi<_Rank> uint3_to_pointi(uint3 u);

template <>
inline MATAZURE_GENERAL pointi<1> uint3_to_pointi(uint3 u) {
    return {static_cast<int_t>(u.x)};
}

template <>
inline MATAZURE_GENERAL pointi<2> uint3_to_pointi(uint3 u) {
    return {static_cast<int_t>(u.x), static_cast<int>(u.y)};
}

template <>
inline MATAZURE_GENERAL pointi<3> uint3_to_pointi(uint3 u) {
    return {static_cast<int>(u.x), static_cast<int>(u.y), static_cast<int>(u.z)};
}

inline MATAZURE_GENERAL dim3 pointi_to_dim3(pointi<1> p) {
    return {static_cast<unsigned int>(p[0]), 1, 1};
}

inline MATAZURE_GENERAL dim3 pointi_to_dim3(pointi<2> p) {
    return {static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]), 1};
}

inline MATAZURE_GENERAL dim3 pointi_to_dim3(pointi<3> p) {
    return {static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]),
            static_cast<unsigned int>(p[2])};
}

template <int_t _Rank>
inline MATAZURE_GENERAL pointi<_Rank> dim3_to_pointi(dim3 u);

template <>
inline MATAZURE_GENERAL pointi<1> dim3_to_pointi(dim3 u) {
    return {static_cast<int_t>(u.x)};
}

template <>
inline MATAZURE_GENERAL pointi<2> dim3_to_pointi(dim3 u) {
    return {static_cast<int_t>(u.x), static_cast<int>(u.y)};
}

template <>
inline MATAZURE_GENERAL pointi<3> dim3_to_pointi(dim3 u) {
    return {static_cast<int>(u.x), static_cast<int>(u.y), static_cast<int>(u.z)};
}

}  // namespace internal

template <typename Function, typename... Arguments>
MATAZURE_GLOBAL void kernel(Function f, Arguments... args) {
    f(args...);
}

template <typename _ExecutionPolicy, typename _Fun, typename... _Args>
inline void launch(_ExecutionPolicy exe_policy, _Fun f, _Args... args) {
    configure_grid(exe_policy, kernel<_Fun, _Args...>);
    kernel<<<cuda::internal::pointi_to_dim3(exe_policy.grid_dim()),
             cuda::internal::pointi_to_dim3(exe_policy.block_dim()), exe_policy.shared_mem_bytes(),
             exe_policy.stream()>>>(f, args...);
    assert_runtime_success(cudaGetLastError());
}

template <typename _Fun, typename... _Args>
inline void launch(_Fun f, _Args... args) {
    default_execution_policy exe_policy;
    launch(exe_policy, f, args...);
}

}  // namespace cuda
}  // namespace matazure
