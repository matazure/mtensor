#pragma once

#include <matazure/cuda/execution.hpp>
#include <matazure/layout.hpp>
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

template <typename _BlockDim>
class block_index {
   public:
    const static int_t rank = _BlockDim::size();

    MATAZURE_GENERAL block_index(pointi<rank> grid_extent, pointi<rank> local_idx,
                                 pointi<rank> block_idx, pointi<rank> global_idx)
        : block_dim(_BlockDim::value()),
          grid_dim(grid_extent),
          global_dim(block_dim * grid_extent),
          local(local_idx),
          block(block_idx),
          global(global_idx) {}

   public:
    const pointi<rank> block_dim;
    const pointi<rank> grid_dim;
    const pointi<rank> global_dim;
    const pointi<rank> local;
    const pointi<rank> block;
    const pointi<rank> global;
};

template <typename _Ext, typename _Fun, typename _ExecutionPolicy>
inline void block_for_index(_ExecutionPolicy policy, pointi<_Ext::size()> grid_ext, _Fun fun) {
    auto grid_dim = internal::pointi_to_dim3(grid_ext);
    auto block_dim = internal::pointi_to_dim3(_Ext::value());
    kenel<<<grid_dim, block_dim, policy.shared_mem_bytes, policy.stream>>>([=] MATAZURE_DEVICE() {
        auto local = internal::uint3_to_pointi<_Ext::size()>(threadIdx);
        auto block = internal::uint3_to_pointi<_Ext::size()>(blockIdx);
        auto block_dim = internal::dim3_to_pointi<_Ext::size()>(blockDim);
        auto global = block * block_dim + local;
        block_index<_Ext> block_idx(grid_ext, local, block, global);
        fun(block_idx);
    });

    cudaStreamSynchronize(nullptr);

    assert_runtime_success(cudaGetLastError());
}

template <typename _Ext, typename _Fun>
inline void block_for_index(pointi<_Ext::size()> grid_ext, _Fun fun) {
    auto grid_dim = internal::pointi_to_dim3(grid_ext);
    auto block_dim = internal::pointi_to_dim3(_Ext::value());
    kenel<<<grid_dim, block_dim>>>([=] MATAZURE_DEVICE() {
        auto local = internal::uint3_to_pointi<_Ext::size()>(threadIdx);
        auto block = internal::uint3_to_pointi<_Ext::size()>(blockIdx);
        auto block_dim = internal::dim3_to_pointi<_Ext::size()>(blockDim);
        auto global = block * block_dim + local;
        block_index<_Ext> block_idx(grid_ext, local, block, global);
        fun(block_idx);
    });

    cudaStreamSynchronize(nullptr);

    assert_runtime_success(cudaGetLastError());
}

}  // namespace cuda
}  // namespace matazure
