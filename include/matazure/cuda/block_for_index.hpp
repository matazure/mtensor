#pragma once

#include <matazure/cuda/execution_policy.hpp>
#include <matazure/cuda/launch.hpp>
#include <matazure/layout.hpp>
#include <matazure/point.hpp>

namespace matazure {
namespace cuda {

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

template <typename _Ext, typename _Fun>
struct block_for_index_functor {
    MATAZURE_DEVICE void operator()() const {
        auto local = internal::uint3_to_pointi<_Ext::size()>(threadIdx);
        auto block = internal::uint3_to_pointi<_Ext::size()>(blockIdx);
        auto block_dim = internal::dim3_to_pointi<_Ext::size()>(blockDim);
        auto global = block * block_dim + local;
        block_index<_Ext> block_idx(grid_ext, local, block, global);
        fun(block_idx);
    }

    pointi<_Ext::rank> grid_ext;
    _Fun fun;
};

template <typename _Ext, typename _Fun, typename _ExecutionPolicy>
inline void block_for_index(_ExecutionPolicy policy, pointi<_Ext::size()> grid_ext, _Fun fun) {
    auto grid_dim = internal::pointi_to_dim3(grid_ext);
    auto block_dim = internal::pointi_to_dim3(_Ext::value());
    kernel<<<grid_dim, block_dim, policy.shared_mem_bytes(), policy.stream()>>>(
        block_for_index_functor<_Ext, _Fun>{grid_ext, fun});

    assert_runtime_success(cudaGetLastError());
}

template <typename _Ext, typename _Fun>
inline void block_for_index(pointi<_Ext::size()> grid_ext, _Fun fun) {
    auto grid_dim = internal::pointi_to_dim3(grid_ext);
    auto block_dim = internal::pointi_to_dim3(_Ext::value());
    kernel<<<grid_dim, block_dim>>>(block_for_index_functor<_Ext, _Fun>{grid_ext, fun});

    cudaStreamSynchronize(nullptr);

    assert_runtime_success(cudaGetLastError());
}

}  // namespace cuda
}  // namespace matazure
