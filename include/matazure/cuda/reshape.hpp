#pragma once

#include <matazure/cuda/tensor.hpp>

namespace matazure {
namespace cuda {

template <typename _ValueType, int_t _Rank, typename _Layout, int_t _OutDim,
          typename _OutLayout = _Layout>
inline auto reshape(cuda::tensor<_ValueType, _Rank, _Layout> ts, pointi<_OutDim> ext,
                    _OutLayout* = nullptr) -> cuda::tensor<_ValueType, _OutDim, _OutLayout> {
    /// TODO: assert size equal
    cuda::tensor<_ValueType, _OutDim, _OutLayout> re(ext, ts.shared_data());
    return re;
}

}  // namespace cuda

using cuda::reshape;

}  // namespace matazure
