#pragma once

#include <matazure/tensor.hpp>

namespace matazure {

/**
 * @brief reshapes a tensor
 * @param ts source tensor
 * @param ext a valid new shape
 * @return a new ext shape tensor which uses the source tensor memory
 */
template <typename _ValueType, int_t _Rank, typename _Layout, int_t _OutDim,
          typename _OutLayout = _Layout>
inline auto reshape(tensor<_ValueType, _Rank, _Layout> ts, pointi<_OutDim> ext,
                    _OutLayout* = nullptr) -> tensor<_ValueType, _OutDim, _OutLayout> {
    tensor<_ValueType, _OutDim, _OutLayout> re(ext, ts.shared_data());
    MATAZURE_ASSERT(re.size() == ts.size(), "reshape need the size is the same");
    return re;
}

}  // namespace matazure
