#pragma once

#include <matazure/view/slice.hpp>

namespace matazure {
namespace view {

template <typename _Tensor>
inline auto pad(_Tensor ts, int_t padding)
    -> decltype(view::slice(ts, pointi<_Tensor::rank>::all(padding), ts.shape() - 2 * padding)) {
    return view::slice(ts, pointi<_Tensor::rank>::all(padding), ts.shape() - 2 * padding);
}

}  // namespace view
}  // namespace matazure
