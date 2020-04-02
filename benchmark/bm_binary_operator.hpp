#pragma once

#include "bm_config.hpp"

#define BM_TENSOR_BINARY_OPERATOR_FUNC(OpName, Op)                                        \
    template <typename _Tensor>                                                           \
    inline void bm_tensor_##OpName(benchmark::State& state) {                             \
        _Tensor ts0(pointi<_Tensor::rank>::all(state.range(0)));                          \
        _Tensor ts1(ts0.shape());                                                         \
        fill(ts0, zero<typename _Tensor::value_type>::value());                           \
        fill(ts1, zero<typename _Tensor::value_type>::value());                           \
        decltype((ts0 Op ts1).persist()) ts_re(ts0.shape());                              \
                                                                                          \
        while (state.KeepRunning()) {                                                     \
            copy(ts0 Op ts1, ts_re);                                                      \
            benchmark::DoNotOptimize(ts_re.data());                                       \
        }                                                                                 \
                                                                                          \
        auto bytes_size = static_cast<size_t>(ts_re.size()) * sizeof(decltype(ts_re[0])); \
        state.SetBytesProcessed(state.iterations() * bytes_size * 2);                     \
        state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts0.size()));    \
    }

// Arithmetic
BM_TENSOR_BINARY_OPERATOR_FUNC(add, +)
BM_TENSOR_BINARY_OPERATOR_FUNC(sub, -)
BM_TENSOR_BINARY_OPERATOR_FUNC(mul, *)
BM_TENSOR_BINARY_OPERATOR_FUNC(div, /)
