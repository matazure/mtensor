#pragma once

#include "bm_config.hpp"

template <typename tensor_type>
inline void bm_tensor_copy(benchmark::State& state) {
    int ts_size = state.range(0);
    constexpr int_t rank = tensor_type::rank;
    pointi<rank> shape;
    fill(shape, ts_size);

    tensor_type ts_src(shape);
    tensor_type ts_dst(shape);

    while (state.KeepRunning()) {
        copy(ts_src, ts_dst);
        benchmark::DoNotOptimize(ts_src.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(ts_src[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()));
}
