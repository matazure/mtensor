#pragma once

#include "bm_config.hpp"

using namespace matazure;

template <typename _TensorSrc, typename _TensorDst>
static void bm_tensor_mem_copy(benchmark::State& state) {
    _TensorSrc ts_src(state.range(0));
    _TensorDst ts_dst(ts_src.shape());

    while (state.KeepRunning()) {
        mem_copy(ts_src, ts_dst);
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(ts_dst[0]));
}
