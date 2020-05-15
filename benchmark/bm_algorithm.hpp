#pragma once

#include <array>
#include "bm_config.hpp"

template <typename tensor_type>
inline void bm_tensor_fill(benchmark::State& state) {
    int ts_size = state.range(0);
    constexpr int_t rank = tensor_type::rank;
    pointi<rank> shape;
    fill(shape, ts_size);

    auto v = zero<typename tensor_type::value_type>::value();

    tensor_type ts_dst(shape);

    while (state.KeepRunning()) {
        fill(ts_dst, v);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_dst.size()) *
                            sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_dst.size()));
}

template <typename tensor_type>
inline void bm_tensor_for_each(benchmark::State& state) {
    int ts_size = state.range(0);
    constexpr int_t rank = tensor_type::rank;
    pointi<rank> shape;
    fill(shape, ts_size);

    auto v = zero<typename tensor_type::value_type>::value();

    tensor_type ts_dst(shape);

    while (state.KeepRunning()) {
        for_each(ts_dst, [v] MATAZURE_GENERAL(typename tensor_type::value_type & e) { e = v; });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_dst.size()) *
                            sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_dst.size()));
}

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
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(ts_src[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()));
}

template <typename tensor_type>
inline void bm_tensor_transform(benchmark::State& state) {
    int ts_size = state.range(0);
    constexpr int_t rank = tensor_type::rank;
    pointi<rank> shape;
    fill(shape, ts_size);

    tensor_type ts_src(shape);
    tensor_type ts_dst(shape);

    typedef typename tensor_type::value_type value_type;

    while (state.KeepRunning()) {
        transform(ts_src, ts_dst, [] MATAZURE_GENERAL(value_type & v) { return v; });
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(ts_src[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()));
}
