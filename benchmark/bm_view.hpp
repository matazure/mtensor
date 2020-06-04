#pragma once

#include "bm_config.hpp"

template <typename tensor_type>
inline void bm_tensor_view_slice(benchmark::State& state) {
    tensor_type ts_src(pointi<tensor_type::rank>::all(state.range(0)));
    auto center = ts_src.shape() / 4;
    auto dst_shape = ts_src.shape() / 2;
    tensor_type ts_dst(dst_shape);

    while (state.KeepRunning()) {
        copy(view::slice(ts_src, center, dst_shape), ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * ts_dst.size() * sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * ts_dst.size());
}

template <typename tensor_type>
inline void bm_tensor_view_stride(benchmark::State& state) {
    tensor_type ts_src(pointi<tensor_type::rank>::all(state.range(0)));
    auto center = ts_src.shape() / 4;
    auto dst_shape = ts_src.shape() / 2;
    tensor_type ts_dst(dst_shape);

    while (state.KeepRunning()) {
        copy(view::stride(ts_src, 2), ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * ts_dst.size() * sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * ts_dst.size());
}

template <typename tensor_type>
inline void bm_tensor_view_gather_scalar_axis0(benchmark::State& state) {
    typedef utensor<runtime_t<tensor_type>, typename tensor_type::value_type, tensor_type::rank - 1>
        dst_tensor_type;
    tensor_type ts_src(point2i{10, state.range(0)});
    dst_tensor_type ts_dst(gather_point<0>(ts_src.shape()));

    int i = 0;
    while (state.KeepRunning()) {
        ++i;
        copy(view::gather<0>(ts_src, i % ts_src.shape(0)), ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * ts_dst.size() * sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * ts_dst.size());
}

template <typename tensor_type>
inline void bm_tensor_view_gather_scalar_axis1(benchmark::State& state) {
    typedef utensor<runtime_t<tensor_type>, typename tensor_type::value_type, tensor_type::rank - 1>
        dst_tensor_type;
    tensor_type ts_src(point2i{state.range(0), 10});
    dst_tensor_type ts_dst(gather_point<1>(ts_src.shape()));

    int i = 0;
    while (state.KeepRunning()) {
        ++i;
        copy(view::gather<1>(ts_src, i % ts_src.shape(1)), ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * ts_dst.size() * sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * ts_dst.size());
}

#ifndef MATAZURE_CUDA
template <typename tensor_type>
inline void bm_tensor_view_zip2(benchmark::State& state) {
    tensor_type ts0(pointi<tensor_type::rank>::all(state.range(0)));
    tensor_type ts1(ts0.shape());

    typedef utensor<runtime_t<tensor_type>,
                    tuple<typename tensor_type::value_type, typename tensor_type::value_type>,
                    tensor_type::rank>
        tensor_tuple_type;

    tensor_tuple_type ts_dst(ts0.shape());

    while (state.KeepRunning()) {
        copy(view::zip(ts0, ts1), ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * ts_dst.size() *
                            sizeof(typename tensor_tuple_type::value_type));
    state.SetItemsProcessed(state.iterations() * ts_dst.size());
}
#endif

template <typename tensor_type>
inline void bm_tensor_view_eye(benchmark::State& state) {
    tensor_type ts_dst(pointi<tensor_type::rank>::all(state.range(0)));
    auto center = ts_dst.shape() / 4;

    while (state.KeepRunning()) {
        copy(view::eye<typename tensor_type::value_type>(ts_dst.shape(), ts_dst.runtime(),
                                                         ts_dst.layout()),
             ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * ts_dst.size() * sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * ts_dst.size());
}

#ifndef MATAZURE_CUDA
template <typename runtime_type, typename value_type>
inline void bm_tensor_view_meshgrid2(benchmark::State& state) {
    point2i shape{state.range(0), state.range(0)};
    typedef utensor<runtime_type, point<value_type, 2>, 2> tensor_type;
    tensor_type ts(shape);

    typedef utensor<runtime_type, value_type, 1> vector_type;
    vector_type v0(shape[0]);
    vector_type v1(shape[1]);

    while (state.KeepRunning()) {
        copy(view::meshgrid(v0, v1), ts);
        benchmark::DoNotOptimize(ts.data());
    }

    state.SetBytesProcessed(state.iterations() * ts.size() * sizeof(ts[0]));
    state.SetItemsProcessed(state.iterations() * ts.size());
}
#endif
