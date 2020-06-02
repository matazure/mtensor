#pragma once

#include "bm_config.hpp"

namespace _tmp {
#ifndef MATAZURE_CUDA
using matazure::for_index;
using matazure::tensor;
#else
using matazure::cuda::for_index;
using matazure::cuda::tensor;
#endif
}

template <typename tensor_type>
inline void bm_tensor2f_general_roll_conv(benchmark::State& state) {
    const static int_t rank = tensor_type::rank;
    typedef typename tensor_type::value_type value_type;

    pointi<rank> shape;
    fill(shape, state.range(0));
    pointi<rank> kernel_shape;
    fill(kernel_shape, 3);
    tensor_type kernel(kernel_shape);
    auto kernel_radius = kernel_shape / 2;
    tensor_type ts_dst(shape);
    tensor_type ts_src_pad(shape + kernel.shape() - 1);
    auto ts_src_view = view::slice(ts_src_pad, kernel_radius, shape);

    while (state.KeepRunning()) {
        _tmp::for_index(ts_dst.shape(), MLAMBDA(pointi<rank> idx) {
            auto re = zero<value_type>::value();
            for_index(zero<pointi<rank>>::value(), kernel_shape, [&](pointi<rank> neigbor_idx) {
                re += kernel(neigbor_idx) * ts_src_view(idx + neigbor_idx - kernel_radius);
            });

            ts_dst(idx) = re;
        });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src_pad.size()) *
                            sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_dst.size()) *
                            kernel.size() * 2);
}

template <typename tensor_type>
inline void bm_tensor2f_general_unroll_conv(benchmark::State& state) {
    const static int_t rank = tensor_type::rank;
    typedef typename tensor_type::value_type value_type;

    pointi<rank> shape;
    fill(shape, state.range(0));
    pointi<rank> kernel_shape;
    fill(kernel_shape, 3);
    tensor_type kernel(kernel_shape);
    auto kernel_radius = kernel_shape / 2;
    tensor_type ts_dst(shape);
    tensor_type ts_src_pad(shape + kernel.shape() - 1);
    auto ts_src_view = view::slice(ts_src_pad, kernel_radius, shape);

    while (state.KeepRunning()) {
        _tmp::for_index(ts_dst.shape(), MLAMBDA(pointi<rank> idx) {
            // clang-format off
            auto re = zero<value_type>::value();
            re += kernel(pointi<2>{0, 0}) * ts_src_view(idx + pointi<2>{-1, -1});
            re += kernel(pointi<2>{1, 0}) * ts_src_view(idx + pointi<2>{ 0, -1});
            re += kernel(pointi<2>{2, 0}) * ts_src_view(idx + pointi<2>{ 1, -1});
            re += kernel(pointi<2>{0, 1}) * ts_src_view(idx + pointi<2>{-1,  0});
            re += kernel(pointi<2>{1, 1}) * ts_src_view(idx + pointi<2>{ 0,  0});
            re += kernel(pointi<2>{2, 1}) * ts_src_view(idx + pointi<2>{ 1,  0});
            re += kernel(pointi<2>{0, 2}) * ts_src_view(idx + pointi<2>{-1,  1});
            re += kernel(pointi<2>{1, 2}) * ts_src_view(idx + pointi<2>{ 0,  1});
            re += kernel(pointi<2>{2, 2}) * ts_src_view(idx + pointi<2>{ 1,  1});
            ts_dst(idx) = re;
            // clang-format on
        });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src_pad.size()) *
                            sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_dst.size()) *
                            kernel.size() * 2);
}

template <typename tensor_type>
inline void bm_tensor2f_padding_layout_general_unroll_conv(benchmark::State& state) {
    const static int_t rank = tensor_type::rank;
    typedef typename tensor_type::value_type value_type;

    pointi<rank> shape;
    fill(shape, state.range(0));
    pointi<rank> kernel_shape;
    fill(kernel_shape, 3);
    tensor_type kernel(kernel_shape);
    auto kernel_radius = kernel_shape / 2;
    tensor_type ts_dst(shape);
    _tmp::tensor<value_type, rank, padding_layout<rank>> ts_src(shape, kernel_radius,
                                                                kernel_radius);

    while (state.KeepRunning()) {
        _tmp::for_index(ts_dst.shape(), MLAMBDA(pointi<rank> idx) {
            // clang-format off
            auto re = zero<value_type>::value();
            re += kernel(pointi<2>{0, 0}) * ts_src(idx + pointi<2>{-1, -1});
            re += kernel(pointi<2>{1, 0}) * ts_src(idx + pointi<2>{ 0, -1});
            re += kernel(pointi<2>{2, 0}) * ts_src(idx + pointi<2>{ 1, -1});
            re += kernel(pointi<2>{0, 1}) * ts_src(idx + pointi<2>{-1,  0});
            re += kernel(pointi<2>{1, 1}) * ts_src(idx + pointi<2>{ 0,  0});
            re += kernel(pointi<2>{2, 1}) * ts_src(idx + pointi<2>{ 1,  0});
            re += kernel(pointi<2>{0, 2}) * ts_src(idx + pointi<2>{-1,  1});
            re += kernel(pointi<2>{1, 2}) * ts_src(idx + pointi<2>{ 0,  1});
            re += kernel(pointi<2>{2, 2}) * ts_src(idx + pointi<2>{ 1,  1});
            ts_dst(idx) = re;
            // clang-format on
        });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_dst.size()) *
                            kernel.size() * 2);
}

template <typename tensor_type>
inline void bm_tensor2_view_conv_local_tensor3x3(benchmark::State& state) {
    const static int_t rank = tensor_type::rank;
    typedef typename tensor_type::value_type value_type;

    pointi<rank> shape;
    fill(shape, state.range(0));
    local_tensor<value_type, dim<3, 3>> kernel;

    tensor_type ts_src_pad(shape + kernel.shape() - 1);
    auto ts_src_view = view::slice(ts_src_pad, kernel.shape() / 2, shape);
    tensor_type ts_dst(shape);

    while (state.KeepRunning()) {
        copy(view::conv(ts_src_view, kernel), ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src_pad.size()) *
                            sizeof(ts_src_view[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_dst.size()) *
                            kernel.size() * 2);
}

template <typename tensor_type>
inline void bm_tensor_view_conv_tensor3x3(benchmark::State& state) {
    const static int_t rank = tensor_type::rank;
    typedef typename tensor_type::value_type value_type;

    pointi<rank> shape;
    fill(shape, state.range(0));

    pointi<rank> kernel_shape;
    fill(kernel_shape, 3);
    tensor_type kernel(kernel_shape);

    tensor_type ts_src_pad(shape + kernel.shape() - 1);
    auto ts_src_view = view::slice(ts_src_pad, kernel.shape() / 2, shape);
    tensor_type ts_dst(shape);

    while (state.KeepRunning()) {
        copy(view::conv(ts_src_view, kernel), ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src_pad.size()) *
                            sizeof(ts_src_view[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_dst.size()) *
                            kernel.size() * 2);
}

template <typename tensor_type>
inline void bm_tensor2_view_conv_neighbors_weights3x3(benchmark::State& state) {
    const static int_t rank = tensor_type::rank;
    typedef typename tensor_type::value_type value_type;

    pointi<rank> shape;
    fill(shape, state.range(0));

    tensor<tuple<pointi<rank>, value_type>, 1> neightbors_weights(9);
    neightbors_weights[0] = make_tuple(pointi<rank>{-1, -1}, value_type{1});
    neightbors_weights[0] = make_tuple(pointi<rank>{-1, -0}, value_type{1});
    neightbors_weights[0] = make_tuple(pointi<rank>{-1, +1}, value_type{1});
    neightbors_weights[0] = make_tuple(pointi<rank>{-0, -1}, value_type{1});
    neightbors_weights[0] = make_tuple(pointi<rank>{-0, -0}, value_type{1});
    neightbors_weights[0] = make_tuple(pointi<rank>{-0, +1}, value_type{1});
    neightbors_weights[0] = make_tuple(pointi<rank>{+1, -1}, value_type{1});
    neightbors_weights[0] = make_tuple(pointi<rank>{+1, -0}, value_type{1});
    neightbors_weights[0] = make_tuple(pointi<rank>{+1, +1}, value_type{1});

    pointi<2> padding = {1, 1};
    tensor_type ts_src_pad(shape + 2 * padding);
    auto ts_src_view = view::slice(ts_src_pad, padding, shape);
    tensor_type ts_dst(shape);

    while (state.KeepRunning()) {
        copy(view::conv(ts_src_view, neightbors_weights), ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src_pad.size()) *
                            sizeof(ts_src_view[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_dst.size()) *
                            neightbors_weights.size() * 2);
}

template <typename tensor_type>
inline void bm_tensor2_view_conv_stride2_relu6_local_tensor3x3(benchmark::State& state) {
    const static int_t rank = tensor_type::rank;
    typedef typename tensor_type::value_type value_type;

    pointi<rank> shape;
    fill(shape, state.range(0));
    local_tensor<value_type, dim<3, 3>> kernel;

    tensor_type ts_src_pad(shape + kernel.shape() - 1);
    auto ts_src_view = view::slice(ts_src_pad, kernel.shape() / 2, shape);
    tensor_type ts_dst(shape / 2);

    while (state.KeepRunning()) {
        auto ts_conv_view = view::conv(ts_src_view, kernel);
        auto ts_conv_stride_view = view::stride(ts_conv_view, pointi<2>{2, 2});
        auto ts_conv_stride_relu6_view =
            view::map(ts_conv_stride_view, [] MATAZURE_GENERAL(value_type v) { return v; });
        copy(ts_conv_stride_view, ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src_pad.size()) *
                            sizeof(ts_src_view[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_dst.size()) *
                            kernel.size() * 2);
}
