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
inline void bm_tensor_general_roll_conv(benchmark::State& state) {
    const static int_t rank = tensor_type::rank;
    typedef typename tensor_type::value_type value_type;

    pointi<rank> shape;
    fill(shape, state.range(0));
    pointi<rank> kernel_shape;
    fill(kernel_shape, 3);
    tensor_type ts_kernel(kernel_shape);
    auto kernel_radius = kernel_shape / 2;
    tensor_type ts_dst(shape);
    tensor_type ts_src_container(shape + (kernel_shape + 1) / 2);
    auto ts_src = view::crop(ts_src_container, kernel_radius, shape);

    while (state.KeepRunning()) {
        _tmp::for_index(ts_dst.shape(), [=] __matazure__(pointi<rank> idx) {
            auto re = zero<value_type>::value();
            for_index(zero<pointi<rank>>::value(), kernel_shape, [&](pointi<rank> idx_offset) {
                re += ts_kernel(idx_offset) * ts_src(idx + idx_offset - kernel_radius);
            });

            ts_dst(idx) = re;
        });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            ts_kernel.size() * 2);
}

template <typename tensor_type>
inline void bm_tensor_general_unroll_conv(benchmark::State& state) {
    const static int_t rank = tensor_type::rank;
    typedef typename tensor_type::value_type value_type;

    pointi<rank> shape;
    fill(shape, state.range(0));
    pointi<rank> kernel_shape;
    fill(kernel_shape, 3);
    tensor_type ts_kernel(kernel_shape);
    auto kernel_radius = kernel_shape / 2;
    tensor_type ts_dst(shape);
    tensor_type ts_src_container(shape + (kernel_shape + 1) / 2);
    auto ts_src = view::crop(ts_src_container, kernel_radius, shape);

    while (state.KeepRunning()) {
        _tmp::for_index(ts_dst.shape(), [=] __matazure__(pointi<rank> idx) {
            // clang-format off
            auto re = zero<value_type>::value();
            re += ts_kernel(pointi<2>{0, 0}) * ts_src(idx + pointi<2>{-1, -1});
            re += ts_kernel(pointi<2>{1, 0}) * ts_src(idx + pointi<2>{ 0, -1});
            re += ts_kernel(pointi<2>{2, 0}) * ts_src(idx + pointi<2>{ 1, -1});
            re += ts_kernel(pointi<2>{0, 1}) * ts_src(idx + pointi<2>{-1,  0});
            re += ts_kernel(pointi<2>{1, 1}) * ts_src(idx + pointi<2>{ 0,  0});
            re += ts_kernel(pointi<2>{2, 1}) * ts_src(idx + pointi<2>{ 1,  0});
            re += ts_kernel(pointi<2>{0, 2}) * ts_src(idx + pointi<2>{-1,  1});
            re += ts_kernel(pointi<2>{1, 2}) * ts_src(idx + pointi<2>{ 0,  1});
            re += ts_kernel(pointi<2>{2, 2}) * ts_src(idx + pointi<2>{ 1,  1});
            ts_dst(idx) = re;
            // clang-format on
        });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) * 9 * 2);
}

template <typename tensor_type>
inline void bm_tensor_padding_layout_general_unroll_conv(benchmark::State& state) {
    const static int_t rank = tensor_type::rank;
    typedef typename tensor_type::value_type value_type;

    pointi<rank> shape;
    fill(shape, state.range(0));
    pointi<rank> kernel_shape;
    fill(kernel_shape, 3);
    tensor_type ts_kernel(kernel_shape);
    auto kernel_radius = kernel_shape / 2;
    tensor_type ts_dst(shape);
    _tmp::tensor<value_type, rank, padding_layout<rank>> ts_src(shape, kernel_radius,
                                                                kernel_radius);

    while (state.KeepRunning()) {
        _tmp::for_index(ts_dst.shape(), [=] __matazure__(pointi<rank> idx) {
            // clang-format off
            auto re = zero<value_type>::value();
            re += ts_kernel(pointi<2>{0, 0}) * ts_src(idx + pointi<2>{-1, -1});
            re += ts_kernel(pointi<2>{1, 0}) * ts_src(idx + pointi<2>{ 0, -1});
            re += ts_kernel(pointi<2>{2, 0}) * ts_src(idx + pointi<2>{ 1, -1});
            re += ts_kernel(pointi<2>{0, 1}) * ts_src(idx + pointi<2>{-1,  0});
            re += ts_kernel(pointi<2>{1, 1}) * ts_src(idx + pointi<2>{ 0,  0});
            re += ts_kernel(pointi<2>{2, 1}) * ts_src(idx + pointi<2>{ 1,  0});
            re += ts_kernel(pointi<2>{0, 2}) * ts_src(idx + pointi<2>{-1,  1});
            re += ts_kernel(pointi<2>{1, 2}) * ts_src(idx + pointi<2>{ 0,  1});
            re += ts_kernel(pointi<2>{2, 2}) * ts_src(idx + pointi<2>{ 1,  1});
            ts_dst(idx) = re;
            // clang-format on
        });

        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(ts_dst[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) * 9 * 2);
}