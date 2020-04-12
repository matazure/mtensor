#include "../bm_config.hpp"

void bm_cuda_block_for_index_flops(benchmark::State& state) {
    pointi<2> grid_dim;
    using BLOCK_DIM = dim<16, 16>;
    auto block_dim = BLOCK_DIM::value();
    fill(grid_dim, state.range(0));
    auto global_dim = block_dim * grid_dim;
    cuda::tensor<float, 2> ts_src(global_dim);
    cuda::tensor<float, 2> ts_dst(global_dim);
    while (state.KeepRunning()) {
        cuda::block_for_index<BLOCK_DIM>(grid_dim,
                                         [=] __device__(cuda::block_index<BLOCK_DIM> block_idx) {
                                             auto tmp = ts_src(block_idx.global);
                                             for (int_t k = 0; k < 1000000; ++k) {
                                                 tmp *= 1.01f;
                                             }
                                             ts_dst(block_idx.global) = tmp;
                                         });
    }

    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) * 1000000);
}

BENCHMARK(bm_cuda_block_for_index_flops)->Arg(100);
