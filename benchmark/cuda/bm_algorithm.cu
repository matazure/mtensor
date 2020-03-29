
#include "../bm_config.hpp"

#include <chrono>

void bm_cuda_mem_copy(benchmark::State& state) {
    int ts_size = state.range(0);
    cuda::tensor<float, 1> cts_src(ts_size);
    cuda::tensor<float, 1> cts_dst(ts_size);

    while (state.KeepRunning()) {
        cudaMemcpy(cts_dst.data(), cts_src.data(), sizeof(cts_src[0]) * cts_src.size(),
                   cudaMemcpyDefault);
        cudaDeviceSynchronize();
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(cts_src.size()) *
                            sizeof(float));
}

__global__ void raw1f_copy_kernel(float* p_src, float* p_dst, int_t size) {
    for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        float tmp = p_dst[i];
        // for (int i = 0; i < 1000; ++i) {
        //     tmp *= 1.01f;
        // }

        p_src[i] = tmp;
    }
}

void bm_cuda_raw1f_copy(benchmark::State& state) {
    int ts_size = state.range(0);
    cuda::tensor<float, 1> cts_src(ts_size);
    cuda::tensor<float, 1> cts_dst(ts_size);

    cuda::parallel_execution_policy policy;
    policy.total_size(cts_src.size());
    cuda::configure_grid(policy, raw1f_copy_kernel);

    while (state.KeepRunning()) {
        raw1f_copy_kernel<<<policy.grid_size(), policy.block_dim(), policy.shared_mem_bytes(),
                            policy.stream()>>>(cts_src.data(), cts_dst.data(), cts_src.size());
        cudaDeviceSynchronize();
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(cts_src.size()) *
                            sizeof(cts_src[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(cts_src.size()));
}

BENCHMARK(bm_cuda_mem_copy)->Arg(1_G)->UseRealTime();
BENCHMARK(bm_cuda_raw1f_copy)->Arg(1_G)->UseRealTime();
// BENCHMARK(bm_la_tensor2f_copy)->Arg(10000);
// BENCHMARK(bm_aa_tensor2f_copy)->Arg(10000);

// BENCHMARK

BENCHMARK_MAIN();
