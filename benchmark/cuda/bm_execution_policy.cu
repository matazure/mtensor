#include "../bm_config.hpp"

__global__ void kernel_freq(float* p_src, float* p_dst, int_t size) {
    for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        auto tmp = p_dst[i];
        for (int_t k = 0; k < 1000000; ++k) {
            tmp *= 1.01f;
        }
        p_src[i] = tmp;
    }
}

void bm_cuda_lauch_kernel_freq(benchmark::State& state) {
    int ts_size = state.range(0);
    cuda::tensor<float, 1> ts_src(ts_size);
    cuda::tensor<float, 1> ts_dst(ts_size);

    cuda::parallel_execution_policy policy;
    policy.total_size(ts_src.size());
    cuda::configure_grid(policy, kernel_freq);

    while (state.KeepRunning()) {
        kernel_freq<<<policy.grid_dim(), policy.block_dim(), policy.shared_mem_bytes(),
                      policy.stream()>>>(ts_src.data(), ts_dst.data(), ts_src.size());
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) * 1000000);
}

BENCHMARK(bm_cuda_lauch_kernel_freq)->Arg(1_M);
