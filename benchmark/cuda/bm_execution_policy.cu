#include "../bm_config.hpp"

__global__ void kernel_1M_flops(float* p_src, float* p_dst, int_t size) {
    for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        auto tmp = p_src[i];
        for (int_t k = 0; k < 1000000; ++k) {
            tmp *= 1.01f;
        }
        p_dst[i] = tmp;
    }
}

void bm_cuda_for_index_execution_policy_flops(benchmark::State& state) {
    int ts_size = state.range(0);
    cuda::tensor<float, 1> ts_src(ts_size);
    cuda::tensor<float, 1> ts_dst(ts_size);

    cuda::for_index_execution_policy policy;
    policy.total_size(ts_src.size());
    cuda::configure_grid(policy, kernel_1M_flops);

    while (state.KeepRunning()) {
        kernel_1M_flops<<<policy.grid_dim()[0], policy.block_dim()[0], policy.shared_mem_bytes(),
                          policy.stream()>>>(ts_src.data(), ts_dst.data(), ts_src.size());
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) * 1000000);
}

BENCHMARK(bm_cuda_for_index_execution_policy_flops)->Arg(1_M);

void bm_cuda_default_exectution_policy_flops(benchmark::State& state) {
    int ts_size = state.range(0);
    cuda::tensor<float, 1> ts_src(ts_size);
    cuda::tensor<float, 1> ts_dst(ts_size);

    cuda::default_execution_policy policy;
    cuda::configure_grid(policy, kernel_1M_flops);

    while (state.KeepRunning()) {
        kernel_1M_flops<<<policy.grid_dim()[0], policy.block_dim()[0], policy.shared_mem_bytes(),
                          policy.stream()>>>(ts_src.data(), ts_dst.data(), ts_src.size());
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) * 1000000);
}

BENCHMARK(bm_cuda_default_exectution_policy_flops)->Arg(1_M);

void bm_cuda_exectution_policy_flops(benchmark::State& state) {
    int ts_size = state.range(0);
    cuda::tensor<float, 1> ts_src(ts_size);
    cuda::tensor<float, 1> ts_dst(ts_size);

    cuda::execution_policy policy;
    policy.block_dim({1024, 1, 1});
    policy.grid_dim({34, 1, 1});
    cuda::configure_grid(policy, kernel_1M_flops);

    while (state.KeepRunning()) {
        kernel_1M_flops<<<policy.grid_dim()[0], policy.block_dim()[0], policy.shared_mem_bytes(),
                          policy.stream()>>>(ts_src.data(), ts_dst.data(), ts_src.size());
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) * 1000000);
}

BENCHMARK(bm_cuda_exectution_policy_flops)->Arg(1_M);

void bm_cuda_exectution_policy_mul_stream_flops(benchmark::State& state) {
    int ts_size = state.range(0);
    cuda::tensor<float, 1> ts_src(ts_size);
    cuda::tensor<float, 1> ts_dst(ts_size);

    cuda::execution_policy policy;
    policy.block_dim({1024, 1, 1});
    policy.grid_dim({34, 1, 1});
    cuda::configure_grid(policy, kernel_1M_flops);

    cuda::execution_policy policy1;
    policy1.block_dim({1024, 1, 1});
    policy1.grid_dim({34, 1, 1});
    cuda::configure_grid(policy1, kernel_1M_flops);

    cuda::execution_policy policy2;
    policy2.block_dim({1024, 1, 1});
    policy2.grid_dim({34, 1, 1});
    cuda::configure_grid(policy2, kernel_1M_flops);

    while (state.KeepRunning()) {
        kernel_1M_flops<<<policy1.grid_dim()[0], policy1.block_dim()[0], policy1.shared_mem_bytes(),
                          policy1.stream()>>>(ts_src.data(), ts_dst.data(), ts_src.size() / 2);

        kernel_1M_flops<<<policy2.grid_dim()[0], policy2.block_dim()[0], policy2.shared_mem_bytes(),
                          policy2.stream()>>>(ts_src.data() + ts_size / 2,
                                              ts_dst.data() + ts_size / 2, ts_size / 2);
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) * 1000000);
}

BENCHMARK(bm_cuda_exectution_policy_mul_stream_flops)->Arg(1_M);