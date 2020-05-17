#include "../bm_config.hpp"

__global__ void raw1f_copy_kernel(float* p_src, float* p_dst, int_t size) {
    for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        p_dst[i] = p_src[i];
    }
}

void bm_cuda_raw1f_for_copy(benchmark::State& state) {
    int ts_size = state.range(0);
    cuda::tensor<float, 1> cts_src(ts_size);
    cuda::tensor<float, 1> cts_dst(ts_size);

    cuda::for_index_execution_policy policy;
    policy.total_size(cts_src.size());
    cuda::configure_grid(policy, raw1f_copy_kernel);

    while (state.KeepRunning()) {
        raw1f_copy_kernel<<<policy.grid_dim()[0], policy.block_dim()[0], policy.shared_mem_bytes(),
                            policy.stream()>>>(cts_src.data(), cts_dst.data(), cts_src.size());
        cudaDeviceSynchronize();

        benchmark::DoNotOptimize(cts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(cts_src.size()) *
                            sizeof(cts_src[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(cts_src.size()));
}

BENCHMARK(bm_cuda_raw1f_for_copy)->Arg(1_G);

template <typename tensor_type>
inline void bm_tensor_for_array_index_copy(benchmark::State& state) {
    int ts_size = state.range(0);
    constexpr int_t rank = tensor_type::rank;
    pointi<rank> shape;
    fill(shape, ts_size);

    tensor_type ts_src(shape);
    tensor_type ts_dst(shape);

    while (state.KeepRunning()) {
        cuda::for_index(shape, [ts_src, ts_dst] MATAZURE_GENERAL(pointi<rank> idx) {
            ts_dst(idx) = ts_src(idx);
        });
        // cuda::copy(ts_src, ts_dst);
        benchmark::DoNotOptimize(ts_dst.data());
    }

    state.SetBytesProcessed(state.iterations() * static_cast<size_t>(ts_src.size()) *
                            sizeof(ts_src[0]));
    state.SetItemsProcessed(state.iterations() * static_cast<size_t>(ts_src.size()));
}

auto bm_cuda_tensor1f_for_array_index_copy = bm_tensor_for_array_index_copy<cuda::tensor<float, 1>>;
auto bm_cuda_tensor2f_for_array_index_copy = bm_tensor_for_array_index_copy<cuda::tensor<float, 2>>;
BENCHMARK(bm_cuda_tensor1f_for_array_index_copy)->Arg(1_G);
// cuda中二维的坐标是会耗时更多， 大概有2%的损耗，说明编译器无法将数组访问形式的代码优化点
BENCHMARK(bm_cuda_tensor2f_for_array_index_copy)->Arg(10_K);
