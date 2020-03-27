#include <matazure/bm_config.hpp>

#ifdef USE_CUDA

template <typename _ValueType>
__global__ void gold_tensor_rank1_fill_kernel(_ValueType* p_dst, int_t count, _ValueType v) {
    for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
        p_dst[i] = v;
    }
}

template <typename _ValueType>
void bm_gold_cu_tensor_rank1_fill(benchmark::State& state) {
    cuda::tensor<_ValueType, 1> ts_src(state.range(0));

    while (state.KeepRunning()) {
        cuda::parallel_execution_policy policy;
        policy.total_size(ts_src.size());
        cuda::configure_grid(policy, gold_tensor_rank1_fill_kernel<_ValueType>);
        gold_tensor_rank1_fill_kernel<<<policy.grid_size(), policy.block_dim(),
                                        policy.shared_mem_bytes(), policy.stream()>>>(
            ts_src.data(), ts_src.size(), zero<_ValueType>::value());
        cuda::device_synchronize();

        benchmark::ClobberMemory();
    }

    auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
    state.SetBytesProcessed(state.iterations() * bytes_size);
}

#define BM_GOLD_CU_TENSOR_RANK1_FILL(ValueType)                                                \
    auto bm_gold_cu_tensor_##ValueType##_rank1_fill = bm_gold_cu_tensor_rank1_fill<ValueType>; \
    BENCHMARK(bm_gold_cu_tensor_##ValueType##_rank1_fill)                                      \
        ->RangeMultiplier(bm_config::range_multiplier<ValueType, 1, device_tag>())             \
        ->Range(bm_config::min_shape<ValueType, 1, device_tag>(),                              \
                bm_config::max_shape<ValueType, 1, device_tag>())                              \
        ->UseRealTime();

BM_GOLD_CU_TENSOR_RANK1_FILL(byte)
BM_GOLD_CU_TENSOR_RANK1_FILL(int16_t)
BM_GOLD_CU_TENSOR_RANK1_FILL(int32_t)
BM_GOLD_CU_TENSOR_RANK1_FILL(int64_t)
BM_GOLD_CU_TENSOR_RANK1_FILL(float)
BM_GOLD_CU_TENSOR_RANK1_FILL(double)
BM_GOLD_CU_TENSOR_RANK1_FILL(point3f)
BM_GOLD_CU_TENSOR_RANK1_FILL(point4f)
BM_GOLD_CU_TENSOR_RANK1_FILL(hete_float32x4_t)

template <typename _ValueType>
__global__ void gold_tensor_rank1_copy_kernel(_ValueType* p_src, _ValueType* p_dst, int_t count) {
    for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
        p_dst[i] = p_src[i];
    }
}

template <typename _ValueType>
void bm_gold_cu_copy_rank1_fill(benchmark::State& state) {
    cuda::tensor<_ValueType, 1> ts_src(state.range(0));
    cuda::tensor<_ValueType, 1> ts_dst(ts_src.size());
    fill(ts_src, zero<_ValueType>::value());

    while (state.KeepRunning()) {
        cuda::parallel_execution_policy policy;
        policy.total_size(ts_src.size());
        cuda::configure_grid(policy, gold_tensor_rank1_copy_kernel<_ValueType>);
        gold_tensor_rank1_copy_kernel<<<policy.grid_size(), policy.block_dim(),
                                        policy.shared_mem_bytes(), policy.stream()>>>(
            ts_src.data(), ts_dst.data(), ts_src.size());

        cuda::device_synchronize();

        benchmark::ClobberMemory();
    }

    auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
    state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}

#define BM_GOLD_CU_TENSOR_RANK1_COPY(ValueType)                                              \
    auto bm_gold_cu_tensor_##ValueType##_rank1_copy = bm_gold_cu_copy_rank1_fill<ValueType>; \
    BENCHMARK(bm_gold_cu_tensor_##ValueType##_rank1_copy)                                    \
        ->RangeMultiplier(bm_config::range_multiplier<ValueType, 1, device_tag>())           \
        ->Range(bm_config::min_shape<ValueType, 1, device_tag>(),                            \
                bm_config::max_shape<ValueType, 1, device_tag>())                            \
        ->UseRealTime();

BM_GOLD_CU_TENSOR_RANK1_COPY(byte)
BM_GOLD_CU_TENSOR_RANK1_COPY(int16_t)
BM_GOLD_CU_TENSOR_RANK1_COPY(int32_t)
BM_GOLD_CU_TENSOR_RANK1_COPY(int64_t)
BM_GOLD_CU_TENSOR_RANK1_COPY(float)
BM_GOLD_CU_TENSOR_RANK1_COPY(double)
BM_GOLD_CU_TENSOR_RANK1_COPY(point3f)
BM_GOLD_CU_TENSOR_RANK1_COPY(point4f)
BM_GOLD_CU_TENSOR_RANK1_COPY(hete_float32x4_t)

#endif

#ifdef USE_HOST

template <typename _ValueType>
void bm_gold_host_fill_rank1(benchmark::State& state) {
    tensor<_ValueType, 1> ts_src(state.range(0));
    auto p_data = ts_src.data();
    auto size = ts_src.size();
    fill(ts_src, zero<_ValueType>::value());

    while (state.KeepRunning()) {
        for (int_t i = 0; i < size; ++i) {
            p_data[i] = zero<_ValueType>::value();
        }

        benchmark::ClobberMemory();
    }

    auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
    state.SetBytesProcessed(state.iterations() * bytes_size);
}

#define BM_GOLD_HOST_TENSOR_RANK1_FILL(ValueType)                                           \
    auto bm_gold_host_tensor_##ValueType##_rank1_fill = bm_gold_host_fill_rank1<ValueType>; \
    BENCHMARK(bm_gold_host_tensor_##ValueType##_rank1_fill)                                 \
        ->RangeMultiplier(bm_config::range_multiplier<ValueType, 1, device_tag>())          \
        ->Range(bm_config::min_shape<ValueType, 1, device_tag>(),                           \
                bm_config::max_shape<ValueType, 1, device_tag>())                           \
        ->UseRealTime();

BM_GOLD_HOST_TENSOR_RANK1_FILL(byte)
BM_GOLD_HOST_TENSOR_RANK1_FILL(int16_t)
BM_GOLD_HOST_TENSOR_RANK1_FILL(int32_t)
BM_GOLD_HOST_TENSOR_RANK1_FILL(int64_t)
BM_GOLD_HOST_TENSOR_RANK1_FILL(float)
BM_GOLD_HOST_TENSOR_RANK1_FILL(double)
BM_GOLD_HOST_TENSOR_RANK1_FILL(point3f)
BM_GOLD_HOST_TENSOR_RANK1_FILL(point4f)
BM_GOLD_HOST_TENSOR_RANK1_FILL(hete_float32x4_t)

template <typename _ValueType>
void bm_gold_host_copy_rank1(benchmark::State& state) {
    tensor<_ValueType, 1> ts_src(state.range(0));
    tensor<_ValueType, 1> ts_dst(ts_src.size());
    fill(ts_src, zero<_ValueType>::value());
    auto p_src = ts_src.data();
    auto p_dst = ts_dst.data();
    auto size = ts_src.size();

    while (state.KeepRunning()) {
        for (int_t i = 0; i < size; ++i) {
            p_dst[i] = p_src[i];
        }

        benchmark::ClobberMemory();
    }

    auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(_ValueType);
    state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}

#define BM_GOLD_HOST_TENSOR_RANK1_COPY(ValueType)                                           \
    auto bm_gold_host_tensor_##ValueType##_rank1_copy = bm_gold_host_copy_rank1<ValueType>; \
    BENCHMARK(bm_gold_host_tensor_##ValueType##_rank1_copy)                                 \
        ->RangeMultiplier(bm_config::range_multiplier<ValueType, 1, device_tag>())          \
        ->Range(bm_config::min_shape<ValueType, 1, device_tag>(),                           \
                bm_config::max_shape<ValueType, 1, device_tag>())                           \
        ->UseRealTime();

BM_GOLD_HOST_TENSOR_RANK1_COPY(byte)
BM_GOLD_HOST_TENSOR_RANK1_COPY(int16_t)
BM_GOLD_HOST_TENSOR_RANK1_COPY(int32_t)
BM_GOLD_HOST_TENSOR_RANK1_COPY(int64_t)
BM_GOLD_HOST_TENSOR_RANK1_COPY(float)
BM_GOLD_HOST_TENSOR_RANK1_COPY(double)
BM_GOLD_HOST_TENSOR_RANK1_COPY(point3f)
BM_GOLD_HOST_TENSOR_RANK1_COPY(point4f)
BM_GOLD_HOST_TENSOR_RANK1_COPY(hete_float32x4_t)

#endif

template <typename _Tensor>
void bm_hete_tensor_fill(benchmark::State& state) {
    _Tensor ts_src(pointi<_Tensor::rank>::all(state.range(0)));

    while (state.KeepRunning()) {
        fill(ts_src, zero<typename _Tensor::value_type>::value());
        HETE_SYNCHRONIZE;

        benchmark::ClobberMemory();
    }

    auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
    state.SetBytesProcessed(state.iterations() * bytes_size);
}

#define BM_HETE_TENSOR_FILL(ValueType, Rank)                                        \
    auto bm_hete_tensor_##ValueType##_rank##Rank##_fill =                           \
        bm_hete_tensor_fill<HETE_TENSOR<ValueType, Rank>>;                          \
    BENCHMARK(bm_hete_tensor_##ValueType##_rank##Rank##_fill)                       \
        ->RangeMultiplier(bm_config::range_multiplier<ValueType, Rank, HETE_TAG>()) \
        ->Range(bm_config::min_shape<ValueType, Rank, HETE_TAG>(),                  \
                bm_config::max_shape<ValueType, Rank, HETE_TAG>())                  \
        ->UseRealTime();

#define BM_HETE_TENSOR_FILL_RANK1234(ValueType) \
    BM_HETE_TENSOR_FILL(ValueType, 1)           \
    BM_HETE_TENSOR_FILL(ValueType, 2)           \
    BM_HETE_TENSOR_FILL(ValueType, 3)           \
    BM_HETE_TENSOR_FILL(ValueType, 4)

BM_HETE_TENSOR_FILL_RANK1234(byte)
BM_HETE_TENSOR_FILL_RANK1234(int16_t)
BM_HETE_TENSOR_FILL_RANK1234(int32_t)
BM_HETE_TENSOR_FILL_RANK1234(int64_t)
BM_HETE_TENSOR_FILL_RANK1234(float)
BM_HETE_TENSOR_FILL_RANK1234(double)
BM_HETE_TENSOR_FILL_RANK1234(point3f)
BM_HETE_TENSOR_FILL_RANK1234(point4f)
BM_HETE_TENSOR_FILL_RANK1234(hete_float32x4_t)

template <typename _Tensor>
void bm_hete_tensor_copy(benchmark::State& state) {
    _Tensor ts_src(pointi<_Tensor::rank>::all(state.range(0)));
    _Tensor ts_dst(ts_src.shape());
    fill(ts_src, zero<typename _Tensor::value_type>::value());

    while (state.KeepRunning()) {
        copy(ts_src, ts_dst);
        HETE_SYNCHRONIZE;

        benchmark::ClobberMemory();
    }

    auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
    state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}

#define BM_HETE_TENSOR_COPY(ValueType, Rank)                                        \
    auto bm_hete_tensor_##ValueType##_rank##Rank##_copy =                           \
        bm_hete_tensor_copy<HETE_TENSOR<ValueType, Rank>>;                          \
    BENCHMARK(bm_hete_tensor_##ValueType##_rank##Rank##_copy)                       \
        ->RangeMultiplier(bm_config::range_multiplier<ValueType, Rank, HETE_TAG>()) \
        ->Range(bm_config::min_shape<ValueType, Rank, HETE_TAG>(),                  \
                bm_config::max_shape<ValueType, Rank, HETE_TAG>())                  \
        ->UseRealTime();

#define BM_HETE_TENSOR_RANK1234_COPY(ValueType) \
    BM_HETE_TENSOR_COPY(ValueType, 1)           \
    BM_HETE_TENSOR_COPY(ValueType, 2)           \
    BM_HETE_TENSOR_COPY(ValueType, 3)           \
    BM_HETE_TENSOR_COPY(ValueType, 4)

BM_HETE_TENSOR_RANK1234_COPY(byte)
BM_HETE_TENSOR_RANK1234_COPY(int16_t)
BM_HETE_TENSOR_RANK1234_COPY(int32_t)
BM_HETE_TENSOR_RANK1234_COPY(int64_t)
BM_HETE_TENSOR_RANK1234_COPY(float)
BM_HETE_TENSOR_RANK1234_COPY(double)
BM_HETE_TENSOR_RANK1234_COPY(point3f)
BM_HETE_TENSOR_RANK1234_COPY(point4f)
BM_HETE_TENSOR_RANK1234_COPY(hete_float32x4_t)

template <typename _Tensor>
void bm_hete_tensor_transform(benchmark::State& state) {
    _Tensor ts_src(pointi<_Tensor::rank>::all(state.range(0)));
    _Tensor ts_dst(ts_src.shape());
    fill(ts_src, zero<typename _Tensor::value_type>::value());

    while (state.KeepRunning()) {
        transform(ts_src, ts_dst,
                  [] __matazure__(const typename _Tensor::value_type& e) { return e + e; });
        HETE_SYNCHRONIZE;

        benchmark::ClobberMemory();
    }

    auto bytes_size = static_cast<size_t>(ts_src.size()) * sizeof(decltype(ts_src[0]));
    state.SetBytesProcessed(state.iterations() * bytes_size * 2);
}

#define BM_HETE_TENSOR_TRANSFORM(ValueType, Rank)                                   \
    auto bm_hete_tensor_##ValueType##_rank##Rank##_transform =                      \
        bm_hete_tensor_transform<HETE_TENSOR<ValueType, Rank>>;                     \
    BENCHMARK(bm_hete_tensor_##ValueType##_rank##Rank##_transform)                  \
        ->RangeMultiplier(bm_config::range_multiplier<ValueType, Rank, HETE_TAG>()) \
        ->Range(bm_config::min_shape<ValueType, Rank, HETE_TAG>(),                  \
                bm_config::max_shape<ValueType, Rank, HETE_TAG>())                  \
        ->UseRealTime();

#define BM_HETE_TENSOR_RANK1234_TRANSFORM(ValueType) \
    BM_HETE_TENSOR_TRANSFORM(ValueType, 1)           \
    BM_HETE_TENSOR_TRANSFORM(ValueType, 2)           \
    BM_HETE_TENSOR_TRANSFORM(ValueType, 3)           \
    BM_HETE_TENSOR_TRANSFORM(ValueType, 4)

BM_HETE_TENSOR_RANK1234_TRANSFORM(byte)
BM_HETE_TENSOR_RANK1234_TRANSFORM(int16_t)
BM_HETE_TENSOR_RANK1234_TRANSFORM(int32_t)
BM_HETE_TENSOR_RANK1234_TRANSFORM(int64_t)
BM_HETE_TENSOR_RANK1234_TRANSFORM(float)
BM_HETE_TENSOR_RANK1234_TRANSFORM(double)
BM_HETE_TENSOR_RANK1234_TRANSFORM(point3f)
BM_HETE_TENSOR_RANK1234_TRANSFORM(point4f)
BM_HETE_TENSOR_RANK1234_TRANSFORM(hete_float32x4_t)
