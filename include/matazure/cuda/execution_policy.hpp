#pragma once

#include <cuda_occupancy.h>
#include <algorithm>
#include <map>
#include <matazure/cuda/exception.hpp>
#include <matazure/point.hpp>
#include <mutex>

namespace matazure {
namespace cuda {

namespace internal {

class device_properties_cache {
   public:
    static cudaDeviceProp& get() {
        static device_properties_cache instance;

        int dev_id;
        assert_runtime_success(cudaGetDevice(&dev_id));

        std::lock_guard<std::mutex> guard(instance.mtx_);

        if (instance.device_prop_cache_.find(dev_id) == instance.device_prop_cache_.end()) {
            instance.device_prop_cache_[dev_id] = cudaDeviceProp();
            assert_runtime_success(
                cudaGetDeviceProperties(&instance.device_prop_cache_[dev_id], dev_id));
        }
        return instance.device_prop_cache_[dev_id];
    }

   private:
    std::map<int, cudaDeviceProp> device_prop_cache_;
    std::mutex mtx_;
};

inline size_t availableSharedBytesPerBlock(size_t sharedMemPerMultiprocessor,
                                           size_t sharedSizeBytesStatic, int blocksPerSM,
                                           int smemAllocationUnit) {
    size_t bytes = __occRoundUp(sharedMemPerMultiprocessor / blocksPerSM, smemAllocationUnit) -
                   smemAllocationUnit;
    return bytes - sharedSizeBytesStatic;
}

}  // namespace internal

class execution_policy {
   public:
    execution_policy(
        pointi<3> grid_dim = {{0, 1, 1}}, pointi<3> block_dim = {{0, 1, 1}},
        size_t shared_mem_bytes = 0,
        std::shared_ptr<cudaStream_t> sp_stream = std::make_shared<cudaStream_t>(nullptr))
        : grid_dim_(grid_dim),
          block_dim_(block_dim),
          shared_mem_bytes_(shared_mem_bytes),
          sp_stream_(sp_stream) {
        if (*sp_stream_ == 0) {
            cudaStream_t stream;
            assert_runtime_success(cudaStreamCreate(&stream));
            sp_stream_.reset(new cudaStream_t(stream), [](cudaStream_t* p) {
                assert_runtime_success(cudaStreamSynchronize(*p));
                assert_runtime_success(cudaStreamDestroy(*p));
                delete p;
            });

            // TODO: has bug, refactor it
            // assert_runtime_success(cudaStreamCreate(&stream_));
        }
    }

    pointi<3> grid_dim() const { return grid_dim_; }
    pointi<3> block_dim() const { return block_dim_; }
    size_t shared_mem_bytes() const { return shared_mem_bytes_; }
    cudaStream_t stream() const { return *sp_stream_; }

    void grid_dim(pointi<3> arg) { grid_dim_ = arg; }
    void block_dim(pointi<3> arg) { block_dim_ = arg; }
    void shared_mem_bytes(size_t arg) { shared_mem_bytes_ = arg; }

    void synchronize() { assert_runtime_success(cudaStreamSynchronize(stream())); }

   protected:
    pointi<3> grid_dim_ = {{0, 1, 1}};
    pointi<3> block_dim_ = {{0, 1, 1}};
    // 0 represents not use dynamic shared memory
    size_t shared_mem_bytes_ = 0;
    std::shared_ptr<cudaStream_t> sp_stream_ = nullptr;
};

class default_execution_policy : public execution_policy {
   public:
   protected:
    pointi<3> grid_dim_ = {{0, 1, 1}};
    pointi<3> block_dim_ = {{0, 1, 1}};
    // 0 represents not use dynamic shared memory
    size_t shared_mem_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
};

template <typename _ExePolicy, typename _KernelFunc>
inline void configure_grid(_ExePolicy& exe_policy, _KernelFunc kernel) {
    /// Do none
}

template <typename __KernelFunc>
inline void configure_grid(default_execution_policy& exe_policy, __KernelFunc k) {
    cudaDeviceProp* props;
    props = &internal::device_properties_cache::get();

    cudaFuncAttributes attribs;
    cudaOccDeviceProp occProp(*props);

    assert_runtime_success(cudaFuncGetAttributes(&attribs, k));
    cudaOccFuncAttributes occAttrib(attribs);

    cudaFuncCache cacheConfig;
    assert_runtime_success(cudaDeviceGetCacheConfig(&cacheConfig));
    cudaOccDeviceState occState;
    occState.cacheConfig = (cudaOccCacheConfig)cacheConfig;

    int numSMs = props->multiProcessorCount;

    int bsize = 0, minGridSize = 0;
    verify_occupancy_success(cudaOccMaxPotentialOccupancyBlockSize(
        &minGridSize, &bsize, &occProp, &occAttrib, &occState, exe_policy.shared_mem_bytes()));
    exe_policy.block_dim({bsize, 1, 1});

    cudaOccResult result;
    verify_occupancy_success(cudaOccMaxActiveBlocksPerMultiprocessor(
        &result, &occProp, &occAttrib, &occState, exe_policy.block_dim()[0],
        exe_policy.shared_mem_bytes()));
    exe_policy.grid_dim({result.activeBlocksPerMultiprocessor * numSMs, 1, 1});

    int smemGranularity = 0;
    verify_occupancy_success(cudaOccSMemAllocationGranularity(&smemGranularity, &occProp));
    size_t sbytes = internal::availableSharedBytesPerBlock(
        props->sharedMemPerBlock, attribs.sharedSizeBytes,
        __occDivideRoundUp(exe_policy.grid_dim()[0], numSMs), smemGranularity);

    exe_policy.shared_mem_bytes(sbytes);
}

class for_index_execution_policy : public execution_policy {
   public:
    int_t total_size() const { return total_size_; }
    void total_size(int_t size) { total_size_ = size; }

   protected:
    int_t total_size_ = 0;
};

template <typename __KernelFunc>
inline void configure_grid(for_index_execution_policy& exe_policy, __KernelFunc k) {
    cudaDeviceProp* props;
    props = &internal::device_properties_cache::get();

    cudaFuncAttributes attribs;
    cudaOccDeviceProp occProp(*props);

    assert_runtime_success(cudaFuncGetAttributes(&attribs, k));
    cudaOccFuncAttributes occAttrib(attribs);

    cudaFuncCache cacheConfig;
    assert_runtime_success(cudaDeviceGetCacheConfig(&cacheConfig));
    cudaOccDeviceState occState;
    occState.cacheConfig = (cudaOccCacheConfig)cacheConfig;

    int numSMs = props->multiProcessorCount;

    int bsize = 0, minGridSize = 0;
    verify_occupancy_success(cudaOccMaxPotentialOccupancyBlockSize(
        &minGridSize, &bsize, &occProp, &occAttrib, &occState, exe_policy.shared_mem_bytes()));
    exe_policy.block_dim({bsize, 1, 1});

    cudaOccResult result;
    verify_occupancy_success(cudaOccMaxActiveBlocksPerMultiprocessor(
        &result, &occProp, &occAttrib, &occState, exe_policy.block_dim()[0],
        exe_policy.shared_mem_bytes()));
    exe_policy.grid_dim({result.activeBlocksPerMultiprocessor * numSMs, 1, 1});

    auto pre_block_size = exe_policy.block_dim()[0];
    auto tmp_block_size = __occDivideRoundUp(exe_policy.total_size(), exe_policy.grid_dim()[0]);
    tmp_block_size = __occRoundUp(tmp_block_size, 128);
    exe_policy.block_dim({std::min(tmp_block_size, pre_block_size), 1, 1});

    int smemGranularity = 0;
    verify_occupancy_success(cudaOccSMemAllocationGranularity(&smemGranularity, &occProp));
    size_t sbytes = internal::availableSharedBytesPerBlock(
        props->sharedMemPerBlock, attribs.sharedSizeBytes,
        __occDivideRoundUp(exe_policy.grid_dim()[0], numSMs), smemGranularity);

    exe_policy.shared_mem_bytes(sbytes);
}

}  // namespace cuda
}  // namespace matazure
