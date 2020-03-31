#pragma once

#include <cuda_occupancy.h>
#include <algorithm>
#include <map>
#include <matazure/cuda/exception.hpp>
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
    int grid_size() const { return grid_size_; }
    void grid_size(int arg) { grid_size_ = arg; }

    int block_dim() const { return block_size_; }
    void block_dim(int arg) { block_size_ = arg; }

    size_t shared_mem_bytes() const { return shared_mem_bytes_; }
    void shared_mem_bytes(size_t arg) { shared_mem_bytes_ = arg; }

    cudaStream_t stream() const { return stream_; }
    void stream(cudaStream_t stream) { stream_ = stream; }

    ~execution_policy() { cudaStreamSynchronize(stream_); }

   protected:
    int grid_size_ = 0;
    int block_size_ = 0;
    size_t shared_mem_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
};

template <typename _ExePolicy, typename _KernelFunc>
inline void configure_grid(_ExePolicy& exe_policy, _KernelFunc kernel) {
    /// Do none
}

template <typename __KernelFunc>
inline void configure_grid(execution_policy& exe_policy, __KernelFunc k) {
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
    exe_policy.block_dim(bsize);

    cudaOccResult result;
    verify_occupancy_success(cudaOccMaxActiveBlocksPerMultiprocessor(
        &result, &occProp, &occAttrib, &occState, exe_policy.block_dim(),
        exe_policy.shared_mem_bytes()));
    exe_policy.grid_size(result.activeBlocksPerMultiprocessor * numSMs);

    int smemGranularity = 0;
    verify_occupancy_success(cudaOccSMemAllocationGranularity(&smemGranularity, &occProp));
    size_t sbytes = internal::availableSharedBytesPerBlock(
        props->sharedMemPerBlock, attribs.sharedSizeBytes,
        __occDivideRoundUp(exe_policy.grid_size(), numSMs), smemGranularity);

    exe_policy.shared_mem_bytes(sbytes);
}

class parallel_execution_policy : public execution_policy {
   public:
    int_t total_size() const { return total_size_; }
    void total_size(int_t size) { total_size_ = size; }

   protected:
    int_t total_size_;
};

template <typename __KernelFunc>
inline void configure_grid(parallel_execution_policy& exe_policy, __KernelFunc k) {
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
    exe_policy.block_dim(bsize);

    cudaOccResult result;
    verify_occupancy_success(cudaOccMaxActiveBlocksPerMultiprocessor(
        &result, &occProp, &occAttrib, &occState, exe_policy.block_dim(),
        exe_policy.shared_mem_bytes()));
    exe_policy.grid_size(result.activeBlocksPerMultiprocessor * numSMs);

    auto pre_block_size = exe_policy.block_dim();
    auto tmp_block_size = __occDivideRoundUp(exe_policy.total_size(), exe_policy.grid_size());
    tmp_block_size = __occRoundUp(tmp_block_size, 128);
    exe_policy.block_dim(std::min(tmp_block_size, pre_block_size));

    int smemGranularity = 0;
    verify_occupancy_success(cudaOccSMemAllocationGranularity(&smemGranularity, &occProp));
    size_t sbytes = internal::availableSharedBytesPerBlock(
        props->sharedMemPerBlock, attribs.sharedSizeBytes,
        __occDivideRoundUp(exe_policy.grid_size(), numSMs), smemGranularity);

    exe_policy.shared_mem_bytes(sbytes);
}

}  // namespace cuda
}  // namespace matazure
