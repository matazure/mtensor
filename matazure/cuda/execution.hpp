#pragma once

#include <cuda_occupancy.h>
#include <map>
#include <mutex>

namespace matazure { namespace cuda {

class ExecutionPolicy {
public:
	enum ConfigurationState {
		Automatic = 0,
		SharedMem = 1,
		BlockSize = 2,
		GridSize = 4,
		FullManual = GridSize | BlockSize | SharedMem
	};

	ExecutionPolicy()
		: mState(Automatic),
		mGridSize(0),
		mBlockSize(0),
		mSharedMemBytes(0),
		mStream((cudaStream_t)0) {}

	ExecutionPolicy(int gridSize, int blockSize, size_t sharedMemBytes)
		: mState(0), mStream(0) {
		setGridSize(gridSize);
		setBlockSize(blockSize);
		setSharedMemBytes(sharedMemBytes);
	}

	ExecutionPolicy(int gridSize, int blockSize, size_t sharedMemBytes, cudaStream_t stream)
		: mState(0) {
		setGridSize(gridSize);
		setBlockSize(blockSize);
		setSharedMemBytes(sharedMemBytes);
		setStream(stream);
	}

	~ExecutionPolicy() {}

	int    getConfigState()    const { return mState; }

	int    getGridSize()       const { return mGridSize; }
	int    getBlockSize()      const { return mBlockSize; }
	int    getMaxBlockSize()   const { return mMaxBlockSize; }
	size_t getSharedMemBytes() const { return mSharedMemBytes; }
	cudaStream_t getStream()   const { return mStream; }

	void setGridSize(int arg) {
		mGridSize = arg;
		if (mGridSize > 0) mState |= GridSize;
		else mState &= (FullManual - GridSize);
	}
	void setBlockSize(int arg) {
		mBlockSize = arg;
		if (mBlockSize > 0) mState |= BlockSize;
		else mState &= (FullManual - BlockSize);
	}
	void setMaxBlockSize(int arg) {
		mMaxBlockSize = arg;
	}
	void setSharedMemBytes(size_t arg) {
		mSharedMemBytes = arg;
		mState |= SharedMem;
	}
	void setStream(cudaStream_t stream) {
		mStream = stream;
	}

private:
	int    mState;
	int    mGridSize;
	int    mBlockSize;
	int    mMaxBlockSize;
	size_t mSharedMemBytes;
	cudaStream_t mStream;
};


// Avoid calling cudaGetDeviceProperties() at every single kernel invocation -
// it will ruin performance and make the graphics card make weird, scary noises
class DevicePropertiesCache
{
public:
	// Return a reference to a cudaDeviceProp for the current device
	static cudaDeviceProp & get()
	{
		static DevicePropertiesCache instance;

		int devId;
		cudaError_t status = cudaGetDevice(&devId);
		if (status != cudaSuccess) throw status;

		std::lock_guard<std::mutex> guard(instance.mtx);

		if (instance.dpcache.find(devId) == instance.dpcache.end())
		{
			// cache miss
			instance.dpcache[devId] = cudaDeviceProp();
			status = cudaGetDeviceProperties(&instance.dpcache[devId], devId);
			if (status != cudaSuccess) throw status;
		}
		return instance.dpcache[devId];
	}

private:
	std::map<int, cudaDeviceProp> dpcache;
	std::mutex mtx;
};


inline
size_t availableSharedBytesPerBlock(size_t sharedMemPerMultiprocessor,
									size_t sharedSizeBytesStatic,
									int blocksPerSM,
									int smemAllocationUnit)
{
	size_t bytes = __occRoundUp(sharedMemPerMultiprocessor / blocksPerSM,
								smemAllocationUnit) - smemAllocationUnit;
	return bytes - sharedSizeBytesStatic;
}

template <typename KernelFunc>
cudaError_t condigure_grid(ExecutionPolicy &p, KernelFunc k)
{
	int configState = p.getConfigState();

	if (configState == ExecutionPolicy::FullManual) return cudaSuccess;

	cudaDeviceProp *props;
	try
	{
		props = &DevicePropertiesCache::get();
	}
	catch (cudaError_t status)
	{
		return status;
	}

	cudaFuncAttributes attribs;
	cudaOccDeviceProp occProp(*props);

	cudaError_t status = cudaFuncGetAttributes(&attribs, k);
	if (status != cudaSuccess) return status;
	cudaOccFuncAttributes occAttrib(attribs);

	cudaFuncCache cacheConfig;
	status = cudaDeviceGetCacheConfig(&cacheConfig);
	if (status != cudaSuccess) return status;
	cudaOccDeviceState occState;
	occState.cacheConfig = (cudaOccCacheConfig)cacheConfig;

	int numSMs = props->multiProcessorCount;

	if ((configState & ExecutionPolicy::BlockSize) == 0) {
		int bsize = 0, minGridSize = 0;
		cudaOccError occErr = cudaOccMaxPotentialOccupancyBlockSize(&minGridSize,
																	&bsize,
																	&occProp,
																	&occAttrib,
																	&occState,
																	p.getSharedMemBytes());
		if (occErr != CUDA_OCC_SUCCESS || bsize < 0) return cudaErrorInvalidConfiguration;
		p.setBlockSize(bsize);
	}

	if ((configState & ExecutionPolicy::GridSize) == 0) {
		cudaOccResult result;
		cudaOccError occErr = cudaOccMaxActiveBlocksPerMultiprocessor(&result,
																	  &occProp,
																	  &occAttrib,
																	  &occState,
																	  p.getBlockSize(),
																	  p.getSharedMemBytes());
		if (occErr != CUDA_OCC_SUCCESS) return cudaErrorInvalidConfiguration;
		p.setGridSize(result.activeBlocksPerMultiprocessor * numSMs);
		if (p.getGridSize() < numSMs) return cudaErrorInvalidConfiguration;
	}

	if ((configState & ExecutionPolicy::SharedMem) == 0) {

		int smemGranularity = 0;
		cudaOccError occErr = cudaOccSMemAllocationGranularity(&smemGranularity, &occProp);
		if (occErr != CUDA_OCC_SUCCESS) return cudaErrorInvalidConfiguration;
		size_t sbytes = availableSharedBytesPerBlock(props->sharedMemPerBlock,
													 attribs.sharedSizeBytes,
													 __occDivideRoundUp(p.getGridSize(), numSMs),
													 smemGranularity);
		p.setSharedMemBytes(sbytes);
	}

#ifdef HEMI_DEBUG
	printf("%d %d %ld\n", p.getBlockSize(), p.getGridSize(), p.getSharedMemBytes());
#endif

	return cudaSuccess;
}

} }
