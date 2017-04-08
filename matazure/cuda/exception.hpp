#pragma once

#include <matazure/config.hpp>
#include <cuda_runtime.h>

namespace matazure {
namespace cuda {

class runtime_error : public std::runtime_error {
public:
	runtime_error(cudaError_t error_code) : std::runtime_error(cudaGetErrorString(error_code)), error_code_(error_code)
	{}

private:
	cudaError_t error_code_;
};

inline void assert_runtime_success(cudaError_t result) throw(runtime_error)
{
	if (result != cudaSuccess) {
		throw runtime_error(result);
	}
}

class occupancy_error : public std::runtime_error {
public:
	occupancy_error(cudaOccError error_code) : std::runtime_error("cuda occupancy error"), error_code_(error_code)
	{}

private:
	cudaOccError error_code_;
};

inline void assert_occupancy_success(cudaOccError result) throw(occupancy_error)
{
	if (result != cudaSuccess) {
		throw occupancy_error(result);
	}
}

}
}
