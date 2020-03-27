#pragma once

#include <cuda_occupancy.h>
#include <cuda_runtime.h>
#include <matazure/config.hpp>

namespace matazure {
namespace cuda {

class runtime_error : public std::runtime_error {
   public:
    runtime_error(cudaError_t error_code)
        : std::runtime_error(cudaGetErrorString(error_code)), error_code_(error_code) {}

   private:
    cudaError_t error_code_;
};

inline void assert_runtime_success(cudaError_t result) {
    if (result != cudaSuccess) {
        throw runtime_error(result);
    }
}

class occupancy_error : public std::runtime_error {
   public:
    occupancy_error(cudaOccError error_code)
        : std::runtime_error("cuda occupancy error"), error_code_(error_code) {}

   private:
    cudaOccError error_code_;
};

inline void verify_occupancy_success(cudaOccError result) {
    if (result != CUDA_OCC_SUCCESS) {
        throw occupancy_error(result);
    }
}

}  // namespace cuda
}  // namespace matazure
