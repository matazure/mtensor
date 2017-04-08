#pragma once

#include <matazure/cuda/exception.hpp>

namespace matazure {
namespace cuda {

inline void set_device(int device) {
	assert_runtime_success(cudaSetDevice(device));
}

} }
