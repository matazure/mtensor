#pragma once

#include <matazure/cuda/exception.hpp>

namespace matazure {
namespace cuda {

inline void set_device(int device) { assert_runtime_success(cudaSetDevice(device)); }

inline int get_device() {
    int id = 0;
    assert_runtime_success(cudaGetDevice(&id));
    return id;
}

}  // namespace cuda
}  // namespace matazure
