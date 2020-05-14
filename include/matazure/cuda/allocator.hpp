#pragma once

#include <matazure/cuda/exception.hpp>

namespace matazure {
namespace cuda {

template <typename _Type>
class allocator {
   public:
    MATAZURE_GENERAL allocator() {}
    MATAZURE_GENERAL allocator(const allocator& rhs) {}
    MATAZURE_GENERAL allocator& operator=(const allocator& rhs) {}

    _Type* allocate(size_t size) {
        _Type* data = nullptr;
        assert_runtime_success(cudaMalloc(&data, size * sizeof(_Type)));
        return data;
    }

    void deallocate(_Type* p, size_t size) { assert_runtime_success(cudaFree(p)); }

    MATAZURE_GENERAL ~allocator(){};
};

}  // namespace cuda

}  // namespace matazure