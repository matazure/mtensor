#pragma once

#include <matazure/cuda/exception.hpp>

namespace matazure {
namespace cuda {

template <typename _Type>
class allocator : std::allocator<_Type> {
   public:
    _Type* allocate(size_t size) {
        _Type* data = nullptr;
        assert_runtime_success(cudaMalloc(&data, size * sizeof(_Type)));
        return data;
    }

    void deallocate(_Type* p, size_t size) { assert_runtime_success(cudaFree(p)); }
};

}  // namespace cuda

}  // namespace matazure