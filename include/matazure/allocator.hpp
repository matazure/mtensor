#pragma once

#include <stdlib.h>
#include <matazure/config.hpp>

namespace matazure {

template <typename _Type, int_t _Alignment>
class aligned_allocator : public std::allocator<_Type> {
   public:
    aligned_allocator() {}
    aligned_allocator(const aligned_allocator& rhs) {}
    aligned_allocator& operator=(const aligned_allocator& rhs) { return *this; }

    _Type* allocate(size_t size) {
        _Type* data = nullptr;
#ifdef __WIN32
        data = reinterpret_cast<_Type*>(_aligned_malloc(size * sizeof(_Type), _Alignment));
#else
        posix_memalign(reinterpret_cast<void **>(&data), _Alignment, size * sizeof(_Type));
#endif
        return data;
    }

    void deallocate(_Type* p, size_t size) {
#ifdef __WIN32
        _aligned_free(p);
#else
        free(p);
#endif
    }

    ~aligned_allocator(){};
};

}  // namespace matazure
