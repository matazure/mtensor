#pragma once

#include <malloc.h>
#include <matazure/config.hpp>

namespace matazure {

template <typename _Type, int_t _Alignment>
class aligned_allocator : public std::allocator<_Type> {
   public:
    aligned_allocator() {}
    aligned_allocator(const aligned_allocator& rhs) {}
    aligned_allocator& operator=(const aligned_allocator& rhs) { return *this; }

    _Type* allocate(size_t size) {
#ifdef __GNUC__
        _Type* data = reinterpret_cast<_Type*>(memalign(_Alignment, size * sizeof(_Type)));
#else
        _Type* data = reinterpret_cast<_Type*>(_aligned_malloc(size * sizeof(_Type), _Alignment));
#endif
        return data;
    }

    void deallocate(_Type* p, size_t size) {
#ifdef __GNUC__
        free(p);
#else
        _aligned_free(p);
#endif
    }

    ~aligned_allocator(){};
};

}  // namespace matazure