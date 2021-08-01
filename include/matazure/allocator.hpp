#pragma once

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

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
        _Type* data = nullptr;
        posix_memalign(reinterpret_cast<void**>(&data), _Alignment, size * sizeof(_Type));
        MATAZURE_ASSERT(data != nullptr, "Failed to alloc align memory");
        // _Type* data = reinterpret_cast<_Type*>(memalign(_Alignment, size * sizeof(_Type)));
#else
        posix_memalign(reinterpret_cast<void**>(&data), _Alignment, size * sizeof(_Type));
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
