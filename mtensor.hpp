#pragma once

/*****************************************************************************
MIT License

Copyright (c) 2017 Zhang Zhimin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*****************************************************************************/

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

// for cuda
#if defined(__CUDACC__) && !defined(MATAZURE_DISABLE_CUDA)
#ifdef __clang__
#if __clang_major__ < 9
#error clang minimum version is 9 for cuda
#endif
#else
#if __CUDACC_VER_MAJOR__ < 9
#error CUDA minimum version is 10.0
#endif
#endif

#define MATAZURE_CUDA
#endif

#ifdef MATAZURE_CUDA
#define MATAZURE_GENERAL __host__ __device__
#ifndef __clang__
#define MATAZURE_NV_EXE_CHECK_DISABLE #pragma nv_exec_check_disable
#else
#define MATAZURE_NV_EXE_CHECK_DISABLE
#endif
#else
#define MATAZURE_GENERAL
#define MATAZURE_NV_EXE_CHECK_DISABLE
#endif

#define __general__ MATAZURE_GENERAL

#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900)
#else
#error "use c++11 at least"
#endif

namespace matazure {

class assert_failed : public std::exception {
   public:
    assert_failed(const std::string& expr, const std::string& file, size_t line,
                  const std::string& msg = " ")
        : _expr(expr), _file(file), _line(line), _msg(msg) {
        _what_str = _expr + ", " + _file + ", " + std::to_string(_line) + ", " + _msg;
    }

    virtual const char* what() const noexcept override { return _what_str.c_str(); }

   private:
    std::string _expr;
    std::string _file;
    size_t _line;
    std::string _msg;
    std::string _what_str;
};

inline void raise_assert_failed(const std::string& expr, const std::string& file, long line,
                                const std::string& msg = " ") {
    throw assert_failed(expr, file, line, msg);
}

inline void raise_verify_failed(const std::string& expr, const std::string& file, long line,
                                const std::string& msg = " ") {
    throw assert_failed(expr, file, line, msg);
}

}  // namespace matazure

#if defined(MATAZURE_DISABLE_ASSERTS)
#define MATAZURE_ASSERT(expr, msg) ((void)0)
#else
#define MATAZURE_ASSERT(expr, ...) \
    ((!!(expr)) ? ((void)0)        \
                : ::matazure::raise_assert_failed(#expr, __FILE__, __LINE__, ##__VA_ARGS__))
#endif

#define MATAZURE_VERIFY(expr, ...) \
    ((!!(expr)) ? ((void)0)        \
                : ::matazure::raise_verify_failed(#expr, __FILE__, __LINE__, ##__VA_ARGS__))

#ifdef MATAZURE_CUDA
#include "cuda_occupancy.h"
#include "cuda_runtime.h"
#endif

namespace matazure {

typedef int int_t;

enum struct runtime { host, cuda };

template <typename _ValueType, int_t _Rank>
class point {
   public:
    static const int_t rank = _Rank;
    typedef _ValueType value_type;
    typedef value_type& reference;
    typedef const value_type& const_reference;

    MATAZURE_GENERAL constexpr const_reference operator[](int_t i) const { return elements_[i]; }

    MATAZURE_GENERAL reference operator[](int_t i) { return elements_[i]; }

    MATAZURE_GENERAL constexpr int_t size() const { return rank; }

    MATAZURE_GENERAL static point all(value_type v) {
        point re{};
        for (int_t i = 0; i < re.size(); ++i) {
            re[i] = v;
        }
        return re;
    }

   public:
    value_type elements_[rank];
};

// binary opertor
#define MATAZURE_POINT_BINARY_OPERATOR(op)                                \
    template <typename _T, int_t _Rank>                                   \
    inline MATAZURE_GENERAL auto operator op(const point<_T, _Rank>& lhs, \
                                             const point<_T, _Rank>& rhs) \
        ->point<decltype(lhs[0] op rhs[0]), _Rank> {                      \
        point<decltype(lhs[0] op rhs[0]), _Rank> re;                      \
        for (int_t i = 0; i < _Rank; ++i) {                               \
            re[i] = lhs[i] op rhs[i];                                     \
        }                                                                 \
        return re;                                                        \
    }

// assignment operators
#define MATAZURE_POINT_ASSIGNMENT_OPERATOR(op)                                                   \
    template <typename _T, int_t _Rank>                                                          \
    inline MATAZURE_GENERAL auto operator op(point<_T, _Rank>& lhs, const point<_T, _Rank>& rhs) \
        ->point<_T, _Rank> {                                                                     \
        for (int_t i = 0; i < _Rank; ++i) {                                                      \
            lhs[i] op rhs[i];                                                                    \
        }                                                                                        \
        return lhs;                                                                              \
    }

// Arithmetic
MATAZURE_POINT_BINARY_OPERATOR(+)
MATAZURE_POINT_BINARY_OPERATOR(-)
MATAZURE_POINT_BINARY_OPERATOR(*)
MATAZURE_POINT_BINARY_OPERATOR(/)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(+=)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(-=)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(*=)
MATAZURE_POINT_ASSIGNMENT_OPERATOR(/=)

template <typename _T, int_t _Rank>
inline MATAZURE_GENERAL point<_T, _Rank> operator+(const point<_T, _Rank>& p) {
    return p;
}

template <typename _T, int_t _Rank>
inline MATAZURE_GENERAL point<_T, _Rank> operator-(const point<_T, _Rank>& p) {
    point<_T, _Rank> temp;
    for (int_t i = 0; i < _Rank; ++i) {
        temp[i] = -p[i];
    }

    return temp;
}

template <int_t _Rank>
using pointi = point<int_t, _Rank>;

template <int_t _Rank>
class row_major_layout {
   public:
    const static int_t rank = _Rank;

    MATAZURE_GENERAL row_major_layout() : row_major_layout(pointi<rank>{0}){};

    MATAZURE_GENERAL row_major_layout(const pointi<rank>& shape) : shape_(shape) {
        stride_[rank - 1] = 1;
        for (int_t i = rank - 2; i >= 0; --i) {
            stride_[i] = shape[i + 1] * stride_[i + 1];
        }
        size_ = stride_[0] * shape[0];
    }

    MATAZURE_GENERAL row_major_layout(const row_major_layout& rhs)
        : row_major_layout(rhs.shape()) {}

    MATAZURE_GENERAL row_major_layout& operator=(const row_major_layout& rhs) {
        shape_ = rhs.shape();
        stride_ = rhs.stride();
        return *this;
    }

    MATAZURE_GENERAL int_t index2offset(const pointi<rank>& id) const {
        typename pointi<rank>::value_type offset = 0;
        for (int_t i = rank - 1; i >= 0; --i) {
            offset += id[i] * stride_[i];
        }
        return offset;
    };

    MATAZURE_GENERAL pointi<rank> offset2index(int_t offset) const {
        pointi<rank> id;
        for (int_t i = 0; i < rank; ++i) {
            id[i] = offset / stride_[i];
            offset = offset % stride_[i];
        }
        return id;
    }

    MATAZURE_GENERAL int_t size() const { return size_; }

    MATAZURE_GENERAL pointi<rank> shape() const { return shape_; }

    MATAZURE_GENERAL pointi<rank> stride() const { return stride_; }

    MATAZURE_GENERAL ~row_major_layout() {}

   private:
    pointi<rank> shape_;
    pointi<rank> stride_;
    int_t size_;
};

}  // namespace matazure

#ifdef MATAZURE_CUDA
namespace matazure::cuda {
class runtime_error : public std::runtime_error {
   public:
    runtime_error(cudaError_t error_code)
        : std::runtime_error(cudaGetErrorString(error_code)), error_code_(error_code) {}

   private:
    cudaError_t error_code_;
};

inline void verify_runtime_success(cudaError_t result) {
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

namespace internal {

class device_properties_cache {
   public:
    static cudaDeviceProp& get() {
        static device_properties_cache instance;

        int dev_id;
        verify_runtime_success(cudaGetDevice(&dev_id));

        std::lock_guard<std::mutex> guard(instance.mtx_);

        if (instance.device_prop_cache_.find(dev_id) == instance.device_prop_cache_.end()) {
            instance.device_prop_cache_[dev_id] = cudaDeviceProp();
            verify_runtime_success(
                cudaGetDeviceProperties(&instance.device_prop_cache_[dev_id], dev_id));
        }
        return instance.device_prop_cache_[dev_id];
    }

   private:
    std::map<int, cudaDeviceProp> device_prop_cache_;
    std::mutex mtx_;
};

inline size_t availableSharedBytesPerBlock(size_t sharedMemPerMultiprocessor,
                                           size_t sharedSizeBytesStatic, int blocksPerSM,
                                           int smemAllocationUnit) {
    size_t bytes = __occRoundUp(sharedMemPerMultiprocessor / blocksPerSM, smemAllocationUnit) -
                   smemAllocationUnit;
    return bytes - sharedSizeBytesStatic;
}

inline MATAZURE_GENERAL uint3 pointi_to_uint3(pointi<1> p) {
    return {static_cast<unsigned int>(p[0]), 0, 0};
}

inline MATAZURE_GENERAL uint3 pointi_to_uint3(pointi<2> p) {
    return {static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]), 0};
}

inline MATAZURE_GENERAL uint3 pointi_to_uint3(pointi<3> p) {
    return {static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]),
            static_cast<unsigned int>(p[2])};
}

template <int_t _Rank>
inline MATAZURE_GENERAL pointi<_Rank> uint3_to_pointi(uint3 u);

template <>
inline MATAZURE_GENERAL pointi<1> uint3_to_pointi(uint3 u) {
    return {static_cast<int_t>(u.x)};
}

template <>
inline MATAZURE_GENERAL pointi<2> uint3_to_pointi(uint3 u) {
    return {static_cast<int_t>(u.x), static_cast<int>(u.y)};
}

template <>
inline MATAZURE_GENERAL pointi<3> uint3_to_pointi(uint3 u) {
    return {static_cast<int>(u.x), static_cast<int>(u.y), static_cast<int>(u.z)};
}

inline MATAZURE_GENERAL dim3 pointi_to_dim3(pointi<1> p) {
    return {static_cast<unsigned int>(p[0]), 1, 1};
}

inline MATAZURE_GENERAL dim3 pointi_to_dim3(pointi<2> p) {
    return {static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]), 1};
}

inline MATAZURE_GENERAL dim3 pointi_to_dim3(pointi<3> p) {
    return {static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]),
            static_cast<unsigned int>(p[2])};
}

template <int_t _Rank>
inline MATAZURE_GENERAL pointi<_Rank> dim3_to_pointi(dim3 u);

template <>
inline MATAZURE_GENERAL pointi<1> dim3_to_pointi(dim3 u) {
    return {static_cast<int_t>(u.x)};
}

template <>
inline MATAZURE_GENERAL pointi<2> dim3_to_pointi(dim3 u) {
    return {static_cast<int_t>(u.x), static_cast<int>(u.y)};
}

template <>
inline MATAZURE_GENERAL pointi<3> dim3_to_pointi(dim3 u) {
    return {static_cast<int>(u.x), static_cast<int>(u.y), static_cast<int>(u.z)};
}

}  // namespace internal

class execution_policy {
   public:
    execution_policy(
        pointi<3> grid_dim = {{0, 1, 1}}, pointi<3> block_dim = {{0, 1, 1}},
        size_t shared_mem_bytes = 0,
        std::shared_ptr<cudaStream_t> sp_stream = std::make_shared<cudaStream_t>(nullptr))
        : grid_dim_(grid_dim),
          block_dim_(block_dim),
          shared_mem_bytes_(shared_mem_bytes),
          sp_stream_(sp_stream) {
        if (*sp_stream_ == 0) {
            cudaStream_t stream;
            verify_runtime_success(cudaStreamCreate(&stream));
            sp_stream_.reset(new cudaStream_t(stream), [](cudaStream_t* p) {
                verify_runtime_success(cudaStreamSynchronize(*p));
                verify_runtime_success(cudaStreamDestroy(*p));
                delete p;
            });

            // TODO: has bug, refactor it
            // verify_runtime_success(cudaStreamCreate(&stream_));
        }
    }

    pointi<3> grid_dim() const { return grid_dim_; }
    pointi<3> block_dim() const { return block_dim_; }
    size_t shared_mem_bytes() const { return shared_mem_bytes_; }
    cudaStream_t stream() const { return *sp_stream_; }

    void grid_dim(pointi<3> arg) { grid_dim_ = arg; }
    void block_dim(pointi<3> arg) { block_dim_ = arg; }
    void shared_mem_bytes(size_t arg) { shared_mem_bytes_ = arg; }

    void synchronize() { verify_runtime_success(cudaStreamSynchronize(stream())); }

   protected:
    pointi<3> grid_dim_ = {{0, 1, 1}};
    pointi<3> block_dim_ = {{0, 1, 1}};
    // 0 represents not use dynamic shared memory
    size_t shared_mem_bytes_ = 0;
    std::shared_ptr<cudaStream_t> sp_stream_ = nullptr;
};

class default_execution_policy : public execution_policy {
   public:
   protected:
    pointi<3> grid_dim_ = {{0, 1, 1}};
    pointi<3> block_dim_ = {{0, 1, 1}};
    // 0 represents not use dynamic shared memory
    size_t shared_mem_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
};

template <typename _ExePolicy, typename _KernelFunc>
inline void configure_grid(_ExePolicy& exe_policy, _KernelFunc kernel) {
    /// Do none
}

template <typename __KernelFunc>
inline void configure_grid(default_execution_policy& exe_policy, __KernelFunc k) {
    cudaDeviceProp* props;
    props = &internal::device_properties_cache::get();

    cudaFuncAttributes attribs;
    cudaOccDeviceProp occProp(*props);

    verify_runtime_success(cudaFuncGetAttributes(&attribs, k));
    cudaOccFuncAttributes occAttrib(attribs);

    cudaFuncCache cacheConfig;
    verify_runtime_success(cudaDeviceGetCacheConfig(&cacheConfig));
    cudaOccDeviceState occState;
    occState.cacheConfig = (cudaOccCacheConfig)cacheConfig;

    int numSMs = props->multiProcessorCount;

    int bsize = 0, minGridSize = 0;
    verify_occupancy_success(cudaOccMaxPotentialOccupancyBlockSize(
        &minGridSize, &bsize, &occProp, &occAttrib, &occState, exe_policy.shared_mem_bytes()));
    exe_policy.block_dim({bsize, 1, 1});

    cudaOccResult result;
    verify_occupancy_success(cudaOccMaxActiveBlocksPerMultiprocessor(
        &result, &occProp, &occAttrib, &occState, exe_policy.block_dim()[0],
        exe_policy.shared_mem_bytes()));
    exe_policy.grid_dim({result.activeBlocksPerMultiprocessor * numSMs, 1, 1});

    int smemGranularity = 0;
    verify_occupancy_success(cudaOccSMemAllocationGranularity(&smemGranularity, &occProp));
    size_t sbytes = internal::availableSharedBytesPerBlock(
        props->sharedMemPerBlock, attribs.sharedSizeBytes,
        __occDivideRoundUp(exe_policy.grid_dim()[0], numSMs), smemGranularity);

    exe_policy.shared_mem_bytes(sbytes);
}

class for_index_execution_policy : public execution_policy {
   public:
    int_t total_size() const { return total_size_; }
    void total_size(int_t size) { total_size_ = size; }

   protected:
    int_t total_size_ = 0;
};

template <typename __KernelFunc>
inline void configure_grid(for_index_execution_policy& exe_policy, __KernelFunc k) {
    cudaDeviceProp* props;
    props = &internal::device_properties_cache::get();

    cudaFuncAttributes attribs;
    cudaOccDeviceProp occProp(*props);

    verify_runtime_success(cudaFuncGetAttributes(&attribs, k));
    cudaOccFuncAttributes occAttrib(attribs);

    cudaFuncCache cacheConfig;
    verify_runtime_success(cudaDeviceGetCacheConfig(&cacheConfig));
    cudaOccDeviceState occState;
    occState.cacheConfig = (cudaOccCacheConfig)cacheConfig;

    int numSMs = props->multiProcessorCount;

    int bsize = 0, minGridSize = 0;
    verify_occupancy_success(cudaOccMaxPotentialOccupancyBlockSize(
        &minGridSize, &bsize, &occProp, &occAttrib, &occState, exe_policy.shared_mem_bytes()));
    exe_policy.block_dim({bsize, 1, 1});

    cudaOccResult result;
    verify_occupancy_success(cudaOccMaxActiveBlocksPerMultiprocessor(
        &result, &occProp, &occAttrib, &occState, exe_policy.block_dim()[0],
        exe_policy.shared_mem_bytes()));
    exe_policy.grid_dim({result.activeBlocksPerMultiprocessor * numSMs, 1, 1});

    auto pre_block_size = exe_policy.block_dim()[0];
    auto tmp_block_size = __occDivideRoundUp(exe_policy.total_size(), exe_policy.grid_dim()[0]);
    tmp_block_size = __occRoundUp(tmp_block_size, 128);
    exe_policy.block_dim({std::min(tmp_block_size, pre_block_size), 1, 1});

    int smemGranularity = 0;
    verify_occupancy_success(cudaOccSMemAllocationGranularity(&smemGranularity, &occProp));
    size_t sbytes = internal::availableSharedBytesPerBlock(
        props->sharedMemPerBlock, attribs.sharedSizeBytes,
        __occDivideRoundUp(exe_policy.grid_dim()[0], numSMs), smemGranularity);

    exe_policy.shared_mem_bytes(sbytes);
}

template <typename Function, typename... Arguments>
__global__ void kernel(Function f, Arguments... args) {
    f(args...);
}

template <typename _ExecutionPolicy, typename _Fun, typename... _Args>
inline void launch(_ExecutionPolicy exe_policy, _Fun f, _Args... args) {
    configure_grid(exe_policy, kernel<_Fun, _Args...>);
    kernel<<<cuda::internal::pointi_to_dim3(exe_policy.grid_dim()),
             cuda::internal::pointi_to_dim3(exe_policy.block_dim()), exe_policy.shared_mem_bytes(),
             exe_policy.stream()>>>(f, args...);
    verify_runtime_success(cudaGetLastError());
}

template <typename _Fun, typename... _Args>
inline void launch(_Fun f, _Args... args) {
    default_execution_policy exe_policy;
    launch(exe_policy, f, args...);
}

template <typename _Fun>
struct linear_index_functor_kernel {
    int last;
    _Fun fun;

    __device__ void operator()() {
        for (int_t i = threadIdx.x + blockIdx.x * blockDim.x; i < last;
             i += blockDim.x * gridDim.x) {
            fun(i);
        };
    }
};

template <typename _ExecutionPolicy, typename _Fun>
inline void for_linear_index(_ExecutionPolicy policy, int_t last, _Fun fun) {
    linear_index_functor_kernel<_Fun> func{last, fun};
    launch(policy, func);
}

template <typename _ExecutionPolicy, int_t _Rank, typename _Fun>
inline void for_index(_ExecutionPolicy policy, pointi<_Rank> end, _Fun fun) {
    auto extent = end;

    pointi<_Rank> stride;
    stride[0] = extent[0];
    for (int_t i = 1; i < _Rank; ++i) {
        stride[i] = extent[i] * stride[i - 1];
    }

    row_major_layout<_Rank> layout(extent);
    auto max_size = layout.index2offset(end - pointi<_Rank>::all(1)) + 1;  // 要包含最后一个元素

    cuda::for_linear_index(policy, max_size,
                           [=] __device__(int_t i) { fun(layout.offset2index(i)); });
}

template <int_t _Rank, typename _Fun>
inline void for_index(pointi<_Rank> end, _Fun fun) {
    default_execution_policy p;
    cuda::for_index(p, end, fun);
}

}  // namespace matazure::cuda
#endif

namespace matazure {

template <typename _Fun>
struct function_traits : public function_traits<decltype(&_Fun::operator())> {};

/// implements
template <typename _ClassType, typename _ReturnType, typename... _Args>
struct function_traits<_ReturnType (_ClassType::*)(_Args...) const> {
    enum { arguments_size = sizeof...(_Args) };

    typedef _ReturnType result_type;

    template <int_t _index>
    struct arguments {
        typedef typename std::tuple_element<_index, std::tuple<_Args...>>::type type;
    };
};

template <typename _Tensor>
class tensor_expression {
   public:
    typedef _Tensor tensor_type;

    const tensor_type& operator()() const { return *static_cast<const tensor_type*>(this); }

    tensor_type& operator()() { return *static_cast<tensor_type*>(this); }

   protected:
    MATAZURE_GENERAL tensor_expression() {}
    MATAZURE_GENERAL ~tensor_expression() {}
};

template <typename _Type, int_t _Rank>
class tensor : public tensor_expression<tensor<_Type, _Rank>> {
   public:
    static const int_t rank = _Rank;
    typedef _Type value_type;
    typedef value_type& reference;
    typedef value_type* pointer;

    tensor() : tensor(pointi<rank>::all(0)) {}

    explicit tensor(pointi<rank> ext, runtime rt = runtime::host)
        : shape_(ext), layout_(ext), runtime_(rt) {
        if (rt == runtime::host) {
            auto p = new value_type[layout_.size()];
            this->sp_data_ = std::shared_ptr<value_type>(p, [](value_type* p) { delete[] p; });
        } else {
            value_type* p = nullptr;
#ifdef MATAZURE_CUDA
            cuda::verify_runtime_success(cudaMalloc(&p, layout_.size() * sizeof(value_type)));
            this->sp_data_ = std::shared_ptr<value_type>(
                p, [](value_type* p) { cuda::verify_runtime_success(cudaFree(p)); });
#else
            MATAZURE_ASSERT(false, "not in cuda runtime");
#endif
        }

        this->data_ = sp_data_.get();
    }

    explicit tensor(pointi<rank> ext, runtime rt, std::shared_ptr<value_type> sp_data)
        : shape_(ext), layout_(ext), runtime_(rt), sp_data_(sp_data), data_(sp_data_.get()) {}

    template <typename _VT>
    tensor(const tensor<_VT, _Rank>& ts)
        : shape_(ts.shape()), layout_(ts.layout_), sp_data_(ts.shared_data()), data_(ts.data()) {}

    MATAZURE_GENERAL
    std::shared_ptr<value_type> shared_data() const { return sp_data_; }

    MATAZURE_GENERAL reference operator()(const pointi<rank>& index) const {
        return (*this)[layout_.index2offset(index)];
    }

    template <typename... _Idx>
    MATAZURE_GENERAL reference operator()(_Idx... idx) const {
        return (*this)(pointi<rank>{idx...});
    }

    MATAZURE_GENERAL reference operator[](int_t i) const { return data_[i]; }

    MATAZURE_GENERAL pointi<rank> shape() const { return shape_; }

    MATAZURE_GENERAL int_t shape(int_t i) const { return shape_[i]; };

    MATAZURE_GENERAL pointi<rank> stride() const { return layout_.stride(); }

    MATAZURE_GENERAL int_t size() const { return layout_.size(); }

    MATAZURE_GENERAL pointer data() const { return data_; }

    MATAZURE_GENERAL enum runtime runtime() const { return runtime_; }

    tensor<value_type, rank> clone() {
        tensor<value_type, rank> ts_re(shape(), runtime());
        if (runtime_ == runtime::host) {
            memcpy(ts_re.data(), this->data(), sizeof(value_type) * ts_re.size());
        } else {
#ifdef MATAZURE_CUDA
            cuda::verify_runtime_success(cudaMemcpy(
                ts_re.data(), this->data(), sizeof(value_type) * ts_re.size(), cudaMemcpyDefault));
#else
            MATAZURE_ASSERT(false, "not in cuda runtime");
#endif
        }
        return ts_re;
    }

    tensor<value_type, rank> sync(enum runtime rt) {
        if (runtime() == rt) {
            return *this;
        } else {
#ifdef MATAZURE_CUDA
            tensor<value_type, rank> ts_re(shape(), rt);
            cuda::verify_runtime_success(cudaMemcpy(
                ts_re.data(), this->data(), sizeof(value_type) * ts_re.size(), cudaMemcpyDefault));
            return ts_re;
#else
            MATAZURE_ASSERT(false, "not in cuda runtime");
#endif
        }
    }

    MATAZURE_GENERAL ~tensor() {}

   private:
    pointi<rank> shape_;
    enum runtime runtime_;
    row_major_layout<rank> layout_;
    std::shared_ptr<value_type> sp_data_;

    pointer data_;
};

// nvcc walkaround, sometimes you need declare the tensor_type before using
using tensor1b = tensor<uint8_t, 1>;
using tensor2b = tensor<uint8_t, 2>;
using tensor3b = tensor<uint8_t, 3>;
using tensor4b = tensor<uint8_t, 4>;
using tensor1s = tensor<short, 1>;
using tensor2s = tensor<short, 2>;
using tensor3s = tensor<short, 3>;
using tensor4s = tensor<short, 4>;
using tensor1i = tensor<int_t, 1>;
using tensor2i = tensor<int_t, 2>;
using tensor3i = tensor<int_t, 3>;
using tensor4i = tensor<int_t, 4>;
using tensor1f = tensor<float, 1>;
using tensor2f = tensor<float, 2>;
using tensor3f = tensor<float, 3>;
using tensor4f = tensor<float, 4>;
using tensor1d = tensor<double, 1>;
using tensor2d = tensor<double, 2>;
using tensor3d = tensor<double, 3>;
using tensor4d = tensor<double, 4>;

struct sequence_policy {};

MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun>
MATAZURE_GENERAL inline void for_index(sequence_policy, pointi<1> end, _Fun fun) {
    for (int_t i = 0; i < end[0]; ++i) {
        fun(pointi<1>{{i}});
    }
}

MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun>
MATAZURE_GENERAL inline void for_index(sequence_policy, pointi<2> end, _Fun fun) {
    for (int_t i = 0; i < end[0]; ++i) {
        for (int_t j = 0; j < end[1]; ++j) {
            fun(pointi<2>{{i, j}});
        }
    }
}

MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun>
MATAZURE_GENERAL inline void for_index(sequence_policy, pointi<3> end, _Fun fun) {
    for (int_t i = 0; i < end[0]; ++i) {
        for (int_t j = 0; j < end[1]; ++j) {
            for (int_t k = 0; k < end[2]; ++k) {
                fun(pointi<3>{{i, j, k}});
            }
        }
    }
}

MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun>
MATAZURE_GENERAL inline void for_index(sequence_policy, pointi<4> end, _Fun fun) {
    for (int_t i = 0; i < end[0]; ++i) {
        for (int_t j = 0; j < end[1]; ++j) {
            for (int_t k = 0; k < end[2]; ++k) {
                for (int_t l = 0; l < end[3]; ++l) {
                    fun(pointi<4>{{i, j, k, l}});
                }
            }
        }
    }
}

MATAZURE_NV_EXE_CHECK_DISABLE
template <typename _Fun, int_t _Rank>
MATAZURE_GENERAL inline void for_index(pointi<_Rank> end, _Fun fun, runtime rt = runtime::host) {
    if (rt == runtime::host) {
        sequence_policy policy{};
        for_index(policy, end, fun);
    } else {
#ifdef MATAZURE_CUDA
        cuda::for_index(end, fun);
#else
        MATAZURE_ASSERT(false, "not in cuda runtime");
#endif
    }
}

template <typename _Fun>
class lambda_tensor : public tensor_expression<lambda_tensor<_Fun>> {
    typedef function_traits<_Fun> functor_traits;

   public:
    static_assert(functor_traits::arguments_size == 1, "functor should be a parameter");
    typedef std::decay_t<typename functor_traits::template arguments<0>::type> index_type;
    static const int_t rank = index_type::rank;  // functor parameter should be a point
    typedef typename functor_traits::result_type reference;
    typedef std::remove_reference_t<reference> value_type;

   public:
    lambda_tensor(const pointi<rank>& ext, _Fun fun) : shape_(ext), layout_(ext), functor_(fun) {}

    MATAZURE_GENERAL reference operator()(const pointi<rank>& idx) const { return functor_(idx); }

    template <typename... _Idx>
    MATAZURE_GENERAL reference operator()(_Idx... idx) const {
        return (*this)(pointi<rank>{idx...});
    }

    tensor<std::decay_t<value_type>, rank> persist(runtime rt = runtime::host) const {
        tensor<std::decay_t<value_type>, rank> re(this->shape(), rt);
        auto functor_ = this->functor_;
        for_index(
            re.shape(), [=] __general__(pointi<rank> idx) { re(idx) = functor_(idx); }, rt);
        return re;
    }

    MATAZURE_GENERAL pointi<rank> shape() const { return shape_; }

    MATAZURE_GENERAL int_t shape(int_t i) const { return shape()[i]; }

    MATAZURE_GENERAL int_t size() const { return layout_.size(); }

   private:
    pointi<rank> shape_;
    row_major_layout<rank> layout_;
    _Fun functor_;
};

template <int_t _Rank, typename _Fun>
inline auto make_lambda_tensor(pointi<_Rank> extent, _Fun fun) -> lambda_tensor<_Fun> {
    static_assert(lambda_tensor<_Fun>::rank == _Rank, "_Fun rank is not matched with _Rank");
    return lambda_tensor<_Fun>(extent, fun);
}

#define MATAZURE_STATIC_ASSERT_DIM_MATCHED(T1, T2) \
    static_assert(T1::rank == T2::rank, "the rank is not matched")

#define MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(T1, T2)                                \
    static_assert(std::is_same<typename T1::value_type, typename T2::value_type>::value, \
                  "the value type is not matched")

#define __MATAZURE_ARRAY_INDEX_TENSOR_BINARY_OPERATOR(name, op)              \
    template <typename _T1, typename _T2>                                    \
    struct name {                                                            \
       private:                                                              \
        _T1 x1_;                                                             \
        _T2 x2_;                                                             \
                                                                             \
       public:                                                               \
        MATAZURE_STATIC_ASSERT_DIM_MATCHED(_T1, _T2);                        \
        MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_T1, _T2);                 \
        MATAZURE_GENERAL name(_T1 x1, _T2 x2) : x1_(x1), x2_(x2) {}          \
                                                                             \
        MATAZURE_GENERAL auto operator()(const pointi<_T1::rank>& idx) const \
            -> decltype(this->x1_(idx) op this->x2_(idx)) {                  \
            return x1_(idx) op x2_(idx);                                     \
        }                                                                    \
    };

#define TENSOR_BINARY_OPERATOR(name, op)                                               \
    __MATAZURE_ARRAY_INDEX_TENSOR_BINARY_OPERATOR(__##name##_functor__, op)            \
    template <typename _TS1, typename _TS2>                                            \
    inline lambda_tensor<__##name##_functor__<_TS1, _TS2>> operator op(                \
        const tensor_expression<_TS1>& e_lhs, const tensor_expression<_TS2>& e_rhs) {  \
        return make_lambda_tensor(e_lhs().shape(),                                     \
                                  __##name##_functor__<_TS1, _TS2>(e_lhs(), e_rhs())); \
    }

TENSOR_BINARY_OPERATOR(add, +)
TENSOR_BINARY_OPERATOR(sub, -)
TENSOR_BINARY_OPERATOR(mul, *)
TENSOR_BINARY_OPERATOR(div, /)

}  // namespace matazure
