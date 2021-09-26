#pragma once

#include <matazure/lambda_tensor.hpp>
#include <matazure/tensor.hpp>

namespace matazure {

enum dtype {
    none = 0,
    dt_uint8,
    dt_uint16,
    dt_uint32,
    dt_uint64,
    dt_int8,
    dt_int16,
    dt_int32,
    dt_int64,
    dt_float16,
    dt_float32,
    dt_float64
};

template <typename _T>
struct get_data_type_traits;
template <>
struct get_data_type_traits<std::uint8_t> {
    const static dtype value = dtype::dt_uint8;
};
template <>
struct get_data_type_traits<std::uint16_t> {
    const static dtype value = dtype::dt_uint16;
};
template <>
struct get_data_type_traits<std::uint32_t> {
    const static dtype value = dtype::dt_uint32;
};
template <>
struct get_data_type_traits<std::uint64_t> {
    const static dtype value = dtype::dt_uint64;
};
template <>
struct get_data_type_traits<std::int8_t> {
    const static dtype value = dtype::dt_int8;
};
template <>
struct get_data_type_traits<std::int16_t> {
    const static dtype value = dtype::dt_int16;
};
template <>
struct get_data_type_traits<std::int32_t> {
    const static dtype value = dtype::dt_int32;
};
template <>
struct get_data_type_traits<std::int64_t> {
    const static dtype value = dtype::dt_int64;
};

// template <> struct get_data_type_traits<std::float> { const static dtype value =
// dtype::dt_float16; };
template <>
struct get_data_type_traits<float> {
    const static dtype value = dtype::dt_float32;
};
template <>
struct get_data_type_traits<double> {
    const static dtype value = dtype::dt_float64;
};

inline int_t get_data_type_size(dtype data_type) {
    switch (data_type) {
        case dtype::dt_uint8:
            return 1;
        case dtype::dt_int8:
            return 1;
        case dtype::dt_uint16:
            return 2;
        case dtype::dt_int16:
            return 2;
        case dtype::dt_uint32:
            return 4;
        case dtype::dt_int32:
            return 4;
        case dtype::dt_float16:
            return 2;
        case dtype::dt_float32:
            return 4;
        case dtype::dt_float64:
            return 8;
        default:
            MATAZURE_ASSERT(false, "unreachable");
    }

    return 0;
}

class dynamic_tensor {
   public:
    using shape_type = tensor<int_t, 1>;

    dynamic_tensor() {}

    dynamic_tensor(dtype data_type, shape_type shape, shape_type stride, shared_ptr<void> sp_mem)
        : data_type_(data_type),
          shape_(shape),
          stride_(stride),
          size_(reduce(shape_, 1, [](int_t x0, int_t x1) { return x0 * x1; })),
          sp_mem_(std::static_pointer_cast<void>(sp_mem)) {}

    dynamic_tensor(const dynamic_tensor& other)
        : data_type_(other.data_type_),
          shape_(other.shape_),
          stride_(other.stride_),
          size_(other.size_),
          sp_mem_(other.sp_mem_) {}

    dynamic_tensor& operator=(const dynamic_tensor& other) {
        data_type_ = other.data_type_;
        shape_ = other.shape_;
        stride_ = other.stride_;
        size_ = other.size_;
        sp_mem_ = other.sp_mem_;
        return *this;
    }

    dtype dtype() const { return data_type_; }

    shape_type shape() const { return shape_; }

    shape_type stride() const { return stride_; }

    int_t shape(int_t i) const { return shape_[i]; }

    int_t rank() const { return shape_.size(); }

    int_t size() const { return size_; }

    template <typename _Type = void>
    shared_ptr<_Type> shared_data() {
        auto sp_mem = sp_mem_;
        shared_ptr<_Type> sp_tmp(data<_Type>(), [sp_mem](_Type* p) {});
        return sp_tmp;
    }

    template <typename _Type = void>
    shared_ptr<const _Type> shared_data() const {
        auto sp_mem = sp_mem_;
        shared_ptr<const _Type> sp_tmp(data<_Type>(), [sp_mem](const _Type* p) {});
        return sp_tmp;
    }

    template <typename _Type = void>
    _Type* data() {
        return reinterpret_cast<_Type*>(sp_mem_.get());
    }

    template <typename _Type = void>
    const _Type* data() const {
        return reinterpret_cast<_Type*>(sp_mem_.get());
    }

    int_t element_size() const { return get_data_type_size(data_type_); }

   private:
    enum dtype data_type_;
    shared_ptr<void> sp_mem_ = nullptr;
    shape_type shape_;
    shape_type stride_;
    int_t size_;
};

template <typename _Tensor>
dynamic_tensor dynamic_tensor_wrap(_Tensor ts) {
    auto rank = _Tensor::rank;
    dynamic_tensor::shape_type shape(rank);
    copy(ts.shape(), shape);
    dynamic_tensor::shape_type stride(rank);
    copy(ts.layout().stride(), stride);
    shared_ptr<void> sp_tmp(reinterpret_cast<void*>(ts.data()), [ts](void* p) {});
    return dynamic_tensor(get_data_type_traits<typename _Tensor::value_type>::value, shape, stride, sp_tmp);
}

}  // namespace matazure
