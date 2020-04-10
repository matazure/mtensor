#pragma once

#include <cuda_runtime.h>
#include <matazure/cuda/algorithm.hpp>
#include <matazure/cuda/allocator.hpp>
#include <matazure/cuda/runtime.hpp>
#include <matazure/tensor.hpp>

#define MATAZURE_IS_D_LAMBDA(X) __nv_is_extended_device_lambda_closure_type(X)
#define MATAZURE_IS_HD_LAMBDA(X) __nv_is_extended_host_device_lambda_closure_type(X)

namespace matazure {
namespace cuda {

template <typename _Type, int_t _Rank, typename _Layout = column_major_layout<_Rank>,
          typename _Allocator = cuda::allocator<_Type>>
class tensor : public tensor_expression<tensor<_Type, _Rank, _Layout, _Allocator>> {
   public:
    static_assert(std::is_pod<_Type>::value, "only supports pod type now");

    static const int_t rank = _Rank;
    typedef _Type value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef linear_index index_type;
    typedef _Layout layout_type;
    typedef device_tag memory_type;
    typedef _Allocator allocator_type;

    MATAZURE_GENERAL tensor() : tensor(zero<pointi<rank>>::value()) {}

    template <typename... _Ext, typename _Tmp = enable_if_t<sizeof...(_Ext) == rank>>
    MATAZURE_GENERAL explicit tensor(_Ext... ext) : tensor(pointi<rank>{ext...}) {}

    MATAZURE_GENERAL explicit tensor(pointi<rank> ext)
        : shape_(ext),
          layout_(ext),
          sp_data_(malloc_shared_memory(layout_.stride()[rank - 1])),
          data_(sp_data_.get()) {}

    MATAZURE_GENERAL
    explicit tensor(pointi<rank> ext, std::shared_ptr<value_type> sp_data)
        : shape_(ext), layout_(ext), sp_data_(sp_data), data_(sp_data_.get()) {}

    /**
     *
     *
     */
    MATAZURE_GENERAL
    explicit tensor(const pointi<rank>& shape, const pointi<rank>& origin_padding,
                    const pointi<rank>& end_padding)
        : shape_(shape),
          layout_(shape, origin_padding, end_padding),
          sp_data_(malloc_shared_memory(layout_.stride()[rank - 1])),
          data_(sp_data_.get()) {}

    template <typename _VT>
    MATAZURE_GENERAL tensor(const tensor<_VT, _Rank, _Layout>& ts)
        : allocator_(ts.allocator_),
          shape_(ts.shape()),
          layout_(ts.layout_),
          sp_data_(ts.shared_data()),
          data_(ts.data()) {}

    tensor(std::initializer_list<int_t> v) = delete;

    MATAZURE_GENERAL
    shared_ptr<value_type> shared_data() const { return sp_data_; }

    MATAZURE_GENERAL reference operator[](const pointi<rank>& index) const {
        return (*this)[layout_.index2offset(index)];
    }

    template <typename... _Idx>
    MATAZURE_GENERAL reference operator()(_Idx... idx) const {
        return (*this)[pointi<rank>{idx...}];
    }

    MATAZURE_GENERAL reference operator[](int_t i) const { return data_[i]; }

    MATAZURE_GENERAL pointi<rank> shape() const { return shape_; }

    MATAZURE_GENERAL pointi<rank> stride() const { return layout_.stride(); }
    MATAZURE_GENERAL int_t size() const { return layout_.stride()[rank - 1]; }

    MATAZURE_GENERAL pointer data() const { return data_; }

    allocator_type get_allocator() const { return allocator_; }

    MATAZURE_GENERAL ~tensor() {}

   private:
    shared_ptr<value_type> malloc_shared_memory(int_t size) {
        value_type* data = allocator_.allocate(size);
        return shared_ptr<value_type>(data,
                                      [=](value_type* ptr) { allocator_.deallocate(data, size); });
    }

   private:
    allocator_type allocator_;
    pointi<rank> shape_;
    layout_type layout_;
    shared_ptr<value_type> sp_data_;
    pointer data_;
};

#ifndef MATAZURE_DISABLE_MATRIX_VECTOR_ALIAS
template <typename _ValueType, typename _Layout = column_major_layout<1>>
using vector = tensor<_ValueType, 1, _Layout>;
template <typename _ValueType, typename _Layout = column_major_layout<2>>
using matrix = tensor<_ValueType, 2, _Layout>;
template <typename _ValueType, typename _Layout = column_major_layout<3>>
using volume = tensor<_ValueType, 3, _Layout>;
#endif

template <int_t _Rank, typename _Layout = column_major_layout<_Rank>>
using tensorb = tensor<byte, _Rank, column_major_layout<_Rank>>;
template <int_t _Rank, typename _Layout = column_major_layout<_Rank>>
using tensors = tensor<short, _Rank, column_major_layout<_Rank>>;
template <int_t _Rank, typename _Layout = column_major_layout<_Rank>>
using tensori = tensor<int_t, _Rank, column_major_layout<_Rank>>;
template <int_t _Rank, typename _Layout = column_major_layout<_Rank>>
using tensorf = tensor<float, _Rank, column_major_layout<_Rank>>;
template <int_t _Rank, typename _Layout = column_major_layout<_Rank>>
using tensord = tensor<double, _Rank, column_major_layout<_Rank>>;

namespace __walkaround {

using tensor1b = tensor<byte, 1>;
using tensor2b = tensor<byte, 2>;
using tensor3b = tensor<byte, 3>;
using tensor4b = tensor<byte, 4>;

using tensor1s = tensor<short, 1>;
using tensor2s = tensor<short, 2>;
using tensor3s = tensor<short, 3>;
using tensor4s = tensor<short, 4>;

using tensor1us = tensor<unsigned short, 1>;
using tensor2us = tensor<unsigned short, 2>;
using tensor3us = tensor<unsigned short, 4>;
using tensor4us = tensor<unsigned short, 4>;

using tensor1i = tensor<int, 1>;
using tensor2i = tensor<int, 2>;
using tensor3i = tensor<int, 3>;
using tensor4i = tensor<int, 4>;

using tensor1ui = tensor<unsigned int, 1>;
using tensor2ui = tensor<unsigned int, 2>;
using tensor3ui = tensor<unsigned int, 3>;
using tensor4ui = tensor<unsigned int, 4>;

using tensor1l = tensor<long, 1>;
using tensor2l = tensor<long, 2>;
using tensor3l = tensor<long, 3>;
using tensor4l = tensor<long, 4>;

using tensor1ul = tensor<unsigned long, 1>;
using tensor2ul = tensor<unsigned long, 2>;
using tensor3ul = tensor<unsigned long, 3>;
using tensor4ul = tensor<unsigned long, 4>;

using tensor1f = tensor<float, 1>;
using tensor2f = tensor<float, 2>;
using tensor3f = tensor<float, 3>;
using tensor4f = tensor<float, 4>;

using tensor1d = tensor<double, 1>;
using tensor2d = tensor<double, 1>;
using tensor3d = tensor<double, 1>;
using tensor4d = tensor<double, 1>;

}  // namespace __walkaround

}  // namespace cuda

}  // namespace matazure
