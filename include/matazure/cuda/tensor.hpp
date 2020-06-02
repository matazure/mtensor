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

template <typename _Type, int_t _Rank, typename _Layout = row_major_layout<_Rank>,
          typename _Allocator = cuda::allocator<_Type>>
class tensor : public tensor_expression<tensor<_Type, _Rank, _Layout, _Allocator>> {
   public:
    static const int_t rank = _Rank;
    typedef _Type value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef linear_index index_type;
    typedef _Layout layout_type;
    typedef device_t runtime_type;
    typedef _Allocator allocator_type;

    tensor() : tensor(zero<pointi<rank>>::value()) {}

    template <typename... _Ext, typename _Tmp = enable_if_t<sizeof...(_Ext) == rank>>
    explicit tensor(_Ext... ext) : tensor(pointi<rank>{ext...}) {}

    explicit tensor(pointi<rank> ext)
        : shape_(ext),
          layout_(ext),
          sp_data_(malloc_shared_memory(layout_.size())),
          data_(sp_data_.get()) {}

    explicit tensor(pointi<rank> ext, std::shared_ptr<value_type> sp_data)
        : shape_(ext), layout_(ext), sp_data_(sp_data), data_(sp_data_.get()) {}

    /**
     *
     *
     */

    explicit tensor(const pointi<rank>& shape, const pointi<rank>& origin_padding,
                    const pointi<rank>& end_padding)
        : shape_(shape),
          layout_(shape, origin_padding, end_padding),
          sp_data_(malloc_shared_memory(layout_.size())),
          data_(sp_data_.get()) {}

    template <typename _VT>
    tensor(const tensor<_VT, _Rank, _Layout>& ts)
        : allocator_(ts.allocator_),
          shape_(ts.shape()),
          layout_(ts.layout_),
          sp_data_(ts.shared_data()),
          data_(ts.data()) {}

    tensor(std::initializer_list<int_t> v) = delete;

    MATAZURE_GENERAL
    shared_ptr<value_type> shared_data() const { return sp_data_; }

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

    MATAZURE_GENERAL layout_type layout() const { return layout_; }

    MATAZURE_GENERAL constexpr runtime_type runtime() const { return runtime_type{}; }

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
template <typename _ValueType, typename _Layout = row_major_layout<1>>
using vector = tensor<_ValueType, 1, _Layout>;
template <typename _ValueType, typename _Layout = row_major_layout<2>>
using matrix = tensor<_ValueType, 2, _Layout>;
template <typename _ValueType, typename _Layout = row_major_layout<3>>
using volume = tensor<_ValueType, 3, _Layout>;
#endif

template <int_t _Rank, typename _Layout = row_major_layout<_Rank>>
using tensorb = tensor<byte, _Rank, row_major_layout<_Rank>>;
template <int_t _Rank, typename _Layout = row_major_layout<_Rank>>
using tensors = tensor<short, _Rank, row_major_layout<_Rank>>;
template <int_t _Rank, typename _Layout = row_major_layout<_Rank>>
using tensori = tensor<int_t, _Rank, row_major_layout<_Rank>>;
template <int_t _Rank, typename _Layout = row_major_layout<_Rank>>
using tensorf = tensor<float, _Rank, row_major_layout<_Rank>>;
template <int_t _Rank, typename _Layout = row_major_layout<_Rank>>
using tensord = tensor<double, _Rank, row_major_layout<_Rank>>;

// nvcc walkaround, sometimes you need declare the cuda::tensor_type before using
using tensor1b = tensorb<1>;
using tensor2b = tensorb<2>;
using tensor3b = tensorb<3>;
using tensor4b = tensorb<4>;
using tensor1s = tensors<1>;
using tensor2s = tensors<2>;
using tensor3s = tensors<3>;
using tensor4s = tensors<4>;
using tensor1i = tensori<1>;
using tensor2i = tensori<2>;
using tensor3i = tensori<3>;
using tensor4i = tensori<4>;
using tensor1f = tensorf<1>;
using tensor2f = tensorf<2>;
using tensor3f = tensorf<3>;
using tensor4f = tensorf<4>;
using tensor1d = tensord<1>;
using tensor2d = tensord<2>;
using tensor3d = tensord<3>;
using tensor4d = tensord<4>;

}  // namespace cuda

}  // namespace matazure
