/**
 * Defines tensor classes of host end
 */

#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/allocator.hpp>
#include <matazure/exception.hpp>
#include <matazure/for_index.hpp>
#include <matazure/layout.hpp>
#include <matazure/local_tensor.hpp>
#include <matazure/tensor_initializer.hpp>
#include <matazure/type_traits.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/exception.hpp>
#endif

///  matazure is the top namespace for all classes and functions
namespace matazure {

/**
 * @brief the base class of tensor expression models
 *
 * this is a nonassignable class, implement the casts to the statically derived
 * for disambiguating glommable expression templates.
 *
 * @tparam _Tensor an tensor expression type
 *
 */
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

/**
@ brief a dense tensor on host which uses dynamic alloced memory
@ tparam _ValueType the value type of elements
@ tparam _Rank the rank of tensor
@ tparam _Layout the memory layout of tensor, the default is first major
*/
template <typename _ValueType, int_t _Rank, typename _Layout = row_major_layout<_Rank>,
          typename _Allocator = aligned_allocator<_ValueType, 32>>
class tensor : public tensor_expression<tensor<_ValueType, _Rank, _Layout, _Allocator>> {
   public:
    /// the rank of tensor
    static const int_t rank = _Rank;
    /**
     * @brief tensor element value type
     *
     * a value type should be pod without & qualifier. when a value with const qualifier,
     * tensor is readable only.
     */
    typedef _ValueType value_type;
    typedef _ValueType& reference;
    typedef _Layout layout_type;
    /// primitive linear access mode
    typedef linear_index index_type;
    /// host memory type
    typedef host_t runtime_type;
    typedef _Allocator allocator_type;

   public:
    /// default constructor
    tensor() : tensor(zero<pointi<rank>>::value()) {}

    /**
     * @brief constructs by the shape
     * @prama ext the shape of tensor
     */
    explicit tensor(pointi<rank> ext)
        : shape_(ext),
          layout_(ext),
          sp_data_(malloc_shared_memory(layout_.size())),
          data_(sp_data_.get()) {}

    /**
     * @brief constructs by the shape
     * @prama ext the packed shape parameters
     */
    template <typename... _Ext, typename _Tmp = enable_if_t<sizeof...(_Ext) == rank>>
    explicit tensor(_Ext... ext) : tensor(pointi<rank>{ext...}) {}

    /**
     * @brief constructs by the shape and alloced memory
     * @param ext the shape of tensor
     * @param sp_data the shared_ptr of host memory
     */
    explicit tensor(const pointi<rank>& ext, std::shared_ptr<value_type> sp_data)
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

    /**
     * @brief shallowly copy constructor
     *
     * we use _VT template argument as source value type.
     * a tensor<int, 3> could be construced by a tensor<const intver,3>,
     * but a tensor<const int, 3> could not be constructed by a tensor<int, 3>
     *
     * @param ts the source tensor
     * @tparam _VT the value type of the source tensor, should be value_type or const value_type
     */
    template <typename _VT>
    tensor(const tensor<_VT, rank, layout_type, allocator_type>& ts)
        : allocator_(ts.allocator_),
          shape_(ts.shape()),
          layout_(ts.shape()),
          sp_data_(ts.shared_data()),
          data_(ts.data()) {}

    tensor(typename nested_initializer_list<value_type, rank>::type init)
        : tensor(nested_initializer_list<value_type, rank>::shape(init)) {
        for_index(shape(), [&](pointi<rank> idx) {
            (*this)(idx) = nested_initializer_list<value_type, rank>::access(init, idx);
        });
    }

    /**
     * @brief shallowly assign operator
     *
     * we use _VT template argument as source value type.
     * a tensor<int, 3> could be assigned by a tensor<const intver,3>,
     * but a tensor<const int, 3> could not be assigned by a tensor<int, 3>
     *
     * @param ts the source tensor
     * @tparam _VT the value type of the source tensor, should be value_type or const value_type
     */
    template <typename _VT>
    const tensor& operator=(const tensor<_VT, _Rank, _Layout>& ts) {
        shape_ = ts.shape();
        layout_ = ts.layout_;
        sp_data_ = ts.shared_data();
        data_ = ts.data();

        return *this;
    }

    /**
     * @brief accesses element by linear access mode
     * @param i linear index
     * @return element referece
     */
    reference operator[](int_t i) const { return data_[i]; }

    /**
     * @brief accesses element by array access mode
     * @param idx array index
     * @return element const reference
     */
    reference operator()(const pointi<rank>& idx) const {
        return (*this)[layout_.index2offset(idx)];
    }

    /**
     * @brief accesses element by array access mode
     * @param idx packed array index parameters
     * @return element const reference
     */
    template <typename... _Idx>
    reference operator()(_Idx... idx) const {
        return (*this)(pointi<rank>{idx...});
    }

    /// prevents operator() const matching with pointi<rank> argument
    // template <typename _Idx>
    // reference operator()(_Idx idx) const {
    //	static_assert(std::is_same<_Idx, int_t>::value && rank == 1,\
		//				  "only operator [] support access data by pointi");
    //	return (*this)[pointi<1>{idx}];
    //}

    /// @return the shape of tensor
    pointi<rank> shape() const { return shape_; }

    int_t shape(int_t i) const { return shape()[i]; }

    /// return the total size of tensor elements
    int_t size() const { return layout_.size(); }

    /// return the shared point of tensor elements
    shared_ptr<value_type> shared_data() const { return sp_data_; }

    /// return the pointer of tensor elements
    value_type* data() const { return sp_data_.get(); }

    constexpr int_t element_size() const { return sizeof(value_type); }

    layout_type layout() const { return layout_; }

    constexpr runtime_type runtime() const { return runtime_type{}; }

    allocator_type get_allocator() const { return allocator_; }

   private:
    shared_ptr<value_type> malloc_shared_memory(int_t size) {
        value_type* data = allocator_.allocate(size);
        for (int_t i = 0; i < size; ++i) {
            allocator_.construct(data + i);
        }
        return shared_ptr<value_type>(data, [=](value_type* ptr) {
            for (int_t i = 0; i < size; ++i) {
                allocator_.destroy(data + i);
            }
            allocator_.deallocate(data, size);
        });
    }

   public:
    allocator_type allocator_;
    pointi<rank> shape_;
    layout_type layout_;
    shared_ptr<value_type> sp_data_;
    value_type* data_;
};

template <typename _Type, int_t _Rank, typename _Layout = row_major_layout<_Rank>>
auto make_tensor(pointi<_Rank> ext, _Type* p_data) -> tensor<_Type, _Rank, _Layout> {
    std::shared_ptr<_Type> sp_data(p_data, [](_Type* p) {});
    return tensor<_Type, _Rank, _Layout>(ext, sp_data);
}

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

}  // namespace matazure
