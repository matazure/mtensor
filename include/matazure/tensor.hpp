/**
 * Defines tensor classes of host end
 */

#pragma once

#include <matazure/algorithm.hpp>
#include <matazure/exception.hpp>
#include <matazure/type_traits.hpp>
#ifdef MATAZURE_CUDA
#include <matazure/cuda/exception.hpp>
#endif
#include <matazure/layout.hpp>
#include <matazure/local_tensor.hpp>

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
template <typename _ValueType, int_t _Rank, typename _Layout = column_major_layout<_Rank>>
class tensor : public tensor_expression<tensor<_ValueType, _Rank, _Layout>> {
   public:
    // static_assert(std::is_pod<_ValueType>::value, "only supports pod type now");
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
    typedef host_tag memory_type;

   public:
    /// default constructor
    tensor() : tensor(pointi<rank>::zeros()) {}

#ifndef MATZURE_CUDA

    /**
     * @brief constructs by the shape
     * @prama ext the shape of tensor
     */
    explicit tensor(pointi<rank> ext)
        : shape_(ext),
          layout_(ext),
          sp_data_(malloc_shared_memory(layout_.stride()[rank - 1])),
          data_(sp_data_.get()) {}

#else

    /**
     * @brief constructs by the shape. alloc host pinned memory when with cuda by default
     * @param ext the shape of tensor
     */
    explicit tensor(pointi<rank> ext) : tensor(ext, pinned{}) {}

    /**
     * @brief constructs by the shape. alloc host pinned memory when with cuda
     * @param ext the shape of tensor
     * @param pinned_v  the instance of pinned
     */
    explicit tensor(pointi<rank> ext, pinned pinned_v)
        : shape_(ext),
          layout_(ext),
          sp_data_(malloc_shared_memory(layout_.stride()[rank - 1], pinned_v)),
          data_(sp_data_.get()) {}

    /**
     * @brief constructs by the shape. alloc host unpinned memory when with cuda
     * @param ext the shape of tensor
     * @param pinned_v  the instance of unpinned
     */
    explicit tensor(pointi<rank> ext, unpinned)
        : shape_(ext),
          layout_(ext),
          sp_data_(malloc_shared_memory(layout_.stride()[rank - 1])),
          data_(sp_data_.get()) {}

#endif

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
        : shape_(ext),
          layout_(ext),  // if compile error, make sure you call right constructor
          sp_data_(sp_data),
          data_(sp_data_.get()) {}

    /**
     *
     *
     */
    explicit tensor(const pointi<rank>& shape, const pointi<rank>& origin_padding,
                    const pointi<rank>& end_padding)
        : shape_(shape),
          layout_(shape, origin_padding, end_padding),
          sp_data_(malloc_shared_memory(layout_.stride()[rank - 1])),
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
    tensor(const tensor<_VT, rank, layout_type>& ts)
        : shape_(ts.shape()), layout_(ts.shape()), sp_data_(ts.shared_data()), data_(ts.data()) {}

    /// prevents constructing tensor by {...}, such as tensor<int,3> ts({10, 10});
    tensor(std::initializer_list<int_t> v) = delete;

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
    reference operator[](const pointi<rank>& idx) const {
        return (*this)[layout_.index2offset(idx)];
    }

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

    /// return the total size of tensor elements
    int_t size() const { return layout_.stride()[rank - 1]; }

    /// return the shared point of tensor elements
    shared_ptr<value_type> shared_data() const { return sp_data_; }

    /// return the pointer of tensor elements
    value_type* data() const { return sp_data_.get(); }

    constexpr int_t element_size() const { return sizeof(value_type); }

    layout_type layout() const { return layout_; }

   private:
    shared_ptr<value_type> malloc_shared_memory(int_t size) {
        size = size > 0 ? size : 1;
        // value_type *data = new decay_t<value_type>[size];
#ifdef __GNUC__
        value_type* data;
        auto tmp = posix_memalign((void**)&data, 32, size * sizeof(value_type));
        return shared_ptr<value_type>(data, [](value_type* ptr) { free(ptr); });
#else
        value_type* data;
        data = static_cast<value_type*>(_aligned_malloc(size * sizeof(value_type), 32));
        return shared_ptr<value_type>(data, [](value_type* ptr) { _aligned_free(ptr); });
#endif
    }

#ifdef MATAZURE_CUDA
    shared_ptr<value_type> malloc_shared_memory(int_t size, pinned) {
        decay_t<value_type>* data = nullptr;
        cuda::assert_runtime_success(cudaMallocHost(&data, size * sizeof(value_type)));
        return shared_ptr<value_type>(data, [](value_type* ptr) {
            cuda::assert_runtime_success(cudaFreeHost(const_cast<decay_t<value_type>*>(ptr)));
        });
    }
#endif

   public:
    pointi<rank> shape_;
    layout_type layout_;
    shared_ptr<value_type> sp_data_;
    value_type* data_;
};

template <typename _Type, int_t _Rank, typename _Layout = column_major_layout<_Rank>>
auto make_tensor(pointi<_Rank> ext, _Type* p_data) -> tensor<_Type, _Rank, _Layout> {
    std::shared_ptr<_Type> sp_data(p_data, [](_Type* p) {});
    return tensor<_Type, _Rank, _Layout>(ext, sp_data);
}

/**
 * @brief memcpy a dense tensor to another dense tensor
 * @param ts_src source tensor
 * @param ts_dst dest tensor
 */
template <typename _TensorSrc, typename _TensorDst>
inline void mem_copy(_TensorSrc ts_src, _TensorDst ts_dst,
                     enable_if_t<are_host_memory<_TensorSrc, _TensorDst>::value &&
                                 is_same<typename _TensorSrc::layout_type,
                                         typename _TensorDst::layout_type>::value>* = nullptr) {
    MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_TensorSrc, _TensorDst);
    memcpy(ts_dst.data(), ts_src.data(), sizeof(typename _TensorDst::value_type) * ts_src.size());
}

/**
 * @brief deeply clone a tensor
 * @param ts source tensor
 * @return a new tensor which clones source tensor
 */
template <typename _ValueType, int_t _Rank, typename _Layout>
inline tensor<_ValueType, _Rank, _Layout> mem_clone(tensor<_ValueType, _Rank, _Layout> ts,
                                                    host_tag) {
    tensor<decay_t<_ValueType>, _Rank, _Layout> ts_re(ts.shape());
    mem_copy(ts, ts_re);
    return ts_re;
}

/**
 * @brief deeply clone a tensor
 * @param ts source tensor
 * @return a new tensor which clones source tensor
 */
template <typename _ValueType, int_t _Rank, typename _Layout>
inline auto mem_clone(tensor<_ValueType, _Rank, _Layout> ts)
    -> decltype(mem_clone(ts, host_tag{})) {
    return mem_clone(ts, host_tag{});
}

/**
 * @brief reshapes a tensor
 * @param ts source tensor
 * @param ext a valid new shape
 * @return a new ext shape tensor which uses the source tensor memory
 */
template <typename _ValueType, int_t _Rank, typename _Layout, int_t _OutDim,
          typename _OutLayout = _Layout>
inline auto reshape(tensor<_ValueType, _Rank, _Layout> ts, pointi<_OutDim> ext,
                    _OutLayout* = nullptr) -> tensor<_ValueType, _OutDim, _OutLayout> {
    tensor<_ValueType, _OutDim, _OutLayout> re(ext, ts.shared_data());
    MATAZURE_ASSERT(re.size() == ts.size(), "reshape need the size is the same");
    return re;
}

template <typename _T, int_t _Rank>
inline auto split(tensor<_T, _Rank, column_major_layout<_Rank>> ts)
    -> tensor<tensor<_T, _Rank - 1, column_major_layout<_Rank - 1>>, 1> {
    const auto slice_ext = slice_point<_Rank - 1>(ts.shape());
    auto slice_size = cumulative_prod(slice_ext)[slice_ext.size() - 1];

    using splitted_tensor_t = tensor<_T, _Rank - 1, column_major_layout<_Rank - 1>>;
    tensor<splitted_tensor_t, 1> ts_splitted(ts.shape()[_Rank - 1]);
    for (int_t i = 0; i < ts_splitted.size(); ++i) {
        ts_splitted[i] = splitted_tensor_t(
            slice_ext, shared_ptr<_T>(ts.shared_data().get() + i * slice_size, [ts](_T*) {}));
    }

    return ts_splitted;
}

/// alias of tensor<_ValueType, 2>
template <typename _ValueType, typename _Layout = column_major_layout<2>>
using matrix = tensor<_ValueType, 2, _Layout>;

#ifdef MATAZURE_ENABLE_VECTOR_ALIAS
/// alias of tensor <_ValueType, 1>
template <typename _ValueType, typename _Layout = column_major_layout<1>>
using vector = tensor<_ValueType, 1, _Layout>;
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

using tensor1i = tensor<int_t, 1>;
using tensor2i = tensor<int_t, 2>;
using tensor3i = tensor<int_t, 3>;
using tensor4i = tensor<int_t, 4>;

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
using tensor2d = tensor<double, 2>;
using tensor3d = tensor<double, 3>;
using tensor4d = tensor<double, 4>;

}  // namespace matazure
