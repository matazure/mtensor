/**
* Defines tensor classes of host end
*/

#pragma once

#include <malloc.h>
#include <matazure/meta.hpp>
#include <matazure/type_traits.hpp>
#include <matazure/algorithm.hpp>
#include <matazure/exception.hpp>
#ifdef MATAZURE_CUDA
#include <matazure/cuda/exception.hpp>
#endif

///  matazure is the top namespace for all classes and functions
namespace matazure {

template <int_t... _Values>
using dim = meta::array<_Values ...>;

/**
* \defgroup Tensor Memory Layout
* @{
*/

template <int_t _Rank>
class first_major_layout{
public:
	const static int_t rank = _Rank;

	first_major_layout(const pointi<rank> &shape) :
		shape_(shape),
		stride_(get_stride(shape))
	{
		for_each(shape_, [](int_t b){
			if (b < 0) throw invalid_shape{};
		});
	}

	first_major_layout(const first_major_layout &rhs) :
		first_major_layout(rhs.shape())
	{ }

	first_major_layout & operator=(const first_major_layout &rhs) {
		shape_ = rhs.shape();
		stride_ = get_stride(shape_);

		matazure::for_each(shape_, [](int_t b) {
			if (b < 0) throw invalid_shape{};
		});

		return *this;
	}

	MATAZURE_GENERAL int_t index2offset(const pointi<rank> &id) const {
		int_t offset = id[0];
		for (int_t i = 1; i < rank; ++i) {
			offset += id[i] * stride_[i - 1];
		}

		return offset;
	};

	MATAZURE_GENERAL pointi<rank> offset2index(int_t offset) const {
		pointi<rank> id{};
		for (int_t i = rank - 1; i > 0; --i) {
			id[i] = offset / stride_[i - 1];
			offset = offset % stride_[i - 1];
		}
		id[0] = offset;

		return id;
	}

	MATAZURE_GENERAL pointi<rank> shape() const{
		return shape_;
	}

	MATAZURE_GENERAL pointi<rank> stride() const{
		return stride_;
	}

private:
	static pointi<rank> get_stride(pointi<rank> ext) {
		pointi<rank>  stride{};
		stride[0] = ext[0];
		for (int_t i = 1; i < rank; ++i) {
			stride[i] = ext[i] * stride[i - 1];
		}

		return stride;
	}

private:
	pointi<rank> shape_;
	pointi<rank> stride_;
};

template <int_t _Rank>
class last_major_layout{
public:
	const static int_t rank = _Rank;

	last_major_layout(const pointi<rank> &shape) :
		shape_(shape),
		stride_(get_stride(shape))
	{ }

	MATAZURE_GENERAL int_t index2offset(const pointi<rank> &id) const {
		typename pointi<rank>::value_type offset = id[rank - 1];
		for (int_t i = 1; i < rank; ++i) {
			offset += id[rank - 1 - i] * stride_[i - 1];
		}

		return offset;
	};

	MATAZURE_GENERAL pointi<rank> offset2index(int_t offset) const {
		pointi<rank> id{};
		for (int_t i = rank - 1; i > 0; --i) {
			id[rank - 1 - i] = offset / stride_[i - 1];
			offset = offset % stride_[i - 1];
		}
		id[rank - 1] = offset;

		return id;
	}

	pointi<rank> shape() const{
		return shape_;
	}

	pointi<rank> stride() const{
		return stride_;
	}


private:
	static pointi<rank> get_stride(pointi<rank> ext) {
		pointi<rank>  stride{};
		stride[0] = ext[rank - 1];
		for (int_t i = 1; i < rank; ++i) {
			stride[i] = ext[rank - 1 -i] * stride[i - 1];
		}
		return stride;
	}

private:
	pointi<rank> shape_;
	pointi<rank> stride_;
};

// template <int_t _Rank>
// class last_major_layout{
// 	const static int_t rank = _Rank;
//
// private:
// 	pointi<rank> shape_;
// 	pointi<rank> stride_;
// };

/**
* @brief convert array index to linear index by first marjor
* @param id array index
* @param stride tensor stride
* @param first_major
* @return linear index
*/
template <int_t _Rank>
inline MATAZURE_GENERAL typename pointi<_Rank>::value_type index2offset(const pointi<_Rank> &id, const pointi<_Rank> &stride, first_major) {
	typename pointi<_Rank>::value_type offset = id[0];
	for (int_t i = 1; i < _Rank; ++i) {
		offset += id[i] * stride[i - 1];
	}

	return offset;
};

/**
* @brief convert array index to linear index by first marjor
* @param offset linear index
* @param stride the stride of tensor
* @param first_major
* @return array index
*/
template <int_t _Rank>
inline MATAZURE_GENERAL pointi<_Rank> offset2index(typename pointi<_Rank>::value_type offset, const pointi<_Rank> &stride, first_major) {
	pointi<_Rank> id;
	for (int_t i = _Rank - 1; i > 0; --i) {
		id[i] = offset / stride[i - 1];
		offset = offset % stride[i - 1];
	}
	id[0] = offset;

	return id;
}

/**
* @brief convert array index to linear index by last marjor
* @param id array index
* @param stride tensor stride
* @param first_major
* @return linear index
*/
template <int_t _Rank>
inline MATAZURE_GENERAL typename pointi<_Rank>::value_type index2offset(const pointi<_Rank> &id, const pointi<_Rank> &stride, last_major) {
	typename pointi<_Rank>::value_type offset = id[_Rank - 1];
	for (int_t i = 1; i < _Rank; ++i) {
		offset += id[_Rank - 1 - i] * stride[i - 1];
	}

	return offset;
};

/**
* @brief convert array index to linear index by last marjor
* @param offset linear index
* @param stride the stride of tensor
* @param first_major
* @return array index
*/
template <int_t _Rank>
inline MATAZURE_GENERAL pointi<_Rank> offset2index(typename pointi<_Rank>::value_type offset, const pointi<_Rank> &stride, last_major) {
	pointi<_Rank> id;
	for (int_t i = _Rank - 1; i > 0; --i) {
		id[_Rank - 1 - i] = offset / stride[i - 1];
		offset = offset % stride[i - 1];
	}
	id[_Rank - 1] = offset;

	return id;
}

/**@}*/

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
	typedef _Tensor						tensor_type;

	MATAZURE_GENERAL const tensor_type &operator()() const {
		return *static_cast<const tensor_type *>(this);
	}

	MATAZURE_GENERAL tensor_type &operator()() {
		return *static_cast<tensor_type *>(this);
	}

protected:
	MATAZURE_GENERAL tensor_expression() {}
	MATAZURE_GENERAL ~tensor_expression() {}
};

/**
* @brief a compile time tensor which uses static memory
* @tparam _ValueType element value type, must be pod
* @tparam _Ext a fixed shape, must be dim(meta::array) type
*/
template <typename _ValueType, typename _Ext>
class static_tensor {
private:
	template <int_t ..._Values>
	struct traits;

	template <int_t _S0>
	struct traits<_S0> {
		MATAZURE_GENERAL static constexpr int_t size() {
			return _S0;
		}

		MATAZURE_GENERAL static constexpr pointi<1> stride() {
			return pointi<1>{ { _S0 } };
		}
	};

	template <int_t _S0, int_t _S1>
	struct traits<_S0, _S1> {
		MATAZURE_GENERAL static constexpr int_t size() {
			return _S0 * _S1;
		}

		MATAZURE_GENERAL static constexpr pointi<2> stride() {
			return{ { _S0, _S0 * _S1 } };
		}
	};

	template <int_t _S0, int_t _S1, int_t _S2>
	struct traits<_S0, _S1, _S2> {
		MATAZURE_GENERAL static constexpr int_t size() {
			return _S0 * _S1 * _S2;
		}

		MATAZURE_GENERAL static constexpr pointi<3> stride() {
			return{ { _S0, _S0 * _S1, _S0 * _S1 * _S2 } };
		}
	};

	template <int_t _S0, int_t _S1, int_t _S2, int_t _S3>
	struct traits<_S0, _S1, _S2, _S3> {
		MATAZURE_GENERAL static constexpr int_t size() {
			return _S0 * _S1 * _S2 * _S3;
		}

		MATAZURE_GENERAL static constexpr pointi<4> stride() {
			return{ { _S0, _S0 * _S1, _S0 * _S1 * _S2, _S0 * _S1 * _S2 * _S3 } };
		}
	};

	template <typename _T>
	struct traits_helper;

	template <int_t ..._Values>
	struct traits_helper<dim<_Values...>> {
		typedef traits<_Values...> type;
	};

	typedef typename traits_helper<_Ext>::type traits_t;

	/// @todo should check each dim
	static_assert(traits_t::size() > 0, "");

public:
	/// the meta shape type which has compile time ext
	typedef _Ext					meta_shape_type;
	static	const int_t				rank = meta_shape_type::size();
	typedef _ValueType				value_type;
	typedef value_type *			pointer;
	typedef const pointer			const_pointer;
	typedef value_type &			reference;
	typedef const value_type &		const_reference;
	typedef linear_index			index_type;
	typedef local_tag				memory_type;

	/**
	* @brief accesses element by linear access mode
	* @param i linear index
	* @return element const referece
	*/
	MATAZURE_GENERAL reference operator[](int_t i) {
		 return elements_[i];
	}

	/**
	* @brief accesses element by linear access mode
	* @param i linear index
	* @return element referece
	*/
	MATAZURE_GENERAL constexpr const_reference operator[](int_t i) const {
		return elements_[i];
	}

	/**
	* @brief accesses element by array access mode
	* @param idx array index
	* @return element const reference
	*/
	MATAZURE_GENERAL constexpr const_reference operator[](const pointi<rank> &idx) const {
		return (*this)[index2offset(idx, stride(), first_major{})];
	}

	/**
	* @brief accesses element by array access mode
	* @param idx array index
	* @return element reference
	*/
	MATAZURE_GENERAL reference operator[](const pointi<rank> &idx) {
		return (*this)[index2offset(idx, stride(), first_major{})];
	}

	/**
	* @brief accesses element by array access mode
	* @param idx packed array index parameters
	* @return element const reference
	*/
	template <typename ..._Idx>
	MATAZURE_GENERAL reference operator()(_Idx... idx) {
		return (*this)[pointi<rank>{ idx... }];
	}

	/// prevents operator() const matching with pointi<rank> argument
	template <typename _Idx>
	MATAZURE_GENERAL reference operator()(_Idx idx) const {
		static_assert(std::is_same<_Idx, int_t>::value && rank == 1, "only operator [] support access data by pointi");
		return (*this)[pointi<1>{idx}];
	}

	/**
	* @brief accesses element by array access mode
	* @param idx packed array index parameters
	* @return element reference
	*/
	template <typename ..._Idx>
	MATAZURE_GENERAL constexpr const_reference operator()(_Idx... idx) const {
		return (*this)[pointi<rank>{ idx... }];
	}

	/// prevents operator() matching with pointi<rank> argument
	template <typename _Idx>
	MATAZURE_GENERAL const_reference operator()(_Idx idx) {
		static_assert(std::is_same<_Idx, int_t>::value && rank == 1, "only operator [] support access data by pointi");
		return (*this)[pointi<1>{idx}];
	}

	/// @return the meta shape instance of tensor
	MATAZURE_GENERAL static constexpr meta_shape_type meta_shape() {
		return meta_shape_type();
	}

	/// @return the shape of tensor
	MATAZURE_GENERAL constexpr pointi<rank> shape() const {
		return meta_shape_type::value();
	}

	/// @return the stride of tensor
	MATAZURE_GENERAL constexpr pointi<rank> stride() const {
		return traits_t::stride();
	}

	/// return the total size of tensor elements
	MATAZURE_GENERAL constexpr int_t size() const {
		return traits_t::size();
	}

	/// return the const pointer of tensor elements
	MATAZURE_GENERAL const_pointer data() const {
		return elements_;
	}

	/// return the pointer of tensor elements
	MATAZURE_GENERAL pointer data() {
		return elements_;
	}

	MATAZURE_GENERAL  constexpr int_t element_size() const {
		return sizeof(value_type);
	}

public:
	value_type			elements_[traits_t::size()];
};

static_assert(std::is_pod<static_tensor<float, dim<3,3>>>::value, "static_tensor should be pod type");

/// alias of static_tensor<_ValueType, 2>
template <typename _ValueType, typename _Ext, typename _Tmp = enable_if_t<_Ext::size() == 2>>
using static_matrix = static_tensor<_ValueType, _Ext>;

/// alias of static_tensor<_ValueType, 1>
template <typename _ValueType, typename _Ext, typename _Tmp = enable_if_t<_Ext::size() == 1>>
using static_vector = static_tensor<_ValueType, _Ext>;

/// special for static_tensor
template <typename _ValueType, typename _Ext>
struct is_tensor<static_tensor<_ValueType, _Ext>> : bool_constant<true> {};

/**
@ brief a dense tensor on host which uses dynamic alloced memory
@ tparam _ValueType the value type of elements
@ tparam _Rank the rank of tensor
@ tparam _Layout the memory layout of tensor, the default is first major
*/
template <typename _ValueType, int_t _Rank, typename _Layout = first_major_layout<_Rank>>
class tensor : public tensor_expression<tensor<_ValueType, _Rank, _Layout>> {
public:
	//static_assert(std::is_pod<_ValueType>::value, "only supports pod type now");
	/// the rank of tensor
	static const int_t						rank = _Rank;
	/**
	* @brief tensor element value type
	*
	* a value type should be pod without & qualifier. when a value with const qualifier,
	* tensor is readable only.
	*/
	typedef _ValueType						value_type;
	typedef _ValueType &					reference;
	typedef _Layout							layout_type;
	/// primitive linear access mode
	typedef linear_index					index_type;
	/// host memory type
	typedef host_tag						memory_type;

public:
	/// default constructor
	tensor() :
		tensor(pointi<rank>::zeros())
	{ }

#ifndef MATZURE_CUDA

	/**
	* @brief constructs by the shape
	* @prama ext the shape of tensor
	*/
	explicit tensor(pointi<rank> ext) :
		extent_(ext),
		layout_(ext),
		sp_data_(malloc_shared_memory(layout_.stride()[rank - 1])),
		data_(sp_data_.get())
	{ }

#else

	/**
	* @brief constructs by the shape. alloc host pinned memory when with cuda by default
	* @param ext the shape of tensor
	*/
	explicit tensor(pointi<rank> ext) :
		tensor(ext, pinned{})
	{}

	/**
	* @brief constructs by the shape. alloc host pinned memory when with cuda
	* @param ext the shape of tensor
	* @param pinned_v  the instance of pinned
	*/
	explicit tensor(pointi<rank> ext, pinned pinned_v) :
		extent_(ext),
		layout_(ext),
		sp_data_(malloc_shared_memory(layout_.stride()[rank - 1], pinned_v)),
		data_(sp_data_.get())
	{ }

	/**
	* @brief constructs by the shape. alloc host unpinned memory when with cuda
	* @param ext the shape of tensor
	* @param pinned_v  the instance of unpinned
	*/
	explicit tensor(pointi<rank> ext, unpinned) :
		extent_(ext),
		layout_(ext),
		sp_data_(malloc_shared_memory(layout_.stride()[rank - 1])),
		data_(sp_data_.get())
	{ }

#endif

	/**
	* @brief constructs by the shape
	* @prama ext the packed shape parameters
	*/
	 template <typename ..._Ext, typename _Tmp = enable_if_t<sizeof...(_Ext) == rank>>
	 explicit tensor(_Ext... ext) :
	 	tensor(pointi<rank>{ ext... })
	 {}

	/**
	* @brief constructs by the shape and alloced memory
	* @param ext the shape of tensor
	* @param sp_data the shared_ptr of host memory
	*/
	explicit tensor(const pointi<rank> &ext, std::shared_ptr<value_type> sp_data) :
		extent_(ext),
		layout_(ext),
		sp_data_(sp_data),
		data_(sp_data_.get())
	{ }

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
	tensor(const tensor<_VT, rank, layout_type> &ts) :
		extent_(ts.shape()),
		layout_(ts.shape()),
		sp_data_(ts.shared_data()),
		data_(ts.data())
	{ }

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
	const tensor &operator=(const tensor<_VT, _Rank, _Layout> &ts) {
		extent_ = ts.shape();
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
	reference operator[](int_t i) const {
		 return data_[i];
	}

	/**
	* @brief accesses element by array access mode
	* @param idx array index
	* @return element const reference
	*/
	reference operator[](const pointi<rank> &idx) const {
		return (*this)[layout_.index2offset(idx)];
	}

	/**
	* @brief accesses element by array access mode
	* @param idx array index
	* @return element const reference
	*/
	reference operator()(const pointi<rank> &idx) const {
		return (*this)[layout_.index2offset(idx)];
	}

	/**
	* @brief accesses element by array access mode
	* @param idx packed array index parameters
	* @return element const reference
	*/
	template <typename ..._Idx>
	reference operator()(_Idx... idx) const {
		return (*this)(pointi<rank>{ idx... });
	}

	/// prevents operator() const matching with pointi<rank> argument
	//template <typename _Idx>
	//reference operator()(_Idx idx) const {
	//	static_assert(std::is_same<_Idx, int_t>::value && rank == 1,\
	//				  "only operator [] support access data by pointi");
	//	return (*this)[pointi<1>{idx}];
	//}

	/// @return the shape of tensor
	pointi<rank> shape() const {
		return extent_;
	}


	/// return the total size of tensor elements
	int_t size() const {
		 return layout_.stride()[rank - 1];
	}

	/// return the shared point of tensor elements
	shared_ptr<value_type> shared_data() const {
		 return sp_data_;
	}

	/// return the pointer of tensor elements
	value_type * data() const { return sp_data_.get(); }

	constexpr int_t element_size() const { return sizeof(value_type); };

private:
	shared_ptr<value_type> malloc_shared_memory(int_t size) {
		size = size > 0 ? size : 1;
		value_type *data = new decay_t<value_type>[size];
		return shared_ptr<value_type>(data, [](value_type *ptr) {
			delete[] ptr;
		});
	}

#ifdef MATAZURE_CUDA
	shared_ptr<value_type> malloc_shared_memory(int_t size, pinned) {
		decay_t<value_type> *data = nullptr;
		cuda::assert_runtime_success(cudaMallocHost(&data, size * sizeof(value_type)));
		return shared_ptr<value_type>(data, [](value_type *ptr) {
			cuda::assert_runtime_success(cudaFreeHost(const_cast<decay_t<value_type> *>(ptr)));
		});
	}
#endif

public:
	pointi<rank>	extent_;
	layout_type		layout_;
	shared_ptr<value_type>	sp_data_;
	value_type * data_;
};

using column_major_layout = first_major_layout<2>;
using row_major_layout = last_major_layout<2>;

/// alias of tensor<static_tensor<_ValueType, _BlockDim>, _BlockDim::size(), _Layout>
template <typename _ValueType, typename _BlockDim, typename _Layout = first_major_layout<_BlockDim::size()>>
using block_tensor = tensor<static_tensor<_ValueType, _BlockDim>, _BlockDim::size(), _Layout>;

template <typename _Type, int_t _Rank, typename _Layout = first_major_layout<_Rank>>
auto make_tensor(pointi<_Rank> ext, _Type *p_data)->tensor<_Type, _Rank, _Layout>{
	std::shared_ptr<_Type> sp_data(p_data, [](_Type *p){});
	return tensor<_Type, _Rank, _Layout>(ext, sp_data);
}

template <typename _Type, int_t _Rank, typename _Layout = first_major_layout<_Rank>>
auto make_tensor(pointi<_Rank> ext, _Type t, aligned, size_t alignment=16)->tensor<_Type, _Rank, _Layout>{
	_Layout layout(ext);
	auto ele_size = layout.stride()[_Rank - 1];
#ifdef __linux__
	auto p_aligned = memalign(alignment, ele_size * sizeof(_Type));
	std::shared_ptr<_Type> sp_data((_Type *)p_aligned, [](_Type *p) { free(p); });
#else
	auto p_aligned = _aligned_malloc(ele_size * sizeof(_Type), alignment);
	std::shared_ptr<_Type> sp_data((_Type *)p_aligned, [](_Type *p) { _aligned_free(p); });
#endif
	return tensor<_Type, _Rank, _Layout>(ext, sp_data);
}

namespace internal {

template <int_t _Rank, typename _Func>
struct get_functor_accessor_type {
private:
	typedef function_traits<_Func>						functor_traits;
	static_assert(functor_traits::arguments_size == 1, "functor must be unary");
	typedef	decay_t<typename functor_traits::template arguments<0>::type> _tmp_type;

public:
	typedef conditional_t<
		is_same<int_t, _tmp_type>::value,
		linear_index,
		conditional_t<is_same<_tmp_type, pointi<_Rank>>::value, array_index, void>
	> type;
};

}

/**
* @brief a tensor without memory defined by the shape and lambda(functor)
* @tparam _Rank the rank of tensor
* @tparam _Func the functor type of tensor, should be Index -> Value pattern
* @see tensor
*/
template <int_t _Rank, typename _Func, typename _Layout = first_major_layout<_Rank>>
class lambda_tensor : public tensor_expression<lambda_tensor<_Rank, _Func, _Layout>> {
	typedef function_traits<_Func>						functor_traits;
public:
	static const int_t										rank = _Rank;
	typedef _Func											functor_type;
	typedef typename functor_traits::result_type			reference;
	/// the value type of lambdd_tensor, it's the result type of functor_type
	typedef remove_reference_t<reference>					value_type;
	/**
	* @brief the access mode of lambdd_tensor, it's decided by the argument pattern.
	*
	* when the functor is int_t -> value pattern, the access mode is linear access.
	* when the functor is pointi<rank> -> value pattern, the access mode is array access.
	*/
	typedef typename internal::get_functor_accessor_type<_Rank, _Func>::type	index_type;

	typedef _Layout											layout_type;
	typedef host_tag										memory_type;

public:
	/**
	* @brief constructs a lambdd_tensor by the shape and fun
	* @param ext the shape of tensor
	* @param fun the functor of lambdd_tensor, should be Index -> Value pattern
	*/
	lambda_tensor(const pointi<rank> &ext, _Func fun) :
		extent_(ext),
		layout_(ext),
		fun_(fun)
	{}

	/**
	* @brief copy constructor
	*/
	lambda_tensor(const lambda_tensor &rhs) :
		extent_(rhs.extent_),
		layout_(rhs.layout_),
		fun_(rhs.fun_)
	{ }

	/**
	* @brief accesses element by linear access mode
	* @param i linear index
	* @return element referece
	*/
	reference operator[](int_t i) const {
		return offset_imp<index_type>(i);
	}

	/**
	* @brief accesses element by array access mode
	* @param idx array index
	* @return element const reference
	*/
	reference operator[](pointi<rank> idx) const {
		return index_imp<index_type>(idx);
	}

	/**
	* @brief accesses element by array access mode
	* @param idx array index
	* @return element const reference
	*/
	reference operator()(pointi<rank> idx) const {
		return index_imp<index_type>(idx);
	}

	/**
	* @brief accesses element by array access mode
	* @param idx packed array index parameters
	* @return element const reference
	*/
	template <typename ..._Idx>
	reference operator()(_Idx... idx) const {
		return (*this)[pointi<rank>{ idx... }];
	}

	/// prevents operator() const matching with pointi<rank> argument
	template <typename _Idx>
	reference operator()(_Idx idx) const {
		static_assert(std::is_same<_Idx, int_t>::value && rank == 1, \
					  "only operator [] support access data by pointi");
		return (*this)[pointi<1>{idx}];
	}

	/// @return the shape of lambed_tensor
	pointi<rank> shape() const {
		return extent_;
	}

	/// return the total size of lambda_tensor elements
	int_t size() const {
		 return layout_.stride()[rank - 1];
	}

	/**
	* @brief perisits a lambdd_tensor to a tensor with memory
	* @param policy the execution policy
	* @return a tensor which copys elements value from lambdd_tensor
	*/
	template <typename _ExecutionPolicy>
	MATAZURE_GENERAL tensor<decay_t<value_type>, rank> persist(_ExecutionPolicy policy) const {
		tensor<decay_t<value_type>, rank> re(this->shape());
		copy(policy, *this, re);
		return re;
	}

	/// persists a lambdd_tensor to a tensor by sequence policy
	MATAZURE_GENERAL tensor<decay_t<value_type>, rank> persist() const {
		sequence_policy policy{};
		return persist(policy);
	}

private:
	template <typename _Mode>
	enable_if_t<is_same<_Mode, array_index>::value, reference> index_imp(pointi<rank> idx) const {
		return fun_(idx);
	}

	template <typename _Mode>
	enable_if_t<is_same<_Mode, linear_index>::value, reference> index_imp(pointi<rank> idx) const {
		return (*this)[layout_.index2offset(idx)];
	}

	template <typename _Mode>
	enable_if_t<is_same<_Mode, array_index>::value, reference> offset_imp(int_t i) const {
		return (*this)[layout_.offset2index(i)];
	}

	template <typename _Mode>
	enable_if_t<is_same<_Mode, linear_index>::value, reference> offset_imp(int_t i) const {
		return fun_(i);
	}

private:
	const pointi<rank> extent_;
	const layout_type layout_;
	const _Func fun_;
};

/**
* @brief memcpy a dense tensor to another dense tensor
* @param ts_src source tensor
* @param ts_dst dest tensor
*/
template <typename _TensorSrc, typename _TensorDst>
inline void mem_copy(_TensorSrc ts_src, _TensorDst ts_dst, enable_if_t<are_host_memory<_TensorSrc, _TensorDst>::value && is_same<typename _TensorSrc::layout_type, typename _TensorDst::layout_type>::value>* = nullptr) {
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_TensorSrc, _TensorDst);
	memcpy(ts_dst.data(), ts_src.data(), sizeof(typename _TensorDst::value_type) * ts_src.size());
}

/**
* @brief deeply clone a tensor
* @param ts source tensor
* @return a new tensor which clones source tensor
*/
template <typename _ValueType, int_t _Rank, typename _Layout>
inline tensor<_ValueType, _Rank, _Layout> mem_clone(tensor<_ValueType, _Rank, _Layout> ts, host_tag) {
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
inline auto mem_clone(tensor<_ValueType, _Rank, _Layout> ts)->decltype(mem_clone(ts, host_tag{})) {
	return mem_clone(ts, host_tag{});
}

/**
* @brief reshapes a tensor
* @param ts source tensor
* @param ext a valid new shape
* @return a new ext shape tensor which uses the source tensor memory
*/
template <typename _ValueType, int_t _Rank, typename _Layout, int_t _OutDim, typename _OutLayout = _Layout>
inline auto reshape(tensor<_ValueType, _Rank, _Layout> ts, pointi<_OutDim> ext, _OutLayout* = nullptr)->tensor<_ValueType, _OutDim, _OutLayout> {
	tensor<_ValueType, _OutDim, _OutLayout> re(ext, ts.shared_data());
	MATAZURE_ASSERT(re.size() == ts.size(), "reshape need the size is the same");
	return re;
}

template <typename _T, int_t _Rank>
inline auto split(tensor<_T, _Rank, first_major_layout<_Rank>> ts) {
	const auto slice_ext = slice_point<_Rank - 1>(ts.shape());
	auto slice_size = cumulative_prod(slice_ext)[slice_ext.size() - 1];

	using splitted_tensor_t = tensor<_T, _Rank - 1, first_major_layout<_Rank - 1>>;
	tensor<splitted_tensor_t, 1> ts_splitted(ts.shape()[_Rank - 1]);
	for (int_t i = 0; i < ts_splitted.size(); ++i) {
		ts_splitted[i] = splitted_tensor_t(slice_ext, shared_ptr<_T>(ts.shared_data().get() + i * slice_size, [ts](_T *) {}));
	}

	return ts_splitted;
}

/// alias of tensor<_ValueType, 2>
template <typename _ValueType, typename _Layout = column_major_layout>
using matrix = tensor<_ValueType, 2, _Layout>;


#ifdef MATAZURE_ENABLE_VECTOR_ALIAS
/// alias of tensor <_ValueType, 1>
template <typename _ValueType, typename _Layout = first_major_layout<1>>
using vector = tensor<_ValueType, 1, _Layout>;
#endif

template <int_t _Rank, typename _Layout = column_major_layout> using tensorb = tensor<byte, _Rank, column_major_layout>;
template <int_t _Rank, typename _Layout = column_major_layout> using tensors = tensor<short, _Rank, column_major_layout>;
template <int_t _Rank, typename _Layout = column_major_layout> using tensori = tensor<int_t, _Rank, column_major_layout>;
template <int_t _Rank, typename _Layout = column_major_layout> using tensorf = tensor<float, _Rank, column_major_layout>;
template <int_t _Rank, typename _Layout = column_major_layout> using tensord = tensor<double, _Rank, column_major_layout>;

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

}
