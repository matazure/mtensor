typedef int int_t;

/**
@ brief a dense tensor on host which uses dynamic alloced memory
@ tparam _ValueType the value type of elements
@ tparam _Rank the rank of tensor
@ tparam _Layout the memory layout of tensor, the default is first major
*/
template <typename _ValueType, int_t _Rank, typename _Layout = first_major_layout<_Rank>>
class tensor : public tensor_expression<tensor<_ValueType, _Rank, _Layout>> {
public:
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
	typedef host							memory_type;

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
	 template <typename ..._Ext>
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
		static_assert(std::is_same<_Idx, int_t>::value && rank == 1,\
					  "only operator [] support access data by pointi");
		return (*this)[pointi<1>{idx}];
	}

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

private:
	shared_ptr<value_type> malloc_shared_memory(int_t size) {
		size = size > 0 ? size : 1;
		value_type *data = new decay_t<value_type>[size];
		return shared_ptr<value_type>(data, [](value_type *ptr) {
			delete[] ptr;
		});
	}
    // for (int i = 0; i < X.dim32(1); ++i){
    //   const float * p_input = X.template data<float>() + X.dim32(3) * X.dim32(2) * i;
    //   float * p_output = Y->template mutable_data<float>() + Y->dim32(3) * Y->dim32(2) * i;
    //   const float * p_filter = filter.template data<float>();
    //   auto sp_input = std::shared_ptr<const float>(p_input, [](const float *p){});
    //   matazure::tensor<const float, 2> ts_input(pointi<2>{X.dim32(3), X.dim32(2)}, sp_input);
    //   auto sp_output = std::shared_ptr<float>(p_output, [](const float *p){ });
    //   matazure::tensor<float, 2> ts_output(pointi<2>{Y->dim32(3), Y->dim32(2)}, sp_output);
    //   auto sp_filter = std::shared_ptr<const float>(p_filter, [](const float *p){});
    //   static_tensor<float, dim<3,3>> ts_kenel;
    //   fill(ts_kenel, 1.0f);
    //   expr::conv_kenel3x3(ts_input, ts_kenel, ts_output);
    // }
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
	const pointi<rank>	extent_;
	const layout_type		layout_;
	const shared_ptr<value_type>	sp_data_;
	value_type * const data_;
};

__kernel void myKernel(__global my_struct *myStruct)
{
    int gid = get_global_id(0);
    (myStruct+gid)->a = gid;
    (myStruct+gid)->b = gid + 1;
}
