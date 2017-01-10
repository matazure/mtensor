#pragma once

#include <cuda_runtime.h>
#include <matazure/tensor.hpp>
#include <matazure/cuda/algorithm.hpp>

namespace matazure {
namespace cuda {

template <typename _Type, int_t _Dim, typename _Layout = first_major_t>
class tensor : public tensor_expression<tensor<_Type, _Dim, first_major_t>> {
public:
	static_assert(std::is_pod<_Type>::value, "only supports pod type now");

	static const int_t						dim = _Dim;
	typedef _Type							value_type;
	typedef value_type &					reference;
	typedef value_type *					pointer;

	typedef pointi<dim>						extent_type;
	typedef pointi<dim>						index_type;

	typedef linear_access_t					access_type;
	typedef _Layout							layout_type;
	typedef device_t						memory_type;

	tensor() :
		tensor(extent_type::zeros())
	{}

	template <typename ..._Ext>
	explicit tensor(_Ext... ext) :
		tensor(extent_type{ext...})
	{}

	explicit tensor(extent_type extent) :
		extent_(extent),
		stride_(matazure::get_stride(extent)),
		sp_data_(malloc_shared_memory(stride_[dim - 1])),
		data_(sp_data_.get())
	{ }

	explicit tensor(extent_type extent, std::shared_ptr<value_type> sp_data) :
		extent_(extent),
		stride_(matazure::get_stride(extent)),
		sp_data_(sp_data),
		data_(sp_data_.get())
	{ }

	shared_ptr<value_type> shared_data() const { return sp_data_; }

	template <typename ..._Idx>
	MATAZURE_GENERAL reference operator()(_Idx... idx) const {
		return (*this)(index_type{ idx... });
	}

	MATAZURE_GENERAL reference operator()(const index_type &index) const {
		return (*this)[index2offset(index, stride_, layout_type{})];
	}

	MATAZURE_GENERAL reference operator[] (int_t i) const {
		return data_[i];
	}

	MATAZURE_GENERAL extent_type extent() const { return extent_; }
	MATAZURE_GENERAL extent_type stride() const { return stride_; }
	MATAZURE_GENERAL int_t size() const { return stride_[dim - 1]; }

	MATAZURE_GENERAL pointer data() const { return data_; }

private:
	shared_ptr<value_type> malloc_shared_memory(int_t size) {
		decay_t<value_type> *data = nullptr;
		throw_on_error(cudaMalloc(&data, size * sizeof(value_type)));
		return shared_ptr<value_type>(data, [](value_type *ptr) {
			throw_on_error(cudaFree(const_cast<decay_t<value_type> *>(ptr)));
		});
	}

private:
	const extent_type	extent_;
	const extent_type	stride_;
	const shared_ptr<value_type>	sp_data_;
	const pointer data_;
};

template <typename _ValueType, typename _Layout = first_major_t>
using matrix = tensor<_ValueType, 2, _Layout>;

template <typename _TensorSrc, typename _TensorDst>
inline void mem_copy(_TensorSrc ts_src, _TensorDst cts_dst) {
	STATIC_ASSERT_VALUE_TYPE_MATCHED(_TensorSrc, _TensorDst);
	throw_on_error(cudaMemcpy(cts_dst.data(), ts_src.data(), sizeof(typename _TensorDst::value_type) * ts_src.size(), cudaMemcpyDefault));
}

template <typename _TensorSrc, typename _TensorSymbol>
inline void copy_symbol(_TensorSrc src, _TensorSymbol &symbol_dst) {
	throw_on_error(cudaMemcpyToSymbol(symbol_dst, src.data(), src.size() * sizeof(typename _TensorSrc::value_type)));
}

template <typename _ValueType, typename _AccessMode, int_t _Dim, typename _Func>
class device_lambda_tensor {
public:
	static const int_t									dim = _Dim;
	typedef _ValueType									value_type;
	typedef pointi<dim>									extent_type;
	typedef pointi<dim>									index_type;
	typedef _AccessMode									accessor_type;
	typedef device_t									memory_type;

public:
	device_lambda_tensor() = delete;

	device_lambda_tensor(const extent_type &extent, _Func fun) :
		extent_(extent),
		stride_(matazure::get_stride(extent)),
		fun_(fun)
	{ }

	MATAZURE_DEVICE value_type operator()(index_type index) const {
		return index_imp<accessor_type>(index);
	}

	template <typename ..._Idx>
	MATAZURE_GENERAL value_type operator()(_Idx... idx) const {
		return (*this)(index_type{ idx... });
	}

	MATAZURE_DEVICE value_type operator[](int_t i) const {
		return offset_imp<accessor_type>(i);
	}

	tensor<value_type, dim> persist() const {
		tensor<value_type, dim> re(this->extent());
		copy(*this, re);
		return re;
	}

	MATAZURE_GENERAL extent_type extent() const { return extent_; }
	MATAZURE_GENERAL extent_type stride() const { return stride_; }
	MATAZURE_GENERAL int_t size() const { return stride_[dim - 1]; }

private:
	template <typename _Mode>
	MATAZURE_DEVICE enable_if_t<is_same<_Mode, array_access_t>::value, value_type>
		index_imp(index_type index) const {
		return fun_(index);
	}

	template <typename _Mode>
	MATAZURE_DEVICE enable_if_t<is_same<_Mode, linear_access_t>::value, value_type>
		index_imp(index_type index) const {
		return (*this)[index2offset(index, stride())];
	}

	template <typename _Mode>
	MATAZURE_DEVICE enable_if_t<is_same<_Mode, array_access_t>::value, value_type>
		offset_imp(int_t i) const {
		return (*this)(offset2index(i, stride()));
	}

	template <typename _Mode>
	MATAZURE_DEVICE enable_if_t<is_same<_Mode, linear_access_t>::value, value_type>
		offset_imp(int_t i) const {
		return fun_(i);
	}

private:
	const extent_type extent_;
	const extent_type stride_;
	const _Func fun_;
};

template <int_t _Dim, typename _Func>
class general_lambda_tensor : public tensor_expression<general_lambda_tensor<_Dim, _Func>> {
	typedef function_traits<_Func>						functor_traits;
public:
	static const int_t										dim = _Dim;
	typedef typename functor_traits::result_type			value_type;
	typedef matazure::pointi<dim>							extent_type;
	typedef pointi<dim>										index_type;
	typedef typename detail::get_functor_accessor_type<_Dim, _Func>::type		access_type;
	typedef device_t										memory_type;


public:
	general_lambda_tensor(const extent_type &extent, _Func fun) :
		extent_(extent),
		stride_(matazure::get_stride(extent)),
		fun_(fun)
	{}

	MATAZURE_GENERAL value_type operator()(index_type index) const {
		return index_imp<access_type>(index);
	}

	MATAZURE_GENERAL value_type operator[](int_t i) const {
		return offset_imp<access_type>(i);
	}

	tensor<value_type, dim> persist() const {
		tensor<value_type, dim> re(this->extent());
		copy(*this, re);
		return re;
	}

	MATAZURE_GENERAL extent_type extent() const { return extent_; }
	MATAZURE_GENERAL extent_type stride() const { return stride_; }
	MATAZURE_GENERAL int_t size() const { return stride_[dim - 1]; }

public:
	template <typename _Mode>
	MATAZURE_GENERAL enable_if_t<is_same<_Mode, array_access_t>::value, value_type> index_imp(index_type index) const {
		return fun_(index);
	}

	template <typename _Mode>
	MATAZURE_GENERAL enable_if_t<is_same<_Mode, linear_access_t>::value, value_type> index_imp(index_type index) const {
		return (*this)[index2offset(index, stride(), first_major_t{})];
	}

	template <typename _Mode>
	MATAZURE_GENERAL enable_if_t<is_same<_Mode, array_access_t>::value, value_type> offset_imp(int_t i) const {
		return (*this)(offset2index(i, stride(), first_major_t{}));
	}

	template <typename _Mode>
	MATAZURE_GENERAL enable_if_t<is_same<_Mode, linear_access_t>::value, value_type> offset_imp(int_t i) const {
		return fun_(i);
	}

private:
	const extent_type extent_;
	const extent_type stride_;
	const _Func fun_;
};

template <typename _ValueType, typename _AccessMode, int_t _Dim, typename _Func>
inline auto make_device_lambda(pointi<_Dim> extent, _Func fun)->cuda::device_lambda_tensor<_ValueType, _AccessMode, _Dim, _Func>{
	return cuda::device_lambda_tensor<_ValueType, _AccessMode, _Dim, _Func>(extent, fun);
}

template <int_t _Dim, typename _Func>
inline auto make_general_lambda(pointi<_Dim> extent, _Func fun)->general_lambda_tensor<_Dim, _Func>{
	return general_lambda_tensor<_Dim, _Func>(extent, fun);
}

inline void barrier() {
	throw_on_error(cudaDeviceSynchronize());
}

template <typename _ValueType, int_t _Dim>
inline void memset(tensor<_ValueType, _Dim> ts, int v) {
	throw_on_error(cudaMemset(ts.shared_data().get(), v, ts.size() * sizeof(_ValueType)));
}

}//cuda

namespace device {

inline void MATAZURE_DEVICE barrier() {
	__syncthreads();
}

} //device

// alias in matazure
template <typename _ValueType, int_t _Dim, typename _Layout = first_major_t>
using cu_tensor = cuda::tensor<_ValueType, _Dim, _Layout>;

template <typename _ValueType, typename _Layout = first_major_t>
using cu_matrix = cu_tensor<_ValueType, 2, _Layout>;

}
