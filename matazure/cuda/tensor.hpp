#pragma once

#include <cuda_runtime.h>
#include <matazure/tensor.hpp>
#include <matazure/cuda/algorithm.hpp>
#include <matazure/cuda/runtime.hpp>

namespace matazure {
namespace cuda {

template <typename _Type, int_t _Rank, typename _Layout = first_major_t>
class tensor : public tensor_expression<tensor<_Type, _Rank, first_major_t>> {
public:
	static_assert(std::is_pod<_Type>::value, "only supports pod type now");

	static const int_t						rank = _Rank;
	typedef _Type							value_type;
	typedef value_type &					reference;
	typedef value_type *					pointer;
	typedef linear_access_t					access_type;
	typedef _Layout							layout_type;
	typedef device_t						memory_type;

	tensor() :
		tensor(pointi<rank>::zeros())
	{}

	template <typename ..._Ext>
	explicit tensor(_Ext... ext) :
		tensor(pointi<rank>{ext...})
	{}

	explicit tensor(pointi<rank> extent) :
		extent_(extent),
		stride_(matazure::accumulate_stride(extent)),
		sp_data_(malloc_shared_memory(stride_[rank - 1])),
		data_(sp_data_.get())
	{ }

	explicit tensor(pointi<rank> extent, std::shared_ptr<value_type> sp_data) :
		extent_(extent),
		stride_(matazure::accumulate_stride(extent)),
		sp_data_(sp_data),
		data_(sp_data_.get())
	{ }

	template <typename _VT>
	tensor(const tensor<_VT, _Rank, _Layout> &ts) :
		extent_(ts.shape()),
		stride_(ts.stride()),
		sp_data_(ts.shared_data()),
		data_(ts.data())
	{ }

	tensor(std::initializer_list<int_t> v) = delete;

	shared_ptr<value_type> shared_data() const { return sp_data_; }

	 template <typename _Idx>
	 MATAZURE_GENERAL reference operator()(_Idx idx) const {
	 	static_assert(std::is_same<_Idx, int_t>::value && rank == 1, "only operator [] support access data by pointi");
	 	return (*this)[pointi<1>{idx}];
	 }

	template <typename ..._Idx>
	MATAZURE_GENERAL reference operator()(_Idx... idx) const {
		return (*this)[pointi<rank>{ idx... }];
	}

	MATAZURE_GENERAL reference operator[](const pointi<rank> &index) const {
		return (*this)[index2offset(index, stride_, layout_type{})];
	}

	MATAZURE_GENERAL reference operator[] (int_t i) const {
		return data_[i];
	}

	MATAZURE_GENERAL pointi<rank> shape() const { return extent_; }
	MATAZURE_GENERAL pointi<rank> stride() const { return stride_; }
	MATAZURE_GENERAL int_t size() const { return stride_[rank - 1]; }

	MATAZURE_GENERAL pointer data() const { return data_; }

private:
	shared_ptr<value_type> malloc_shared_memory(int_t size) {
		decay_t<value_type> *data = nullptr;
		assert_runtime_success(cudaMalloc(&data, size * sizeof(value_type)));
		return shared_ptr<value_type>(data, [](value_type *ptr) {
			assert_runtime_success(cudaFree(const_cast<decay_t<value_type> *>(ptr)));
		});
	}

private:
	const pointi<rank>	extent_;
	const pointi<rank>	stride_;
	const shared_ptr<value_type>	sp_data_;
	const pointer data_;
};

template <typename _Type, typename _BlockDim, typename _Layout = first_major_t>
using block_tensor = tensor<static_tensor<_Type, _BlockDim>, _BlockDim::size(), _Layout>;

template <typename _ValueType, typename _Layout = first_major_t>
using matrix = tensor<_ValueType, 2, _Layout>;
template <typename _ValueType, typename _Layout = first_major_t>
using vector = tensor<_ValueType, 1, _Layout>;

template <typename _TensorSrc, typename _TensorDst>
inline void mem_copy(_TensorSrc ts_src, _TensorDst cts_dst, enable_if_t<!are_host_memory<_TensorSrc, _TensorDst>::value && is_same<typename _TensorSrc::layout_type, typename _TensorDst::layout_type>::value>* = nullptr) {
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_TensorSrc, _TensorDst);

	assert_runtime_success(cudaMemcpy(cts_dst.data(), ts_src.data(), sizeof(typename _TensorDst::value_type) * ts_src.size(), cudaMemcpyDefault));
}

template <typename _TensorSrc, typename _TensorSymbol>
inline void copy_symbol(_TensorSrc src, _TensorSymbol &symbol_dst) {
	assert_runtime_success(cudaMemcpyToSymbol(symbol_dst, src.data(), src.size() * sizeof(typename _TensorSrc::value_type)));
}

template <typename _Reference, typename _AccessMode, int_t _Rank, typename _Func>
class device_lambda_tensor : public tensor_expression<device_lambda_tensor<_Reference, _AccessMode, _Rank, _Func>> {
public:
	static const int_t									rank = _Rank;
	typedef _Reference									reference;
	typedef remove_reference_t<reference>				value_type;
	typedef _AccessMode									accessor_type;
	typedef device_t									memory_type;

public:
	device_lambda_tensor() = delete;

	device_lambda_tensor(const pointi<rank> &extent, _Func fun) :
		extent_(extent),
		stride_(matazure::accumulate_stride(extent)),
		fun_(fun)
	{ }

	 template <typename _Idx>
	 MATAZURE_GENERAL reference operator()(_Idx idx) const {
	 	static_assert(std::is_same<_Idx, int_t>::value && rank == 1, "only operator [] support access data by pointi");
	 	return (*this)[pointi<1>{idx}];
	 }

	template <typename ..._Idx>
	MATAZURE_DEVICE reference operator()(_Idx... idx) const {
		return (*this)[pointi<rank>{ idx... }];
	}

	MATAZURE_DEVICE reference operator[](int_t i) const {
		return offset_imp<accessor_type>(i);
	}

	MATAZURE_DEVICE reference operator[](const pointi<rank> &idx) const {
		return index_imp<accessor_type>(idx);
	}

	template <typename _ExecutionPolicy>
	tensor<decay_t<value_type>, rank> persist(_ExecutionPolicy policy) const {
		tensor<decay_t<value_type>, rank> re(this->shape());
		copy(policy, *this, re);
		return re;
	}

	tensor<decay_t<value_type>, rank> persist() const {
		parallel_execution_policy policy{};
		return persist(policy);
	}

	MATAZURE_GENERAL pointi<rank> shape() const { return extent_; }
	MATAZURE_GENERAL pointi<rank> stride() const { return stride_; }
	MATAZURE_GENERAL int_t size() const { return stride_[rank - 1]; }

private:
	template <typename _Mode>
	MATAZURE_DEVICE enable_if_t<is_same<_Mode, array_access_t>::value, reference>
		index_imp(pointi<rank> index) const {
		return fun_(index);
	}

	template <typename _Mode>
	MATAZURE_DEVICE enable_if_t<is_same<_Mode, linear_access_t>::value, reference>
		index_imp(pointi<rank> index) const {
		return (*this)[index2offset(index, stride())];
	}

	template <typename _Mode>
	MATAZURE_DEVICE enable_if_t<is_same<_Mode, array_access_t>::value, reference>
		offset_imp(int_t i) const {
		return (*this)(offset2index(i, stride()));
	}

	template <typename _Mode>
	MATAZURE_DEVICE enable_if_t<is_same<_Mode, linear_access_t>::value, reference>
		offset_imp(int_t i) const {
		return fun_(i);
	}

private:
	const pointi<rank> extent_;
	const pointi<rank> stride_;
	const _Func fun_;
};

template <int_t _Rank, typename _Func>
class general_lambda_tensor : public tensor_expression<general_lambda_tensor<_Rank, _Func>> {
	typedef function_traits<_Func>						functor_traits;
public:
	static const int_t										rank = _Rank;
	typedef typename functor_traits::result_type			reference;
	typedef remove_reference_t<reference>					value_type;
	typedef typename matazure::internal::get_functor_accessor_type<_Rank, _Func>::type		access_type;
	typedef device_t										memory_type;


public:
	general_lambda_tensor(const pointi<rank> &extent, _Func fun) :
		extent_(extent),
		stride_(matazure::accumulate_stride(extent)),
		fun_(fun)
	{}

	MATAZURE_GENERAL reference operator[](int_t i) const {
		return offset_imp<access_type>(i);
	}

	MATAZURE_GENERAL reference operator[](const pointi<rank> &idx) const {
		return index_imp<access_type>(idx);
	}

	template <typename ..._Idx>
	MATAZURE_GENERAL reference operator()(_Idx... idx) const {
		return (*this)[pointi<rank>{ idx... }];
	}

	template <typename _ExecutionPolicy>
	tensor<decay_t<value_type>, rank> persist(_ExecutionPolicy policy) const {
		tensor<decay_t<value_type>, rank> re(this->shape());
		cuda::copy(policy, *this, re);
		return re;
	}

	tensor<decay_t<value_type>, rank> persist() const {
		parallel_execution_policy policy{};
		return persist(policy);
	}

	MATAZURE_GENERAL pointi<rank> shape() const { return extent_; }
	MATAZURE_GENERAL pointi<rank> stride() const { return stride_; }
	MATAZURE_GENERAL int_t size() const { return stride_[rank - 1]; }

public:
	template <typename _Mode>
	MATAZURE_GENERAL enable_if_t<is_same<_Mode, array_access_t>::value, reference> index_imp(pointi<rank> index) const {
		return fun_(index);
	}

	template <typename _Mode>
	MATAZURE_GENERAL enable_if_t<is_same<_Mode, linear_access_t>::value, reference> index_imp(pointi<rank> index) const {
		return (*this)[index2offset(index, stride(), first_major_t{})];
	}

	template <typename _Mode>
	MATAZURE_GENERAL enable_if_t<is_same<_Mode, array_access_t>::value, reference> offset_imp(int_t i) const {
		return (*this)(offset2index(i, stride(), first_major_t{}));
	}

	template <typename _Mode>
	MATAZURE_GENERAL enable_if_t<is_same<_Mode, linear_access_t>::value, reference> offset_imp(int_t i) const {
		return fun_(i);
	}

private:
	const pointi<rank> extent_;
	const pointi<rank> stride_;
	const _Func fun_;
};

template <typename _ValueType, typename _AccessMode, int_t _Rank, typename _Func>
inline auto make_device_lambda(pointi<_Rank> extent, _Func fun)->cuda::device_lambda_tensor<_ValueType, _AccessMode, _Rank, _Func>{
	return cuda::device_lambda_tensor<_ValueType, _AccessMode, _Rank, _Func>(extent, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_general_lambda(pointi<_Rank> extent, _Func fun)->general_lambda_tensor<_Rank, _Func>{
	return general_lambda_tensor<_Rank, _Func>(extent, fun);
}

inline void device_synchronize() {
	assert_runtime_success(cudaDeviceSynchronize());
}

template <typename _ValueType, int_t _Rank>
inline void memset(tensor<_ValueType, _Rank> ts, int v) {
	assert_runtime_success(cudaMemset(ts.shared_data().get(), v, ts.size() * sizeof(_ValueType)));
}

namespace device {

inline void MATAZURE_DEVICE barrier() {
	__syncthreads();
}

} //device

template <typename _Type, int_t _Rank, typename _Layout>
inline tensor<_Type, _Rank, _Layout> mem_clone(tensor<_Type, _Rank, _Layout> ts, device_t) {
	tensor<_Type, _Rank, _Layout> ts_re(ts.shape());
	mem_copy(ts, ts_re);
	return ts_re;
}

template <typename _Type, int_t _Rank, typename _Layout>
inline tensor<_Type, _Rank, _Layout> mem_clone(tensor<_Type, _Rank, _Layout> ts) {
	return mem_clone(ts, device_t{});
}

template <typename _Type, int_t _Rank, typename _Layout>
inline tensor<_Type, _Rank, _Layout> mem_clone(matazure::tensor<_Type, _Rank, _Layout> ts, device_t) {
	tensor<_Type, _Rank, _Layout> ts_re(ts.shape());
	mem_copy(ts, ts_re);
	return ts_re;
}

template <typename _Type, int_t _Rank, typename _Layout>
inline matazure::tensor<_Type, _Rank, _Layout> mem_clone(tensor<_Type, _Rank, _Layout> ts, host_t) {
	matazure::tensor<_Type, _Rank, _Layout> ts_re(ts.shape());
	mem_copy(ts, ts_re);
	return ts_re;
}

template <typename _ValueType, int_t _Rank, typename _Layout, int_t _OutDim, typename _OutLayout = _Layout>
inline auto reshape(cuda::tensor<_ValueType, _Rank, _Layout> ts, pointi<_OutDim> ext, _OutLayout* = nullptr)->cuda::tensor<_ValueType, _OutDim, _OutLayout>{
	///TODO: assert size equal
	cuda::tensor<_ValueType, _OutDim, _OutLayout> re(ext, ts.shared_data());
	return re;
}

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

}


}//cuda

using cuda::mem_copy;
using cuda::mem_clone;
using cuda::reshape;

}
