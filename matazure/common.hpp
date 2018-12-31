#pragma once

#include <matazure/tensor.hpp>
#include <matazure/algorithm.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#endif

#include <matazure/unary.hpp>

namespace matazure {

#ifndef MATAZURE_CUDA

/**
* @brief make a lambda_tensor
* @param the shape
* @param the functor, a index -> value pattern
*/
template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> extent, _Func fun)->lambda_tensor<_Rank, _Func>{
	return lambda_tensor<_Rank, _Func>(extent, fun);
}

/**
* @brief make a lambda_tensor
* @param the shape
* @param the functor, a index -> value pattern
*/
template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> extent, _Func fun, host_tag)->lambda_tensor<_Rank, _Func>{
	return lambda_tensor<_Rank, _Func>(extent, fun);
}

/**
* @brief make a lambda_tensor
* @param the shape
* @param the functor, a index -> value pattern
*/
template <int_t _Rank, typename _Func>
inline auto make_host_lambda(pointi<_Rank> extent, _Func fun)->lambda_tensor<_Rank, _Func>{
	return lambda_tensor<_Rank, _Func>(extent, fun);
}

#else

/**
* @brief make a lambda_tensor
* @param the shape
* @param the functor, a index -> value pattern
*/
template <int_t _Rank, typename _Func>
inline auto make_host_lambda(pointi<_Rank> extent, _Func fun)->lambda_tensor<_Rank, _Func>{
	return lambda_tensor<_Rank, _Func>(extent, fun);
}

/**
* @brief make a lambda_tensor
* @param the shape
* @param the functor, should be a host functor a index -> value pattern
*/
template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func) && !MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(make_host_lambda(ext, fun)){
	return make_host_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, host_tag, enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func) && !MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(make_host_lambda(ext, fun)) {
	return make_host_lambda(ext, fun);
}

/**
* @todo: not support device struct operator, it's diffcult to do this.
*/
template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, device_tag, enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func) && !MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(cuda::make_general_lambda(ext, fun)) {
	return cuda::make_general_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, enable_if_t<MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(cuda::make_general_lambda(ext, fun)) {
	return cuda::make_general_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, device_tag, enable_if_t<MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(cuda::make_general_lambda(ext, fun)) {
	return cuda::make_general_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, host_tag, enable_if_t<MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(make_host_lambda(ext, fun)) {
	return make_host_lambda(ext, fun);
}

template <typename _ValueType, typename _Access, int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, enable_if_t<MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(cuda::make_device_lambda<_ValueType, _Access>(ext, fun)) {
	return cuda::make_device_lambda<_ValueType, _Access>(ext, fun);
}

template <typename _ValueType, typename _Access, int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, device_tag, enable_if_t<MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(cuda::make_device_lambda<_ValueType, _Access>(ext, fun)) {
	return cuda::make_device_lambda<_ValueType, _Access>(ext, fun);
}

#endif

namespace internal{

template <typename _Tensor, typename _Func>
struct linear_map_op {
private:
	const _Tensor ts_;
	const _Func fun_;

public:
	linear_map_op(_Tensor ts, _Func fun) : ts_(ts), fun_(fun) {}

	MATAZURE_GENERAL auto operator()(int_t i) const->decltype(this->fun_(this->ts_[i])){
		return fun_(ts_[i]);
	}
};

template <typename _Tensor, typename _Func>
struct array_map_op {
private:
	const _Tensor ts_;
	const _Func fun_;

public:
	array_map_op(_Tensor ts, _Func fun) : ts_(ts), fun_(fun) {}

	MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const->decltype(this->fun_(this->ts_[idx])) {
		return fun_(ts_[idx]);
	}
};

template <typename _Tensor, typename _Func>
struct device_linear_map_op {
private:
	const _Tensor ts_;
	const _Func fun_;

public:
	device_linear_map_op(_Tensor ts, _Func fun) : ts_(ts), fun_(fun) {}

	MATAZURE_DEVICE auto operator()(int_t i) const->decltype(this->fun_(this->ts_[i])) {
		return fun_(ts_[i]);
	}
};

template <typename _Tensor, typename _Func>
struct device_array_map_op {
private:
	const _Tensor ts_;
	const _Func fun_;

public:
	device_array_map_op(_Tensor ts, _Func fun) : ts_(ts), fun_(fun) {}

	MATAZURE_DEVICE auto operator()(pointi<_Tensor::rank> idx) const->decltype(this->fun_(this->ts_[idx])) {
		return fun_(ts_[idx]);
	}
};

template <typename _OutValueType>
struct cast_op {
	template <typename _InValueType>
	MATAZURE_GENERAL _OutValueType operator()(_InValueType v) const {
		return static_cast<_OutValueType>(v);
	}
};

template <typename _OutPointValueType, int_t _Rank>
struct cast_op<point<_OutPointValueType, _Rank>>{
	template <typename _InPointValueType>
	MATAZURE_GENERAL point<_OutPointValueType, _Rank> operator() (const point<_InPointValueType, _Rank> &p) const {
		return point_cast<_OutPointValueType>(p);
	}
};

template <typename _Tensor>
struct section_op {
private:
	_Tensor ts_;
	pointi<_Tensor::rank> offset_;

public:
	section_op(_Tensor ts, pointi<_Tensor::rank> offset) :
		ts_(ts), offset_(offset)
	{}

	MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const->decltype((ts_[idx + offset_])) {
		return ts_[idx + offset_];
	}
};


template <typename _Tensor, typename _StrideType>
struct stride_op {
private:
	_Tensor ts_;
	_StrideType stride_;
public:
	stride_op(_Tensor ts, _StrideType stride):
		ts_(ts), stride_(stride)
	{}

	MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const->decltype((ts_[idx * stride_])){
		return ts_[idx * stride_];
	}
};

template <typename _Tensor>
struct resize_op{
private:
	_Tensor ts_;
	pointi<_Tensor::rank> resize_ext_;
	pointf<_Tensor::rank> resize_scale_;
public:
	resize_op(_Tensor ts, pointi<_Tensor::rank> resize_ext): ts_(ts), resize_ext_(resize_ext) {
		resize_scale_ = point_cast<float>(ts_.shape()) / point_cast<float>(resize_ext_);
	}

	MATAZURE_GENERAL typename _Tensor::value_type operator()(const pointi<_Tensor::rank> &idx) const{
		auto idx_f = point_cast<float>(idx) * resize_scale_;
		return ts_[point_cast<int_t>(idx_f)];
	}
};

template <typename ..._Tensors>
struct zip_op;

template <typename _Tensor0, typename _Tensor1>
struct zip_op<_Tensor0, _Tensor1>{
private:
	_Tensor0 ts0_;
	_Tensor1 ts1_;

public:
	zip_op(_Tensor0 ts0, _Tensor1 ts1): ts0_(ts0), ts1_(ts1){}

	MATAZURE_GENERAL auto operator()(int_t i) const->decltype(tie(ts0_[0], ts1_[0])){
		return tie(ts0_[i], ts1_[i]);
	}
};

template <typename _Tensor0, typename _Tensor1, typename _Tensor2>
struct zip_op<_Tensor0, _Tensor1, _Tensor2>{
private:
	_Tensor0 ts0_;
	_Tensor1 ts1_;
	_Tensor2 ts2_;
public:
	zip_op(_Tensor0 ts0, _Tensor1 ts1, _Tensor2 ts2): ts0_(ts0), ts1_(ts1), ts2_(ts2){}

	MATAZURE_GENERAL auto operator()(int_t i) const->decltype(tie(ts0_[0], ts1_[0], ts2_[0])){
		return tie(ts0_[i], ts1_[i], ts2_[i]);
	}
};

//pointi<2>
template <int_t _SliceDimIdx>
inline pointi<1> slice_point(pointi<2> pt);

template < >
inline pointi<1> slice_point<0>(pointi<2> pt){
	return pointi<1>{get<1>(pt)};
}

template < >
inline pointi<1> slice_point<1>(pointi<2> pt){
	return pointi<1>{get<0>(pt)};
}

template <int_t _CatDimIdx>
inline pointi<2> cat_point(pointi<1> pt, int_t cat_i);

template <>
inline pointi<2> cat_point<0>(pointi<1> pt, int_t cat_i){
	return pointi<2>{cat_i, get<0>(pt)};
}

template <>
inline pointi<2> cat_point<1>(pointi<1> pt, int_t cat_i){
	return pointi<2>{get<0>(pt), cat_i};
}

template <typename _Tensor, int_t _SliceDimIdx>
struct slice_op {
private:
	_Tensor ts_;
	int_t slice_i_;
public:
	slice_op(_Tensor ts, int_t slice_i):
		ts_(ts), slice_i_(slice_i)
	{}

	MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank-1> idx) const->decltype((ts_[cat_point<_SliceDimIdx>(idx, slice_i_)])){
		return ts_[cat_point<_SliceDimIdx>(idx, slice_i_)];
	}
};

template <typename _Tensor>
struct clamp_zero_op {
private:
	_Tensor ts_;

public:
	clamp_zero_op(_Tensor ts) :
		ts_(ts)
	{}

	MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const->decltype(zero<decay_t<typename _Tensor::value_type>>::value()) {
		if (MATAZURE_LIKELY(inside_range(idx, pointi<_Tensor::rank>::zeros(), ts_.shape()))) {
			return ts_[idx];
		}
		else {
			return zero<typename _Tensor::value_type>::value();
		}
	}
};

template <typename _Tensor>
struct global_view_op{
private:
	_Tensor ts_;
	typedef typename _Tensor::value_type block_type;
public:
	global_view_op(_Tensor ts) : ts_(ts) {}

	MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const->decltype((ts_[0][0])){
		auto block_dim = meta::array_to_pointi(block_type::meta_shape());
		auto block_idx = idx / block_dim;
		auto local_idx = idx % block_dim;
		return ts_[block_idx][local_idx];
	}
};

}

#ifndef MATAZURE_CUDA

/**
* @brief apply the functor for each element of a linear indexing tensor
* @param ts the source tensor
* @param fun the functor, element -> value  pattern
*/
template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<linear_index, typename _Tensor::index_type>::value>* = 0)->decltype(make_lambda(ts.shape(), internal::linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

/**
* @brief apply the functor for each element of a array indexing tensor
* @param ts the source tensor
* @param fun the functor, element -> value  pattern
*/
template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<array_index, typename _Tensor::index_type>::value>* = 0)->decltype(make_lambda(ts.shape(), internal::array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

#else

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<linear_index, typename _Tensor::index_type>::value>* = 0, enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(make_lambda(ts.shape(), internal::linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<array_index, typename _Tensor::index_type>::value>* = 0, enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(make_lambda(ts.shape(), internal::array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<linear_index, typename _Tensor::index_type>::value>* = 0, enable_if_t<MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(make_lambda(ts.shape(), internal::device_linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::device_linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<array_index, typename _Tensor::index_type>::value>* = 0, enable_if_t<MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(make_lambda(ts.shape(), internal::device_array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::device_array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

#endif

/**
* @brief cast a tensor to another value_type lambda_tensor
*
* support primitive type static_cast and point_cast.
*
* @param tensor the source tensor
* @tparam _ValueType the dest tensor value type
* @return a lambda_tensor whose value_type is _ValueType
*/
template <typename _ValueType, typename _Tensor>
inline auto cast(_Tensor tensor, enable_if_t<is_tensor<_Tensor>::value>* = 0)->decltype(apply(tensor, internal::cast_op<_ValueType>())) {
	return apply(tensor, internal::cast_op<_ValueType>());
}

///**
//* @brief saturate cast a tensor to another value_type lambda_tensor
//* @param tensor the source tensor
//* @tparam _ValueType the dest tensor value type
//* @return a lambda_tensor whose value_type is _ValueType
//*/
//template <typename _ValueType, typename _Tensor>
//inline auto saturate_cast(_Tensor tensor, enable_if_t<is_tensor<_Tensor>>* = 0)->decltype(apply(tensor, internal::cast_op<_ValueType>())) {
//	// typedef point<byte, 3> (* sature_cast_op)(const point<float, 3> &);
//	// sature_cast_op pointf3_to_pointb3 = &unary::saturate_cast<byte, float, 3>;
//	return apply(tensor, internal::cast_op<_ValueType>());
//}

/**
* @brief produces a subsection lambda_tensor of the source tensor
* @param ts the source tensor
* @param origin the origin of the section
* @param ext the extent of the section
* @return a subsection lambda_tensor
*/
template <typename _Tensor>
inline auto section(_Tensor ts, pointi<_Tensor::rank> origin, pointi<_Tensor::rank> ext)->decltype(make_lambda(ext, internal::section_op<decay_t<_Tensor>>(ts, origin), typename _Tensor::memory_type{})) {
	return make_lambda(ext, internal::section_op<decay_t<_Tensor>>(ts, origin), typename _Tensor::memory_type{});
}

/**
* @brief produces a stride indexing lambda_tensor of the source tensor
* @param ts the source tensor
* @param stride the stride of the indexing
* @return a stride indexing lambda_tensor
*/
template <typename _Tensor, typename _StrideType>
inline auto stride(_Tensor ts, _StrideType stride)->decltype(make_lambda(ts.shape() / stride, internal::stride_op<_Tensor, _StrideType>(ts, stride), typename _Tensor::memory_type{})) {
	return make_lambda(ts.shape() / stride, internal::stride_op<_Tensor, _StrideType>(ts, stride), typename _Tensor::memory_type{});
}

/**
* @brief produces a resized lambda_tensor of the source tensor
* @param ts the source tensor
* @resize_ext the resized shape
* @return a resized lambda_tensor
*/
template <typename _Tensor>
inline auto resize(_Tensor ts, const pointi<_Tensor::rank> &resize_ext)->decltype(make_lambda(resize_ext, internal::resize_op<decay_t<_Tensor>>(ts, resize_ext), typename _Tensor::memory_type{})) {
	return make_lambda(resize_ext, internal::resize_op<decay_t<_Tensor>>(ts, resize_ext), typename _Tensor::memory_type{});
}

/**
* @brief produces a sub-dim lambda_tensor of the source tensor
* @param ts the source tensor
* @tparam _DimIdx the sliced dim(orientation) index
* @param i the slice position index on the sliced dim(orientation)
* @return a sub-dim lambda_tensor
*/
template <int_t _DimIdx, typename _Tensor>
inline auto slice(_Tensor ts, int_t positon_index)->decltype(make_lambda(internal::slice_point<_DimIdx>(ts.shape()), internal::slice_op<_Tensor, _DimIdx>(ts, positon_index), typename _Tensor::memory_type{})){
	return make_lambda(internal::slice_point<_DimIdx>(ts.shape()), internal::slice_op<_Tensor, _DimIdx>(ts, positon_index), typename _Tensor::memory_type{});
}

#ifdef MATAZURE_CUDA

/// special for slice<rank-1>(tensor<_T, rank>, position_index), it produces a cuda::tensor<_T, rank-1>
template <int_t _DimIdx, typename _T, int_t _Rank, typename _Layout>
inline auto slice(cuda::tensor<_T, _Rank, _Layout> ts, int_t positon_index, enable_if_t<_DimIdx == _Rank-1>* = nullptr)->cuda::tensor<_T, _Rank-1, _Layout>{
	auto slice_ext = internal::slice_point<_DimIdx>(ts.shape());
	auto slice_size = cumulative_prod(slice_ext)[_Rank-1];
	cuda::tensor<_T, _Rank-1, _Layout> ts_re(slice_ext, shared_ptr<_T>(ts.shared_data().get() + positon_index * slice_size, [ts](_T *){ }));
	return ts_re;
}

#endif

/**
* @brief procudes a clamped indexing lambda_tensor from the source tensor.
* @param ts the source tensor
* @return a clamped indexing lambda_tensor
*/
template <typename _Tensor>
inline auto clamp_zero(_Tensor ts)->decltype(make_lambda(ts.shape(), internal::clamp_zero_op<decay_t<_Tensor>>(ts), typename _Tensor::memory_type{})) {
	return make_lambda(ts.shape(), internal::clamp_zero_op<decay_t<_Tensor>>(ts), typename _Tensor::memory_type{});
}

/**
* @todo
*/
template <typename _Tensor>
inline auto global_view(_Tensor ts)->decltype(make_lambda(ts.shape() * ts[0].shape(), internal::global_view_op<decay_t<_Tensor>>(ts), typename _Tensor::memory_type{})){
	auto block_dim = meta::array_to_pointi(_Tensor::value_type::meta_shape());
	return make_lambda(ts.shape() * block_dim, internal::global_view_op<decay_t<_Tensor>>(ts), typename _Tensor::memory_type{});
}

/**
* @brief  zip the source tensors
* @param ts0 the first source tensor
* @param ts1 the second source tensor
* @return a zipped lambda_tensor whose value_type is tuple<_Tensor0::value_type &, _Tensor1::value_type &>
* @todo maybe exits a better implement
*/
template <typename _Tensor0, typename _Tensor1>
inline auto zip(_Tensor0 ts0, _Tensor1 ts1)->decltype(make_lambda(ts0.shape(), internal::zip_op<_Tensor0, _Tensor1>(ts0, ts1))) {
	MATAZURE_ASSERT(equal(ts0.shape(), ts1.shape()), "unmatched shape");
	return make_lambda(ts0.shape(), internal::zip_op<_Tensor0, _Tensor1>(ts0, ts1));
}

/**
*
*/
template <typename _Tensor0, typename _Tensor1, typename _Tensor2>
inline auto zip(_Tensor0 ts0, _Tensor1 ts1, _Tensor2 ts2)->decltype(make_lambda(ts0.shape(), internal::zip_op<_Tensor0, _Tensor1, _Tensor2>(ts0, ts1, ts2))) {
	MATAZURE_ASSERT(equal(ts0.shape(), ts1.shape()), "unmatched shape");
	MATAZURE_ASSERT(equal(ts2.shape(), ts1.shape()), "unmatched shape");
	return make_lambda(ts0.shape(), internal::zip_op<_Tensor0, _Tensor1, _Tensor2>(ts0, ts1, ts2));
}

//TODO: use packed parameters
// template <typename ..._Tensors>
// inline auto zip(_Tensors... tensors){
// 	auto tuple_tensors = make_tuple(tensors);
// 	auto ext = get<0>(tuple_tensors).shape();
// 	return make_lambda(ext, [=](int_t i){
// 		return tie(tensors...[i]);
// 	});
// }

/**
* @brief view a zipped source tensor as point value_type
* @param ts the source type
* @returns  a lambda_tensor whose value_type is point_viewer
*/
template <typename _Tensor>
inline auto point_view(_Tensor ts)->decltype(cast<point_viewer<decltype(ts[0])>>(ts)){
	return cast<point_viewer<decltype(ts[0])>>(ts);
}

}
