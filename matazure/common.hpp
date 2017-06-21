#pragma once

#include <matazure/tensor.hpp>
#include <matazure/algorithm.hpp>

#ifdef MATAZURE_CUDA
#include <matazure/cuda/tensor.hpp>
#define MATAZURE_IS_D_LAMBDA(X) __nv_is_extended_device_lambda_closure_type(X)
#define MATAZURE_IS_HD_LAMBDA(X) __nv_is_extended_host_device_lambda_closure_type(X)
#endif

namespace matazure {

#ifndef MATAZURE_CUDA

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> extent, _Func fun)->lambda_tensor<_Rank, _Func>{
	return lambda_tensor<_Rank, _Func>(extent, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> extent, _Func fun, host_t)->lambda_tensor<_Rank, _Func>{
	return lambda_tensor<_Rank, _Func>(extent, fun);
}

#else

template <int_t _Rank, typename _Func>
inline auto make_host_lambda(pointi<_Rank> extent, _Func fun)->lambda_tensor<_Rank, _Func>{
	return lambda_tensor<_Rank, _Func>(extent, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func) && !MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(make_host_lambda(ext, fun)){
	return make_host_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, host_t, enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func) && !MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(make_host_lambda(ext, fun)) {
	return make_host_lambda(ext, fun);
}

///TODO: not support device struct operator
template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, device_t, enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func) && !MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(cuda::make_general_lambda(ext, fun)) {
	return cuda::make_general_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, enable_if_t<MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(cuda::make_general_lambda(ext, fun)) {
	return cuda::make_general_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, device_t, enable_if_t<MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(cuda::make_general_lambda(ext, fun)) {
	return cuda::make_general_lambda(ext, fun);
}

template <int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, host_t, enable_if_t<MATAZURE_IS_HD_LAMBDA(_Func)>* = nullptr)->decltype(make_host_lambda(ext, fun)) {
	return make_host_lambda(ext, fun);
}

template <typename _ValueType, typename _Access, int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, enable_if_t<MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(cuda::make_device_lambda<_ValueType, _Access>(ext, fun)) {
	return cuda::make_device_lambda<_ValueType, _Access>(ext, fun);
}

template <typename _ValueType, typename _Access, int_t _Rank, typename _Func>
inline auto make_lambda(pointi<_Rank> ext, _Func fun, device_t, enable_if_t<MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(cuda::make_device_lambda<_ValueType, _Access>(ext, fun)) {
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
struct shift_op {
private:
	_Tensor ts_;
	pointi<_Tensor::rank> offset_;

public:
	shift_op(_Tensor ts, pointi<_Tensor::rank> offset):
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
		auto idx_f = point_cast<float>[idx] * resize_scale_;
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

//pointi<3>
template <int_t _SliceDimIdx>
inline pointi<2> slice_point(pointi<3> pt);

template < >
inline pointi<2> slice_point<0>(pointi<3> pt){
	return pointi<2>{get<1>(pt), get<2>(pt)};
}

template < >
inline pointi<2> slice_point<1>(pointi<3> pt){
	return pointi<2>{get<0>(pt), get<2>(pt)};
}

template < >
inline pointi<2> slice_point<2>(pointi<3> pt){
	return pointi<2>{get<0>(pt), get<1>(pt)};
}

template <int_t _CatDimIdx>
inline pointi<3> cat_point(pointi<2> pt, int_t cat_i);

template <>
inline pointi<3> cat_point<0>(pointi<2> pt, int_t cat_i){
	return pointi<3>{cat_i, get<0>(pt), get<1>(pt)};
}

template <>
inline pointi<3> cat_point<1>(pointi<2> pt, int_t cat_i){
	return pointi<3>{get<0>(pt), cat_i, get<1>(pt)};
}

template <>
inline pointi<3> cat_point<2>(pointi<2> pt, int_t cat_i){
	return pointi<3>{get<0>(pt), get<1>(pt), cat_i};
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
struct padding_zero_op {
private:
	_Tensor ts_;
	pointi<_Tensor::rank> padding0_;
	pointi<_Tensor::rank> padding1_;

public:
	padding_zero_op(_Tensor ts, pointi<_Tensor::rank> padding0, pointi<_Tensor::rank> padding1) :
		ts_(ts),
		padding0_(padding0),
		padding1_(padding1)
	{}

	MATAZURE_GENERAL auto operator()(pointi<_Tensor::rank> idx) const->decltype(zero<typename _Tensor::value_type>::value()) {
		if (MATAZURE_LIKELY(inside(idx, padding0_, ts_.shape()))) {
			return ts_[idx - padding0_];
		}
		else {
			return zero<typename _Tensor::value_type>::value();
		}
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
		if (MATAZURE_LIKELY(inside(idx, pointi<_Tensor::rank>::zeros(), ts_.shape()))) {
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

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<linear_access_t, typename _Tensor::access_type>::value>* = 0)->decltype(make_lambda(ts.shape(), internal::linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<array_access_t, typename _Tensor::access_type>::value>* = 0)->decltype(make_lambda(ts.shape(), internal::array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

#else

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<linear_access_t, typename _Tensor::access_type>::value>* = 0, enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(make_lambda(ts.shape(), internal::linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<array_access_t, typename _Tensor::access_type>::value>* = 0, enable_if_t<!MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(make_lambda(ts.shape(), internal::array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<linear_access_t, typename _Tensor::access_type>::value>* = 0, enable_if_t<MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(make_lambda(ts.shape(), internal::device_linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::device_linear_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

template <typename _Tensor, typename _Func>
inline auto apply(_Tensor ts, _Func fun, enable_if_t<is_same<array_access_t, typename _Tensor::access_type>::value>* = 0, enable_if_t<MATAZURE_IS_D_LAMBDA(_Func)>* = nullptr)->decltype(make_lambda(ts.shape(), internal::device_array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{}))
{
	return make_lambda(ts.shape(), internal::device_array_map_op<_Tensor, _Func>(ts, fun), typename _Tensor::memory_type{});
}

#endif

template <typename _ValueType, typename _Tensor>
inline auto tensor_cast(_Tensor tensor)->decltype(apply(tensor, internal::cast_op<_ValueType>())) {
	return apply(tensor, internal::cast_op<_ValueType>());
}

template <typename _Tensor>
inline auto shift(_Tensor ts, pointi<_Tensor::rank> offset)->decltype(make_lambda(ts.shape(), internal::shift_op<_Tensor>(ts, offset), typename _Tensor::memory_type{})) {
	return make_lambda(ts.shape(), internal::shift_op<_Tensor>(ts, offset), typename _Tensor::memory_type{});
}

template <typename _Tensor>
inline auto section(_Tensor ts, pointi<_Tensor::rank> origin, pointi<_Tensor::rank> ext)->decltype(make_lambda(ext, internal::shift_op<_Tensor>(ts, origin), typename _Tensor::memory_type{})) {
	return make_lambda(ext, internal::shift_op<_Tensor>(ts, origin), typename _Tensor::memory_type{});
}

template <typename _Tensor, typename _StrideType>
inline auto stride(_Tensor ts, _StrideType stride)->decltype(make_lambda(ts.shape() / stride, internal::stride_op<_Tensor, _StrideType>(ts, stride), typename _Tensor::memory_type{})) {
	return make_lambda(ts.shape() / stride, internal::stride_op<_Tensor, _StrideType>(ts, stride), typename _Tensor::memory_type{});
}

///TODO: assert range out
template <int_t _DimIdx, typename _Tensor>
inline auto slice(_Tensor ts, int_t i)->decltype(make_lambda(internal::slice_point<_DimIdx>(ts.shape()), internal::slice_op<_Tensor, _DimIdx>(ts, i), typename _Tensor::memory_type{})){
	return make_lambda(internal::slice_point<_DimIdx>(ts.shape()), internal::slice_op<_Tensor, _DimIdx>(ts, i), typename _Tensor::memory_type{});
}

template <int_t _DimIdx, typename _T, int_t _Rank, typename _Layout>
inline auto slice(tensor<_T, _Rank, _Layout> ts, int_t i, enable_if_t<_DimIdx == _Rank-1>* = nullptr)->tensor<_T, _Rank-1, _Layout>{
	auto slice_ext = internal::slice_point<_DimIdx>(ts.shape());
	auto slice_size = prod(slice_ext);
	tensor<_T, _Rank-1, _Layout> ts_re(slice_ext, shared_ptr<_T>(ts.shared_data().get() + i * slice_size, [ts](_T *){ }));
	return ts_re;
}

#ifdef MATAZURE_CUDA

template <int_t _DimIdx, typename _T, int_t _Rank, typename _Layout>
inline auto slice(cuda::tensor<_T, _Rank, _Layout> ts, int_t i, enable_if_t<_DimIdx == _Rank-1>* = nullptr)->cuda::tensor<_T, _Rank-1, _Layout>{
	auto slice_ext = internal::slice_point<_DimIdx>(ts.shape());
	auto slice_size = prod(slice_ext);
	cuda::tensor<_T, _Rank-1, _Layout> ts_re(slice_ext, shared_ptr<_T>(ts.shared_data().get() + i * slice_size, [ts](_T *){ }));
	return ts_re;
}

#endif

template <typename _Tensor>
inline auto padding_zero(_Tensor ts, pointi<_Tensor::rank> padding0, pointi<_Tensor::rank> padding1)->decltype(make_lambda(ts.shape() + padding0 + padding1, internal::padding_zero_op<_Tensor>(ts, padding0, padding1), typename _Tensor::memory_type{})) {
	return make_lambda(ts.shape() + padding0 + padding1, internal::padding_zero_op<_Tensor>(ts, padding0, padding1), typename _Tensor::memory_type{});
}

template <typename _Tensor>
inline auto clamp_zero(_Tensor ts)->decltype(make_lambda(ts.shape(), internal::clamp_zero_op<_Tensor>(ts), typename _Tensor::memory_type{})) {
	return make_lambda(ts.shape(), internal::clamp_zero_op<_Tensor>(ts), typename _Tensor::memory_type{});
}

template <typename _Tensor>
inline auto global_view(_Tensor ts)->decltype(make_lambda(ts.shape() * ts[0].shape(), internal::global_view_op<_Tensor>(ts), typename _Tensor::memory_type{})){
	auto block_dim = meta::array_to_pointi(_Tensor::value_type::meta_shape());
	return make_lambda(ts.shape() * block_dim, internal::global_view_op<_Tensor>(ts), typename _Tensor::memory_type{});
}

template <typename _Tensor>
inline auto resize(_Tensor ts, const pointi<_Tensor::rank> &resize_ext)->decltype(make_lambda(resize_ext, internal::resize_op<_Tensor>(ts, resize_ext), typename _Tensor::memory_type{})) {
	return make_lambda(resize_ext, internal::resize_op<_Tensor>(ts, resize_ext), typename _Tensor::memory_type{});
}

template <typename _Tensor0, typename _Tensor1>
inline auto zip(_Tensor0 ts0, _Tensor1 ts1)->decltype(make_lambda(ts0.shape(), internal::zip_op<_Tensor0, _Tensor1>(ts0, ts1))) {
	MATAZURE_ASSERT(equal(ts0.shape(), ts1.shape()), "unmatched shape");
	return make_lambda(ts0.shape(), internal::zip_op<_Tensor0, _Tensor1>(ts0, ts1));
}

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

template <typename _Tensor>
inline auto point_view(_Tensor ts)->decltype(tensor_cast<point_viewer<decltype(ts[0])>>(ts)){
	return tensor_cast<point_viewer<decltype(ts[0])>>(ts);
}

#define __MATAZURE_LINEAR_ACCESS_TENSOR_BINARY_OPERATOR(name, op) \
template <typename _T1, typename _T2> \
struct name { \
private: \
	_T1 x1_; \
	_T2 x2_; \
\
public:\
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_T1, _T2); \
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_T1, _T2); \
\
	MATAZURE_GENERAL name(_T1 x1, _T2 x2) : x1_(x1), x2_(x2) {} \
\
	MATAZURE_GENERAL auto operator ()(int_t i) const->decltype(this->x1_[i] op this->x2_[i]){ \
		return x1_[i] op x2_[i]; \
	} \
};

#define __MATAZURE_ARRAY_ACCESS_TENSOR_BINARY_OPERATOR(name, op) \
template <typename _T1, typename _T2> \
struct name { \
private: \
	_T1 x1_; \
	_T2 x2_; \
\
public:\
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_T1, _T2); \
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_T1, _T2); \
	MATAZURE_GENERAL name(_T1 x1, _T2 x2) : x1_(x1), x2_(x2) {} \
\
	MATAZURE_GENERAL auto operator ()(const pointi<_T1::rank> &idx) const->decltype(this->x1_[idx] op this->x2_[idx]){ \
		return x1_[idx] op x2_[idx]; \
	} \
};

#define __MATAZURE_LINEAR_ACCESS_TENSOR_WITH_VALUE_BINARY_OPERATOR(name, op) \
template <typename _T> \
struct name { \
private: \
	typedef typename _T::value_type value_type; \
\
	_T x_; \
	value_type v_; \
\
public:\
	name(_T x, value_type v) : x_(x), v_(v) {} \
\
MATAZURE_GENERAL auto operator()(int_t i) const->decltype(this->x_[i] op this->v_){ \
		return x_[i] op v_; \
	} \
\
};

#define __MATAZURE_ARRAY_ACCESS_TENSOR_WITH_VALUE_BINARY_OPERATOR(name, op) \
template <typename _T> \
struct name { \
private: \
	typedef typename _T::value_type value_type; \
\
	_T x_; \
	value_type v_; \
\
public:\
	name(_T x, value_type v) : x_(x), v_(v) {} \
\
	MATAZURE_GENERAL auto operator ()(const pointi<_T::rank> &idx) const->decltype(this->x_[idx] op this->v_){ \
		return x_[idx] op v_; \
	} \
};

#define __MATAZURE_VALUE_WITH_LINEAR_ACCESS_TENSOR_BINARY_OPERATOR(name, op) \
template <typename _T> \
struct name { \
private: \
	typedef typename _T::value_type value_type; \
\
	value_type v_; \
	_T x_; \
\
public: \
	name(value_type v, _T x) : v_(v), x_(x) {} \
\
	MATAZURE_GENERAL auto operator ()(const int_t &i) const->decltype((this->_v) op (this->x_[i])){ \
		return  v_ op x_[i]; \
	} \
};

#define __MATAZURE_VALUE_WITH_ARRAY_ACCESS_TENSOR_BINARY_OPERATOR(name, op) \
template <typename _T> \
struct name { \
private: \
	typedef typename _T::value_type value_type; \
\
	value_type v_; \
	_T x_; \
\
public:\
	name(_T x, value_type v) : v_(v), x_(x) {} \
\
	MATAZURE_GENERAL auto operator ()(const pointi<_T::rank> &idx) const->decltype(this->v_ op this->x_[idx]){ \
		return v_ op x_[idx]; \
	} \
\
};

//host tensor operations
#define TENSOR_BINARY_OPERATOR(name, op) \
__MATAZURE_LINEAR_ACCESS_TENSOR_BINARY_OPERATOR(__##name##_linear_access_tensor__, op) \
template <typename _TS1, typename _TS2> \
inline enable_if_t< none_device_memory<_TS1, _TS2>::value && are_linear_access<_TS1, _TS2>::value, lambda_tensor<_TS1::rank, __##name##_linear_access_tensor__<_TS1, _TS2>>> operator op(const tensor_expression<_TS1> &e_lhs, const tensor_expression<_TS2> &e_rhs) { \
	return make_lambda(e_lhs().shape(), __##name##_linear_access_tensor__<_TS1, _TS2>(e_lhs(), e_rhs()), host_t{}); \
} \
__MATAZURE_ARRAY_ACCESS_TENSOR_BINARY_OPERATOR(__##name##_array_access_tensor__, op) \
template <typename _TS1, typename _TS2> \
inline enable_if_t< none_device_memory<_TS1, _TS2>::value && !are_linear_access<_TS1, _TS2>::value, lambda_tensor<_TS1::rank, __##name##_array_access_tensor__<_TS1, _TS2>>> operator op(const tensor_expression<_TS1> &e_lhs, const tensor_expression<_TS2> &e_rhs) { \
	return make_lambda(e_lhs().shape(), __##name##_array_access_tensor__<_TS1, _TS2>(e_lhs(), e_rhs())); \
}

#define TENSOR_WITH_VALUE_BINARY_OPERATOR(name, op) \
__MATAZURE_LINEAR_ACCESS_TENSOR_WITH_VALUE_BINARY_OPERATOR(__##name##_linear_access_tensor_with_value__, op) \
\
template <typename _TS> \
inline enable_if_t< none_device_memory<_TS>::value && are_linear_access<_TS>::value, lambda_tensor<_TS::rank, __##name##_linear_access_tensor_with_value__<_TS>>> operator op(const tensor_expression<_TS> &e_ts, typename _TS::value_type v) { \
	return make_lambda(e_ts().shape(), __##name##_linear_access_tensor_with_value__<_TS>(e_ts(), v)); \
} \
\
__MATAZURE_VALUE_WITH_LINEAR_ACCESS_TENSOR_BINARY_OPERATOR(__##name##_value_with_linear_access_tensor__, op) \
template <typename _TS> \
inline enable_if_t< none_device_memory<_TS>::value && are_linear_access<_TS>::value, lambda_tensor<_TS::rank, __##name##_value_with_linear_access_tensor__<_TS>>> operator op(typename _TS::value_type v, const tensor_expression<_TS> &e_ts) { \
	return make_lambda(e_ts().shape(), __##name##_value_with_linear_access_tensor__<_TS>(v, e_ts())); \
} \
\
__MATAZURE_ARRAY_ACCESS_TENSOR_WITH_VALUE_BINARY_OPERATOR(__##name##_array_access_tensor_with_value__, op) \
template <typename _TS> \
inline enable_if_t< none_device_memory<_TS>::value && !are_linear_access<_TS>::value, lambda_tensor<_TS::rank, __##name##_array_access_tensor_with_value__<_TS>>> operator op(const tensor_expression<_TS> &e_ts, typename _TS::value_type v) { \
	return make_lambda(e_ts().shape(), __##name##_array_access_tensor_with_value__<_TS>(e_ts(), v)); \
}\
\
__MATAZURE_VALUE_WITH_ARRAY_ACCESS_TENSOR_BINARY_OPERATOR(__##name##_value_with_array_access_tensor__, op) \
template <typename _TS> \
inline enable_if_t< none_device_memory<_TS>::value && !are_linear_access<_TS>::value, lambda_tensor<_TS::rank, __##name##_value_with_array_access_tensor__<_TS>>> operator op(typename _TS::value_type v, const tensor_expression<_TS> &e_ts) { \
	return make_lambda(e_ts().shape(), __##name##_value_with_array_access_tensor__<_TS>(v, e_ts())); \
}

//device tensor operations
#define CU_TENSOR_BINARY_OPERATOR(name, op) \
template <typename _TS1, typename _TS2> \
inline enable_if_t<are_device_memory<_TS1, _TS2>::value && are_linear_access<_TS1, _TS2>::value, cuda::general_lambda_tensor<_TS1::rank, __##name##_linear_access_tensor__<_TS1, _TS2>>> operator op(const tensor_expression<_TS1> &e_lhs, const tensor_expression<_TS2> &e_rhs) { \
	return make_lambda(e_lhs().shape(), __##name##_linear_access_tensor__<_TS1, _TS2>(e_lhs(), e_rhs()), device_t{}); \
} \
template <typename _TS1, typename _TS2> \
inline enable_if_t< are_device_memory<_TS1, _TS2>::value && !are_linear_access<_TS1, _TS2>::value, cuda::general_lambda_tensor<_TS1::rank, __##name##_array_access_tensor__<_TS1, _TS2>>> operator op(const tensor_expression<_TS1> &e_lhs, const tensor_expression<_TS2> &e_rhs) { \
	return make_lambda(e_lhs().shape(), __##name##_array_access_tensor__<_TS1, _TS2>(e_lhs(), e_rhs()), device_t{}); \
}

#define CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(name, op) \
\
template <typename _TS> \
inline enable_if_t< are_device_memory<_TS>::value && are_linear_access<_TS>::value, cuda::general_lambda_tensor<_TS::rank, __##name##_linear_access_tensor_with_value__<_TS>>> operator op(const tensor_expression<_TS> &e_ts, typename _TS::value_type v) { \
	return make_lambda(e_ts().shape(), __##name##_linear_access_tensor_with_value__<_TS>(e_ts(), v), device_t{}); \
} \
\
template <typename _TS> \
inline enable_if_t< are_device_memory<_TS>::value && are_linear_access<_TS>::value, cuda::general_lambda_tensor<_TS::rank, __##name##_value_with_linear_access_tensor__<_TS>>> operator op(typename _TS::value_type v, const tensor_expression<_TS> &e_ts) { \
	return make_lambda(e_ts().shape(), __##name##_value_with_linear_access_tensor__<_TS>(v, e_ts()), device_t{}); \
} \
\
template <typename _TS> \
inline enable_if_t< are_device_memory<_TS>::value && !are_linear_access<_TS>::value, cuda::general_lambda_tensor<_TS::rank, __##name##_array_access_tensor_with_value__<_TS>>> operator op(const tensor_expression<_TS> &e_ts, typename _TS::value_type v) { \
	return make_lambda(e_ts().shape(), __##name##_array_access_tensor_with_value__<_TS>(e_ts(), v), device_t{}); \
}\
\
template <typename _TS> \
inline enable_if_t< are_device_memory<_TS>::value && !are_linear_access<_TS>::value, cuda::general_lambda_tensor<_TS::rank, __##name##_value_with_array_access_tensor__<_TS>>> operator op(typename _TS::value_type v, const tensor_expression<_TS> &e_ts) { \
	return make_lambda(e_ts().shape(), __##name##_value_with_array_access_tensor__<_TS>(v, e_ts()), device_t{}); \
}

//Arithmetic
TENSOR_BINARY_OPERATOR(add, +)
TENSOR_BINARY_OPERATOR(sub, -)
TENSOR_BINARY_OPERATOR(mul, *)
TENSOR_BINARY_OPERATOR(div, / )
TENSOR_BINARY_OPERATOR(mod, %)
TENSOR_WITH_VALUE_BINARY_OPERATOR(add, +)
TENSOR_WITH_VALUE_BINARY_OPERATOR(sub, -)
TENSOR_WITH_VALUE_BINARY_OPERATOR(mul, *)
TENSOR_WITH_VALUE_BINARY_OPERATOR(div, /)
TENSOR_WITH_VALUE_BINARY_OPERATOR(mod, %)
//Bit
TENSOR_BINARY_OPERATOR(left_shift, <<)
TENSOR_BINARY_OPERATOR(right_shift, >>)
TENSOR_BINARY_OPERATOR(bit_or, |)
TENSOR_BINARY_OPERATOR(bit_and, &)
TENSOR_BINARY_OPERATOR(bit_xor, ^)
TENSOR_WITH_VALUE_BINARY_OPERATOR(left_shift, <<)
TENSOR_WITH_VALUE_BINARY_OPERATOR(right_shift, >>)
TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_or, |)
TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_and, &)
TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_xor, ^)
//Logic
TENSOR_BINARY_OPERATOR(or , ||)
TENSOR_BINARY_OPERATOR(and, &&)
TENSOR_WITH_VALUE_BINARY_OPERATOR(or , ||)
TENSOR_WITH_VALUE_BINARY_OPERATOR(and, &&)
//Compapre
TENSOR_BINARY_OPERATOR(gt, >)
TENSOR_BINARY_OPERATOR(lt, <)
TENSOR_BINARY_OPERATOR(ge, >=)
TENSOR_BINARY_OPERATOR(le, <=)
TENSOR_BINARY_OPERATOR(equal, ==)
TENSOR_BINARY_OPERATOR(not_equal, !=)
TENSOR_WITH_VALUE_BINARY_OPERATOR(gt, >)
TENSOR_WITH_VALUE_BINARY_OPERATOR(lt, <)
TENSOR_WITH_VALUE_BINARY_OPERATOR(ge, >=)
TENSOR_WITH_VALUE_BINARY_OPERATOR(le, <=)
TENSOR_WITH_VALUE_BINARY_OPERATOR(equal, ==)
TENSOR_WITH_VALUE_BINARY_OPERATOR(not_equal, !=)

#ifdef MATAZURE_CUDA
//Arithmetic
CU_TENSOR_BINARY_OPERATOR(add, +)
CU_TENSOR_BINARY_OPERATOR(sub, -)
CU_TENSOR_BINARY_OPERATOR(mul, *)
CU_TENSOR_BINARY_OPERATOR(div, /)
CU_TENSOR_BINARY_OPERATOR(mod, %)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(add, +)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(sub, -)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(mul, *)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(div, /)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(mod, %)
//Bit
CU_TENSOR_BINARY_OPERATOR(left_shift, <<)
CU_TENSOR_BINARY_OPERATOR(right_shift, >>)
CU_TENSOR_BINARY_OPERATOR(bit_or, |)
CU_TENSOR_BINARY_OPERATOR(bit_and, &)
CU_TENSOR_BINARY_OPERATOR(bit_xor, ^)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(left_shift, <<)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(right_shift, >>)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_or, |)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_and, &)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(bit_xor, ^)
//Logic
CU_TENSOR_BINARY_OPERATOR(or , ||)
CU_TENSOR_BINARY_OPERATOR(and, &&)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(or , ||)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(and, &&)
//Compapre
CU_TENSOR_BINARY_OPERATOR(gt, >)
CU_TENSOR_BINARY_OPERATOR(lt, <)
CU_TENSOR_BINARY_OPERATOR(ge, >=)
CU_TENSOR_BINARY_OPERATOR(le, <=)
CU_TENSOR_BINARY_OPERATOR(equal, ==)
CU_TENSOR_BINARY_OPERATOR(not_equal, !=)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(gt, >)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(lt, <)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(ge, >=)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(le, <=)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(equal, ==)
CU_TENSOR_WITH_VALUE_BINARY_OPERATOR(not_equal, !=)

#endif

}
