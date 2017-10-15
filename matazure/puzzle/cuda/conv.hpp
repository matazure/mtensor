#pragma once

#include <matazure/cuda/tensor.hpp>
#include <matazure/puzzle/cuda/algorithm.hpp>

#define MATAZURE_CUDA_PUZZEL_CONV_LAZY_ARRAY_INDEX_UNCLAMP_CONSTANT_KERNEL(conv_lazy_array_index_unclamp_constant_kernel, mask)		\
namespace matazure{namespace cuda{ namespace puzzle {														\
																											\
namespace internal {																						\
																											\
template <typename _Tensor>																					\
struct conv_array_index_op {																				\
private:																									\
	_Tensor ts_;																							\
public:																										\
	conv_array_index_op(_Tensor ts) :																		\
		ts_(ts)																								\
	{}																										\
																											\
	MATAZURE_GENERAL typename _Tensor::value_type operator()(const pointi<_Tensor::rank> &idx) const {		\
		auto mask_radius = mask.shape() / 2;																\
		auto sum = zero<typename _Tensor::value_type>::value();												\
		device::for_index(pointi<_Tensor::rank>::zeros(), mask.shape(), [&] (const pointi<2> &mask_idx) {	\
			sum += ts_[idx + mask_idx - mask_radius] * mask[mask_idx];										\
		});																									\
		return sum;																							\
	}																										\
};																											\
																											\
}																											\
																											\
template <typename _Tensor>																					\
inline auto conv_lazy_array_index_unclamp_constant_kernel(_Tensor ts)															\
->decltype(make_lambda(ts.shape(), internal::conv_array_index_op<decay_t<_Tensor>>(ts), typename _Tensor::memory_type{})) {		\
	return make_lambda(ts.shape(), internal::conv_array_index_op<decay_t<_Tensor>>(ts), typename _Tensor::memory_type{});		\
}																											\
																											\
}}} //matazure/cuda/puzzle


#define MATAZURE_CUDA_PUZZEL_CONV_BLOCK_ARRAY_INDEX_UNCLAME_CONSTANT_KERNEL(conv_block_array_index_unclamp_constant_kernel, mask)	\
namespace matazure { namespace cuda{ namespace puzzle{														\
																											\
template <typename _BlockDim, typename _Tensor, typename _TensorRe>											\
inline void conv_block_array_index_unclamp_constant_kernel(_Tensor ts, _TensorRe &ts_re) {							\
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_Tensor, decltype(mask));											\
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_Tensor, decltype(mask));										\
	typedef typename _Tensor::value_type value_type;														\
																											\
	constexpr auto block_ext = _BlockDim::value();															\
	auto grid_ext = (ts.shape() + block_ext - 1) / block_ext;												\
	MATAZURE_ASSERT(equal(ts.shape(), ts_re.shape()), "unmatched shape");									\
	auto shape_le = mask.shape() <= _BlockDim::value();														\
	for(int_t i = 0; i <shape_le.size(); ++i){																\
		MATAZURE_ASSERT(shape_le[i], "block dim could not be less than mask shape");						\
	}																										\
																											\
	block_for_index<_BlockDim>(grid_ext, [=] __device__ (block_index<_BlockDim> block_idx) {				\
		auto tmp_shape = meta::sub_c(																		\
			meta::add_c(_BlockDim{}, decltype(mask)::meta_shape()), meta::int_t_c<1>{});					\
		__shared__ static_tensor<value_type, decltype(tmp_shape)> sh_ts_block;								\
																											\
		auto is_valid = inside_range(block_idx.global, pointi<_Tensor::rank>::zeros(), ts.shape());			\
		if (is_valid) {																						\
			device::puzzle::corner_index(pointi<_Tensor::rank>::zeros(), mask.shape(),						\
				[&](pointi<_Tensor::rank> corner_idx) {														\
					sh_ts_block[block_idx.local + corner_idx] = 											\
						ts[block_idx.global + corner_idx - mask.shape() / 2];								\
			}																								\
			);																								\
		}																									\
		device::barrier();																					\
																											\
		if (!is_valid) return;																				\
		auto sum = zero<value_type>::value();																\
		device::for_index(pointi<_Tensor::rank>::zeros(), mask.shape(), [&](const pointi<2> &idx) {			\
			sum += sh_ts_block[block_idx.local + idx] * mask[idx];											\
		});																									\
		ts_re[block_idx.global] = sum;																		\
	});																										\
}																											\
																											\
}}}  //end conv_block_array_index_unclamp_constant_kernel

#define MATAZURE_CUDA_PUZZEL_CONV_BLOCK_CRACK_ARRAY_INDEX_UNCLAMP_CONSTANT_KERNEL(conv_block_crack_array_index_unclamp_constant_kernel, mask)	\
namespace matazure{namespace cuda{namespace puzzle{															\
																											\
template <typename _BlockDim, typename _Tensor, typename _TensorRe>											\
inline void conv_block_crack_array_index_unclamp_constant_kernel(_Tensor ts, _TensorRe &ts_re) {							\
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_Tensor, decltype(mask));											\
	typedef typename _Tensor::value_type value_type;														\
																											\
	constexpr auto block_ext = _BlockDim::value();															\
	auto grid_ext = (ts.shape() + block_ext - 1) / block_ext;												\
	MATAZURE_ASSERT(equal(ts.shape(), ts_re.shape()), "unmatched shape");									\
	auto shape_le = mask.shape() <= _BlockDim::value();														\
	for(int_t i = 0; i <shape_le.size(); ++i){																\
		MATAZURE_ASSERT(shape_le[i], "block dim could not be less than mask shape");						\
	}																										\
																											\
	block_for_index<_BlockDim>(grid_ext, [=] __device__ (block_index<_BlockDim> block_idx) {				\
		__shared__ static_tensor<value_type, _BlockDim> sh_ts_block;										\
																											\
		auto is_valid = inside_range(block_idx.global, pointi<_Tensor::rank>::zeros(), ts.shape());			\
		if (is_valid) {																						\
			sh_ts_block[block_idx.local] = ts[block_idx.global];											\
		}																									\
		device::barrier();																					\
																											\
		auto mask_radius = mask.shape() / 2;																\
		if (inside_range(block_idx.local, mask_radius, block_idx.block_dim - mask_radius) 					\
			&& inside_range(block_idx.global, mask_radius, ts.shape() - mask_radius)) {						\
			auto sum = zero<value_type>::value();															\
			device::for_index(pointi<_Tensor::rank>::zeros(), mask.shape(), [&](const pointi<2> &idx) {		\
				sum += sh_ts_block[block_idx.local + idx - mask_radius] * mask[idx];						\
			});																								\
			ts_re[block_idx.global] = sum;																	\
		}																									\
	});																										\
}																											\
																											\
}}}	 //end conv_block_crack_array_index_unclamp_constant_kernel
																											\
#define MATAZURE_CUDA_PUZZEL_CONV_BLOCK_OVERLAP_ARRAY_INDEX_UNCLAMP_CONSTANT_KERNEL(conv_block_overlap_array_index_unclamp_constant_kernel, mask)	\
namespace matazure { namespace cuda { namespace puzzle {													\
																											\
template <typename _BlockDim, typename _Tensor, typename _TensorRe>											\
inline void conv_block_overlap_array_index_unclamp_constant_kernel(_Tensor ts, _TensorRe &ts_re) {			\
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_Tensor, decltype(mask));											\
	typedef typename _Tensor::value_type value_type;														\
																											\
	auto valid_block_dim = meta::array_to_pointi(															\
		meta::add_c(meta::sub_c(_BlockDim{}, decltype(mask)::meta_shape_type{}), meta::int_t_c<1>{})		\
	);																										\
	auto grid_ext = (ts.shape() + valid_block_dim - 1) / valid_block_dim;									\
	MATAZURE_ASSERT(equal(ts.shape(), ts_re.shape()), "unmatched shape");									\
	auto shape_le = mask.shape() <= _BlockDim::value();														\
	for(int_t i = 0; i <shape_le.size(); ++i){																\
		MATAZURE_ASSERT(shape_le[i], "block dim could not be less than mask shape");						\
	}																										\
																											\
	block_for_index<_BlockDim>(grid_ext, [=] __device__(block_index<_BlockDim> block_idx) {					\
		__shared__ static_tensor<value_type, _BlockDim> sh_ts_block;										\
		auto valid_block_dim = meta::array_to_pointi(														\
			meta::add_c(meta::sub_c(_BlockDim{}, decltype(mask)::meta_shape()), meta::int_t_c<1>{})			\
		);																									\
		auto mask_radius = mask.shape() / 2;																\
		auto valid_global_idx = valid_block_dim * block_idx.block + block_idx.local - mask_radius;			\
																											\
		if (inside_range(valid_global_idx, -mask_radius, ts.shape() + mask_radius)) {						\
			sh_ts_block[block_idx.local] = ts[valid_global_idx];											\
		}																									\
		device::barrier();																					\
																											\
		if (inside_range(block_idx.local, mask_radius, block_idx.block_dim - mask_radius)					\
			&& inside_range(valid_global_idx, pointi<_Tensor::rank>::zeros(), ts.shape())) {				\
			auto sum = zero<value_type>::value();															\
			device::for_index(pointi<_Tensor::rank>::zeros(), mask.shape(), [&](const pointi<2> &idx) {		\
				sum += sh_ts_block[block_idx.local + idx - mask_radius] * mask[idx];						\
			});																								\
			ts_re[valid_global_idx] = sum;																	\
		}																									\
	});																										\
}																											\
																											\
}}}	 // end conv_block_overlap_array_index_unclamp_constant_kernel


#define MATAZURE_CUDA_PUZZEL_CONV_BLOCK_ALIGNED_ARRAY_INDEX_UNCLAMP_CONSTANT_KERNEL(conv_block_aligned_array_index_unclamp_constant_kernel, mask)	\
namespace matazure { namespace cuda{ namespace puzzle{														\
																											\
template <typename _BlockDim, typename _Tensor, typename _TensorRe>											\
inline void conv_block_aligned_array_index_unclamp_constant_kernel(_Tensor ts, _TensorRe &ts_re) {			\
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_Tensor, decltype(mask));											\
	MATAZURE_STATIC_ASSERT_VALUE_TYPE_MATCHED(_Tensor, decltype(mask));										\
	typedef typename _Tensor::value_type value_type;														\
																											\
	constexpr auto block_ext = _BlockDim::value();															\
	auto grid_ext = ts.shape() / block_ext;																	\
	MATAZURE_ASSERT(equal(grid_ext * block_ext, ts.shape()), "unaligned shape");							\
	MATAZURE_ASSERT(equal(ts.shape(), ts_re.shape()), "unmatched shape");									\
	auto shape_le = mask.shape() <= _BlockDim::value();														\
	for(int_t i = 0; i <shape_le.size(); ++i){																\
		MATAZURE_ASSERT(shape_le[i], "block dim should be greater than mask shape");						\
	}																										\
																											\
	block_for_index<_BlockDim>(grid_ext, [=] __device__ (block_index<_BlockDim> block_idx) {				\
		auto tmp_shape = 																					\
			meta::sub_c(meta::add_c(_BlockDim{}, decltype(mask)::meta_shape()), meta::int_t_c<1>{});		\
		__shared__ static_tensor<value_type, decltype(tmp_shape)> sh_ts_block;								\
																											\
		device::puzzle::corner_index(pointi<_Tensor::rank>::zeros(), mask.shape(),							\
			[&](pointi<_Tensor::rank> corner_idx){															\
				sh_ts_block[block_idx.local + corner_idx] = 												\
					ts[block_idx.global + corner_idx - mask.shape() / 2];									\
			}																								\
		);																									\
		device::barrier();																					\
																											\
		auto sum = zero<value_type>::value();																\
		device::for_index(pointi<_Tensor::rank>::zeros(), mask.shape(), [&](const pointi<2> &idx) {			\
			sum += sh_ts_block[block_idx.local + idx] * mask[idx];											\
		});																									\
		ts_re[block_idx.global] = sum;																		\
	});																										\
}																											\
																											\
}}}  //end conv_block_array_index_unclamp_constant_kernel

#define MATAZURE_CUDA_PUZZEL_CONV_BLOCK_CRACK_ALIGNED_ARRAY_INDEX_UNCLAMP_CONSTANT_KERNEL(conv_block_crack_aligned_array_index_unclamp_constant_kernel, mask)	\
namespace matazure{namespace cuda{namespace puzzle{															\
																											\
template <typename _BlockDim, typename _Tensor, typename _TensorRe>											\
inline void conv_block_crack_aligned_array_index_unclamp_constant_kernel(_Tensor ts, _TensorRe &ts_re) {	\
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_Tensor, decltype(mask));											\
	typedef typename _Tensor::value_type value_type;														\
																											\
	constexpr auto block_ext = _BlockDim::value();															\
	auto grid_ext = ts.shape() / block_ext;																	\
	MATAZURE_ASSERT(equal(grid_ext * block_ext, ts.shape()), "unaligned shape");							\
	MATAZURE_ASSERT(equal(ts.shape(), ts_re.shape()), "unmatched shape");									\
	auto shape_le = mask.shape() <= _BlockDim::value();														\
	for(int_t i = 0; i <shape_le.size(); ++i){																\
		MATAZURE_ASSERT(shape_le[i], "block dim could not be less than mask shape");						\
	}																										\
																											\
	block_for_index<_BlockDim>(grid_ext, [=] __device__ (block_index<_BlockDim> block_idx) {				\
		__shared__ static_tensor<value_type, _BlockDim> sh_ts_block;										\
		sh_ts_block[block_idx.local] = ts[block_idx.global];												\
		device::barrier();																					\
																											\
		auto mask_radius = mask.shape() / 2;																\
		if (inside_range(block_idx.local, mask_radius, block_idx.block_dim - mask_radius)) {				\
			auto sum = zero<value_type>::value();															\
			device::for_index(pointi<_Tensor::rank>::zeros(), mask.shape(), [&](const pointi<2> &idx) {		\
				sum += sh_ts_block[block_idx.local + idx - mask_radius] * mask[idx];						\
			});																								\
			ts_re[block_idx.global] = sum;																	\
		}																									\
	});																										\
}																											\
																											\
}}}	 //end conv_block_crack_array_index_unclamp_constant_kernel

#define MATAZURE_CUDA_PUZZEL_CONV_BLOCK_OVERLAP_ALIGNED_ARRAY_INDEX_UNCLAMP_CONSTANT_KERNEL(conv_block_overlap_aligned_array_index_unclamp_constant_kernel, mask)	\
namespace matazure { namespace cuda { namespace puzzle {													\
																											\
template <typename _BlockDim, typename _Tensor, typename _TensorRe>											\
inline void conv_block_overlap_aligned_array_index_unclamp_constant_kernel(_Tensor ts, _TensorRe &ts_re) {	\
	MATAZURE_STATIC_ASSERT_DIM_MATCHED(_Tensor, decltype(mask));											\
	typedef typename _Tensor::value_type value_type;														\
																											\
	auto valid_block_dim = meta::array_to_pointi(															\
		meta::add_c(meta::sub_c(_BlockDim{}, decltype(mask)::meta_shape_type{}), meta::int_t_c<1>{})		\
	);																										\
	auto grid_ext = ts.shape() / valid_block_dim;															\
	MATAZURE_ASSERT(equal(grid_ext * valid_block_dim, ts.shape()), "unaligned shape");						\
	MATAZURE_ASSERT(equal(ts.shape(), ts_re.shape()), "unmatched shape");									\
	auto shape_le = mask.shape() <= _BlockDim::value();														\
	for(int_t i = 0; i <shape_le.size(); ++i){																\
		MATAZURE_ASSERT(shape_le[i], "block dim could not be less than mask shape");						\
	}																										\
																											\
	block_for_index<_BlockDim>(grid_ext, [=] __device__ (block_index<_BlockDim> block_idx) {				\
		__shared__ static_tensor<value_type, _BlockDim> sh_ts_block;										\
		auto valid_block_dim = meta::array_to_pointi(														\
			meta::add_c(meta::sub_c(_BlockDim{}, decltype(mask)::meta_shape()), meta::int_t_c<1>{})			\
		);																									\
		auto mask_radius = mask.shape() / 2;																\
		auto valid_global_idx = valid_block_dim * block_idx.block + block_idx.local - mask_radius;			\
		sh_ts_block[block_idx.local] = ts[valid_global_idx];												\
		device::barrier();																					\
																											\
		if (inside_range(block_idx.local, mask_radius, block_idx.block_dim - mask_radius)) {				\
			auto sum = zero<value_type>::value();															\
			device::for_index(pointi<_Tensor::rank>::zeros(), mask.shape(), [&](const pointi<2> &idx) {		\
				sum += sh_ts_block[block_idx.local + idx - mask_radius] * mask[idx];						\
			});																								\
			ts_re[valid_global_idx] = sum;																	\
		}																									\
	});																										\
}																											\
																											\
}}}	 // end conv_block_overlap_array_index_unclamp_constant_kernel
