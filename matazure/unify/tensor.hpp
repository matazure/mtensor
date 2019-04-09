#pragma once

#include <matazure/tensor.hpp>

#ifdef WITH_CUDA

#include <matazure/cuda/tensor.hpp>

#endif

#ifdef WITH_OPENCL

#include <matazure/opencl/tensor.hpp>

#endif

namespace matazure { namespace unify {

	template <typename _ValueType, int_t _Rank, typename _Layout = first_major_layout<_Rank>>
	class tensor {
	public:
		//typedef _TensorType								tensor_type;
		typedef _ValueType								value_type;
		const static int rank = _Rank;
		typedef value_type &							reference;
		typedef _Layout									layout_type;

		tensor(const matazure::tensor<value_type, rank, layout_type> &ts) :
			shape_(ts.shape_),
			layout_(ts.layout_),
			sp_data_(ts.sp_data_)
		{ }

	#ifdef WITH_CUDA

		tensor(const cuda::tensor<value_type, rank, layout_type> &cu_tensor) :
			shape_(cu_tensor.shape_),
			layout_(cu_tensor.layout_),
			sp_data(cu_tensor.sp_data_)
		{ }

	#endif

	private:
		pointi<rank>			shape_;
		layout_type				layout_;
		shared_ptr<value_type>	sp_data_;
	}

} }
