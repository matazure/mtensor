#pragma once

#include <matazure/tensor.hpp>

#ifdef MATAZURE_CUDA

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

	#ifdef MATAZURE_CUDA

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
	};

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

		lambda_tensor(const matazure::lambda_tensor<rank, functor_type, layout_type> lts) :
			shape_(lts.shape()),
			layout_(lts.layout()),
			functor_(lts.functor())
		{}

	#ifdef MATAZURE_CUDA

		lambda_tensor(const cuda::general_lambda_tensor &glcu_ts) :
			shape_(glcu_ts.shape_),
			layout_(glcu_ts.layout_),
			functor_(glcu_ts.functor_)
		{ }

	#endif

	private:
		const pointi<rank> shape_;
		const layout_type layout_;
		const _Func functor_;
	};

	// template <typename _Rank, typename _Func, typename _Layout>
	// lambda_tensor<_Rank, _Func, _Layout> make_lambda(const matazure::lambda_tensor<_Rank, _Func, _Layout> lts){
	// 	return lambda_tensor<_Rank, _Func, _Layout>(lts);
	// }

} }
