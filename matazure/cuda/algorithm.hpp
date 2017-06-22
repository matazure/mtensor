#pragma once

#include <matazure/point.hpp>
#include <matazure/cuda/execution.hpp>

namespace matazure {
namespace cuda {

namespace internal{

inline MATAZURE_GENERAL uint3 pointi_to_uint3(pointi<1> p) {
	return{ static_cast<unsigned int>(p[0]), 0, 0 };
}

inline MATAZURE_GENERAL uint3 pointi_to_uint3(pointi<2> p) {
	return{ static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]), 0 };
}

inline MATAZURE_GENERAL uint3 pointi_to_uint3(pointi<3> p) {
	return{ static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]), static_cast<unsigned int>(p[2]) };
}

template <int_t _Rank>
inline MATAZURE_GENERAL pointi<_Rank> uint3_to_pointi(uint3 u);

template <>
inline MATAZURE_GENERAL pointi<1> uint3_to_pointi(uint3 u) {
	return{ static_cast<int_t>(u.x) };
}

template <>
inline MATAZURE_GENERAL pointi<2> uint3_to_pointi(uint3 u) {
	return{ static_cast<int_t>(u.x), static_cast<int>(u.y) };
}

template <>
inline MATAZURE_GENERAL pointi<3> uint3_to_pointi(uint3 u) {
	return{ static_cast<int>(u.x), static_cast<int>(u.y), static_cast<int>(u.z) };
}

inline MATAZURE_GENERAL dim3 pointi_to_dim3(pointi<1> p) {
	return{ static_cast<unsigned int>(p[0]), 1, 1 };
}

inline MATAZURE_GENERAL dim3 pointi_to_dim3(pointi<2> p) {
	return{ static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]), 1 };
}

inline MATAZURE_GENERAL dim3 pointi_to_dim3(pointi<3> p) {
	return{ static_cast<unsigned int>(p[0]), static_cast<unsigned int>(p[1]), static_cast<unsigned int>(p[2]) };
}

template <int_t _Rank>
inline MATAZURE_GENERAL pointi<_Rank> dim3_to_pointi(dim3 u);

template <>
inline MATAZURE_GENERAL pointi<1> dim3_to_pointi(dim3 u) {
	return{ static_cast<int_t>(u.x) };
}

template <>
inline MATAZURE_GENERAL pointi<2> dim3_to_pointi(dim3 u) {
	return{ static_cast<int_t>(u.x), static_cast<int>(u.y) };
}

template <>
inline MATAZURE_GENERAL pointi<3> dim3_to_pointi(dim3 u) {
	return{ static_cast<int>(u.x), static_cast<int>(u.y), static_cast<int>(u.z) };
}

}

namespace device {

template <typename _Func>
inline MATAZURE_DEVICE void for_index(int_t first, int_t last, _Func fun) {
	for (int_t i = first; i < last; ++i) {
		fun(i);
	}
}

template <typename _Func>
inline MATAZURE_DEVICE void for_index(pointi<1> origin, pointi<1> extent, _Func fun) {
	for (int_t i = origin[0]; i < extent[0]; ++i) {
		fun(pointi<1>{ { i } });
	}
}

template <typename _Func>
inline MATAZURE_DEVICE void for_index(pointi<2> origin, pointi<2> extent, _Func fun) {
	for (int_t j = origin[1]; j < extent[1]; ++j) {
		for (int_t i = origin[0]; i < extent[0]; ++i) {
			fun(pointi<2>{ { i, j } });
		}
	}
}

template <typename _Func>
inline MATAZURE_DEVICE void for_index(pointi<3> origin, pointi<3> extent, _Func fun) {
	for (int_t k = origin[2]; k < extent[2]; ++k) {
		for (int_t j = origin[1]; j < extent[1]; ++j) {
			for (int_t i = origin[0]; i < extent[0]; ++i) {
				fun(pointi<3>{ { i, j, k } });
			}
		}
	}
}

template <typename _Func>
inline MATAZURE_DEVICE void for_index(pointi<4> origin, pointi<4> extent, _Func fun) {
	for (int_t l = origin[3]; l < extent[3]; ++l) {
		for (int_t k = origin[2]; k < extent[2]; ++k) {
			for (int_t j = origin[1]; j < extent[1]; ++j) {
				for (int_t i = origin[0]; i < extent[0]; ++i) {
					fun(pointi<4>{ {i, j, k, l} });
				}
			}
		}
	}
}

}

template <typename Function, typename... Arguments>
MATAZURE_GLOBAL void kenel(Function f, Arguments... args)
{
	f(args...);
}

template <typename _Fun, typename... _Args>
inline void launch(_Fun f, _Args... args)
{
	execution_policy exe_policy;
	launch(exe_policy, f, args...);
}

template <typename _ExecutionPolicy, typename _Fun, typename... _Args>
inline void launch(_ExecutionPolicy exe_policy, _Fun f, _Args... args)
{
	configure_grid(exe_policy, kenel<_Fun, _Args...>);
	kenel <<< exe_policy.grid_size(), exe_policy.block_size(), exe_policy.shared_mem_bytes(), exe_policy.stream() >>> (f, args...);
	assert_runtime_success(cudaGetLastError());
}

template <typename _ExecutionPolicy, typename _Fun>
inline void for_index(_ExecutionPolicy policy, int_t first, int_t last, _Fun fun) {
	launch(policy, [=] MATAZURE_DEVICE() {
		for (int_t i = first + threadIdx.x + blockIdx.x * blockDim.x; i < last; i += blockDim.x * gridDim.x) {
			fun(i);
		}
	});
}

template <typename _Fun>
inline void for_index(int_t first, int_t last, _Fun fun) {
	parallel_execution_policy policy;
	policy.total_size(last - first);
	cuda::for_index(policy, first, last, fun);
}

template <typename _ExecutionPolicy, int_t _Rank, typename _Fun>
inline void for_index(_ExecutionPolicy policy, pointi<_Rank> ext, _Fun fun) {
	auto stride = matazure::accumulate_stride(ext);
	auto max_size = index2offset((ext - 1), stride, first_major_t{}) + 1; //要包含最后一个元素

	cuda::for_index(policy, 0, max_size, [=] MATAZURE_DEVICE (int_t i) {
		fun(offset2index(i, stride, first_major_t{}));
	});
}

template <int_t _Rank, typename _Fun>
inline void for_index(pointi<_Rank> ext, _Fun fun) {
	execution_policy p;
	cuda::for_index(p, ext, fun);
}

template <typename _BlockDim>
class block_index{
public:
	const static int_t rank = _BlockDim::size();

	MATAZURE_GENERAL block_index(pointi<rank> grid_extent, pointi<rank> local_idx, pointi<rank> block_idx, pointi<rank> global_idx) :
		block_dim(_BlockDim::value()),
		grid_dim(grid_extent),
		global_dim(block_dim * grid_extent),
		local(local_idx),
		block(block_idx),
		global(global_idx)
	{}

public:
	const pointi<rank> block_dim;
	const pointi<rank> grid_dim;
	const pointi<rank> global_dim;
	const pointi<rank> local;
	const pointi<rank> block;
	const pointi<rank> global;
};

template <typename _Ext, typename _Fun>
inline void block_for_index(pointi<_Ext::size()> grid_ext, _Fun fun) {
	auto grid_dim = internal::pointi_to_dim3(grid_ext);
	auto block_dim = internal::pointi_to_dim3(_Ext::value());
	kenel <<<grid_dim, block_dim>>> ([=] MATAZURE_DEVICE() {
		auto local = internal::uint3_to_pointi<_Ext::size()>(threadIdx);
		auto block = internal::uint3_to_pointi<_Ext::size()>(blockIdx);
		auto block_dim = internal::dim3_to_pointi<_Ext::size()>(blockDim);
		auto global = block * block_dim + local;
		block_index<_Ext> block_idx(grid_ext, local, block, global);
		fun(block_idx);
	});

	assert_runtime_success(cudaGetLastError());
}

template <typename _ExecutionPolicy, typename _Tensor, typename _Fun>
inline void for_each(_ExecutionPolicy policy, _Tensor ts, _Fun fun, enable_if_t<are_device_memory<_Tensor>::value && are_linear_access<_Tensor>::value>* = 0) {
	cuda::for_index(policy, 0, ts.size(), [=] MATAZURE_DEVICE(int_t i) {
		fun(ts[i]);
	});
}

template <typename _ExecutionPolicy, typename _Tensor, typename _Fun>
inline void for_each(_ExecutionPolicy policy, _Tensor ts, _Fun fun, enable_if_t<are_device_memory<_Tensor>::value && !are_linear_access<_Tensor>::value>* = 0) {
	cuda::for_index(policy, ts.shape(), [=] MATAZURE_DEVICE(pointi<_Tensor::rank> idx) {
		fun(ts[idx]);
	});
}

template <typename _Tensor, typename _Fun>
inline void for_each(_Tensor ts, _Fun fun, enable_if_t<are_device_memory<_Tensor>::value>* = 0) {
	parallel_execution_policy policy;
	policy.total_size(ts.size());
	for_each(policy, ts, fun, (void *)(0));
}

template <typename _ExecutionPolicy, typename _Tensor>
inline void fill(_ExecutionPolicy policy, _Tensor ts, typename _Tensor::value_type v, enable_if_t<are_device_memory<_Tensor>::value>* = 0) {
	for_each(policy, ts, [v] MATAZURE_DEVICE(typename _Tensor::value_type &element) {
		element = v;
	}, (void *)(0));
}

template <typename _Tensor>
inline void fill(_Tensor ts, typename _Tensor::value_type v, enable_if_t<are_device_memory<_Tensor>::value>* = 0) {
	parallel_execution_policy policy;
	policy.total_size(ts.size());
	fill(policy, ts, v, (void *)(0));
}

template <typename _ExecutionPolicy, typename _T1, typename _T2>
void copy(_ExecutionPolicy policy, _T1 lhs, _T2 rhs, enable_if_t<are_linear_access<_T1, _T2>::value && are_device_memory<_T1, _T2>::value>* = 0) {
	cuda::for_index(policy, 0, lhs.size(), [=] MATAZURE_DEVICE(int_t i) {
		rhs[i] = lhs[i];
	});
}

template <typename _ExecutionPolicy, typename _T1, typename _T2>
void copy(_ExecutionPolicy policy, _T1 lhs, _T2 rhs, enable_if_t<!are_linear_access<_T1, _T2>::value && are_device_memory<_T1, _T2>::value>* = 0) {
	cuda::for_index(policy, lhs.shape(), [=] MATAZURE_DEVICE(pointi<_T1::rank> idx) {
		rhs[idx] = lhs[idx];
	});
}

template <typename _T1, typename _T2>
void copy(_T1 lhs, _T2 rhs, enable_if_t<is_tensor<_T1>::value>* = 0, enable_if_t<are_device_memory<_T1, _T2>::value>* = 0) {
	parallel_execution_policy policy;
	policy.total_size(lhs.size());
	copy(policy, lhs, rhs, (void *)(0));
}

}

//use in matazure
using cuda::for_each;
using cuda::fill;
using cuda::copy;

}
