#pragma once

#include <matazure/point.hpp>
#include <matazure/cuda/execution.hpp>

namespace matazure {
namespace cuda {

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
	kenel<<< exe_policy.grid_size(), exe_policy.block_size(), exe_policy.shared_mem_bytes(), exe_policy.stream() >>>(f, args...);
	assert_runtime_success(cudaGetLastError());
}

template <typename _Fun>
inline void parallel_for_index(int_t first, int_t last, _Fun fun) {
	parallel_execution_policy parallel_policy;
	parallel_policy.parallel_size(last - first);
	parallel_for_index(parallel_policy, first, last, fun);
}

template <typename _ExecutionPolicy, typename _Fun>
inline void parallel_for_index(_ExecutionPolicy policy, int_t first, int_t last, _Fun fun) {
	launch(policy, [=] MATAZURE_DEVICE() {
		for (int_t i = first + threadIdx.x + blockIdx.x * blockDim.x; i < last; i += blockDim.x * gridDim.x) {
			fun(i);
		}
	});
}

template <int_t _Dim, typename _Fun>
inline void parallel_for_index(pointi<_Dim> ext, _Fun fun) {
	execution_policy p;
	parallel_for_index(p, ext, fun);
}

template <typename _ExecutionPolicy, int_t _Dim, typename _Fun>
inline void parallel_for_index(_ExecutionPolicy policy, pointi<_Dim> ext, _Fun fun) {
	auto stride = matazure::get_stride(ext);
	auto max_size = index2offset((ext - 1), stride, first_major_t{}) + 1; //要包含最后一个元素

	parallel_for_index(policy, 0, max_size, [=] MATAZURE_DEVICE (int_t i) {
		fun(offset2index(i, stride, first_major_t{}));
	});
}

template <int_t ..._Dims>
class block_index;

template <int_t _S0, int_t _S1>
class block_index<_S0, _S1> {
public:
	MATAZURE_GENERAL block_index(pointi<2> grid_extent, pointi<2> local_idx, pointi<2> block_idx, pointi<2> global_idx) :
		block_extent{ _S0, _S1 },
		grid_extent(grid_extent),
		global_extent(block_extent * grid_extent),
		local(local_idx),
		block(block_idx),
		global(global_idx)
	{}

public:
	const pointi<2> block_extent;
	const pointi<2> grid_extent;
	const pointi<2> global_extent;
	const pointi<2> local;
	const pointi<2> block;
	const pointi<2> global;
};

template <int_t _S0, int_t _S1, int_t _S2>
class block_index<_S0, _S1, _S2> {
public:

};

template <int_t _BlockSize, typename _Fun>
inline void block_for_index(int_t grid_size, _Fun fun) {
	kenel<<< grid_size, _BlockSize >>>(fun);
}

template <int_t _S0, int_t _S1, typename _Fun>
inline void block_for_index(pointi<2> grid_ext, _Fun fun) {
	kenel <<< dim3(grid_ext[0], grid_ext[1], 1), dim3(_S0, _S1, 1) >>> ([=] MATAZURE_DEVICE() {
	 pointi<2> local = { static_cast<int_t>(threadIdx.x), static_cast<int_t>(threadIdx.y) };
	 pointi<2> block = { static_cast<int_t>(blockIdx.x), static_cast<int_t>(blockIdx.y) };
	 pointi<2> block_ext = { _S0, _S1 };
	 pointi<2> global = block * block_ext + local;
		block_index<_S0, _S1> block_idx(grid_ext, local, block, global);
		fun(block_idx);
	});

	assert_runtime_success(cudaGetLastError());
}

template <int_t _S0, int_t _S1, int_t _S2, typename _Fun>
inline void block_for_index(pointi<3> grid_ext, _Fun fun) {
	kenel << < dim3(grid_ext[0], grid_ext[1], grid_ext[2]), dim3(_S0, _S1, _S2) >> > ([=] MATAZURE_DEVICE() {
	 pointi<3> local = { static_cast<int_t>(threadIdx.x), static_cast<int_t>(threadIdx.y), static_cast<int_t>(threadIdx.z) };
	 pointi<3> block = { static_cast<int_t>(blockIdx.x), static_cast<int_t>(blockIdx.y), static_cast<int_t>(blockIdx.z) };
	 pointi<3> block_ext = { _S0, _S1, _S2 };
	 pointi<3> global = block * block_ext + local;
		block_index<_S0, _S1, _S2> block_idx(grid_ext, local, block, global);
		fun(block_idx);
	});
}

template <typename _Tensor, typename _Fun>
inline void for_each(_Tensor ts, _Fun fun, enable_if_t<are_device_memory<_Tensor>::value && are_linear_access<_Tensor>::value>* = 0) {
	parallel_for_index(0, ts.size(), [=] MATAZURE_DEVICE (int_t i) {
		fun(ts[i]);
	});
}

template <typename _Tensor, typename _Fun>
inline void for_each(_Tensor ts, _Fun fun, enable_if_t<are_device_memory<_Tensor>::value && !are_linear_access<_Tensor>::value>* = 0) {
	parallel_for_index(ts.extent(), [=] MATAZURE_DEVICE(pointi<_Tensor::rank> idx) {
		fun(ts(idx));
	});
}

template <typename _Tensor>
inline void fill(_Tensor ts, typename _Tensor::value_type v, enable_if_t<are_device_memory<_Tensor>::value>* = 0) {
	for_each(ts, [v] MATAZURE_DEVICE (typename _Tensor::value_type &element) {
		element = v;
	});
}

template <typename _T1, typename _T2>
void copy(_T1 lhs, _T2 rhs, enable_if_t<are_linear_access<_T1, _T2>::value && are_device_memory<_T1, _T2>::value>* = 0) {
	parallel_for_index(0, lhs.size(), [=] MATAZURE_DEVICE (int_t i) {
		rhs[i] = lhs[i];
	});
}

template <typename _T1, typename _T2>
void copy(_T1 lhs, _T2 rhs, enable_if_t<!are_linear_access<_T1, _T2>::value && are_device_memory<_T1, _T2>::value>* = 0) {
	parallel_for_index(lhs.extent(), [=] MATAZURE_DEVICE (pointi<_T1::rank> idx) {
		rhs(idx) = lhs(idx);
	});
}

}

//use in matazure
using cuda::for_each;
using cuda::fill;
using cuda::copy;

}
