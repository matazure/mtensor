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
	ExecutionPolicy p;
	launch(p, f, args...);
}

template <typename _ExecutionPolicy, typename _Fun, typename... _Args>
inline void launch(const _ExecutionPolicy &policy, _Fun f, _Args... args)
{
	ExecutionPolicy p = policy;
	throw_on_error(condigure_grid(p, kenel<_Fun, _Args...>));
	kenel<<< p.getGridSize(),
		p.getBlockSize(),
		p.getSharedMemBytes(),
		p.getStream() >>>(f, args...);

	throw_on_error(cudaGetLastError());
}

template <typename _Fun>
inline void parallel_for_index(int_t first, int_t last, _Fun fun) {
	ExecutionPolicy p;
	parallel_for_index(p, first, last, fun);
}

template <typename _ExecutionPolicy, typename _Fun>
inline void parallel_for_index(const _ExecutionPolicy &policy, int_t first, int_t last, _Fun fun) {
	launch(policy, [=] MATAZURE_DEVICE() {
		for (int_t i = first + threadIdx.x + blockIdx.x * blockDim.x; i < last; i += blockDim.x * gridDim.x) {
			fun(i);
		}
	});
}

template <int_t _Dim, typename _Fun>
inline void parallel_for_index(pointi<_Dim> ext, _Fun fun) {
	ExecutionPolicy p;
	parallel_for_index(p, ext, fun);
}

template <typename _ExecutionPolicy, int_t _Dim, typename _Fun>
inline void parallel_for_index(const _ExecutionPolicy &policy, pointi<_Dim> ext, _Fun fun) {
	auto stride = matazure::get_stride(ext);
	auto max_size = index2offset((ext - 1), stride, first_major_t{});

	parallel_for_index(policy, 0, max_size, [=] MATAZURE_DEVICE (int_t i) {
		fun(offset2index(i, stride, first_major_t{}));
	});
}

template <int_t ..._Dims>
class tile_index;

template <int_t _S0, int_t _S1>
class tile_index<_S0, _S1> {
public:
	MATAZURE_GENERAL tile_index(pointi<2> grid_extent, pointi<2> local_idx, pointi<2> tile_idx, pointi<2> global_idx) :
		tile_extent{ _S0, _S1 },
		grid_extent(grid_extent),
		global_extent(tile_extent * grid_extent),
		local(local_idx),
		tile(tile_idx),
		global(global_idx)
	{}

public:
	const pointi<2> tile_extent;
	const pointi<2> grid_extent;
	const pointi<2> global_extent;
	const pointi<2> local;
	const pointi<2> tile;
	const pointi<2> global;
};

template <int_t _S0, int_t _S1, int_t _S2>
class tile_index<_S0, _S1, _S2> {
public:

};

template <int_t _BlockSize, typename _Fun>
inline void tile_for_index(int_t grid_size, _Fun fun) {
	kenel<<< grid_size, _BlockSize >>>(fun);
}

template <int_t _S0, int_t _S1, typename _Fun>
inline void tile_for_index(pointi<2> grid_ext, _Fun fun) {
	kenel <<< dim3(grid_ext[0], grid_ext[1], 1), dim3(_S0, _S1, 1) >>> ([=] MATAZURE_DEVICE() {
	 pointi<2> local = { static_cast<int_t>(threadIdx.x), static_cast<int_t>(threadIdx.y) };
	 pointi<2> tile = { static_cast<int_t>(blockIdx.x), static_cast<int_t>(blockIdx.y) };
	 pointi<2> block_ext = { _S0, _S1 };
	 pointi<2> global = tile * block_ext + local;
		tile_index<_S0, _S1> tile_idx(grid_ext, local, tile, global);
		fun(tile_idx);
	});

	throw_on_error(cudaGetLastError());
}

template <int_t _S0, int_t _S1, int_t _S2, typename _Fun>
inline void tile_for_index(pointi<3> grid_ext, _Fun fun) {
	kenel << < dim3(grid_ext[0], grid_ext[1], grid_ext[2]), dim3(_S0, _S1, _S2) >> > ([=] MATAZURE_DEVICE() {
	 pointi<3> local = { static_cast<int_t>(threadIdx.x), static_cast<int_t>(threadIdx.y), static_cast<int_t>(threadIdx.z) };
	 pointi<3> tile = { static_cast<int_t>(blockIdx.x), static_cast<int_t>(blockIdx.y), static_cast<int_t>(blockIdx.z) };
	 pointi<3> block_ext = { _S0, _S1, _S2 };
	 pointi<3> global = tile * block_ext + local;
		tile_index<_S0, _S1, _S2> tile_idx(grid_ext, local, tile, global);
		fun(tile_idx);
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
	parallel_for_index(ts.extent(), [=] MATAZURE_DEVICE(pointi<_Tensor::dim> idx) {
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
	parallel_for_index(lhs.extent(), [=] MATAZURE_DEVICE (pointi<_T1::dim> idx) {
		rhs(idx) = lhs(idx);
	});
}

}

//use in matazure
using cuda::for_each;
using cuda::fill;
using cuda::copy;

}
