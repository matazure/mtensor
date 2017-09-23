#pragma once

#include <matazure/tensor>
#include <benchmark/benchmark.h>
#include <cmath>

using matazure::host_tag;
using matazure::device_tag;

struct bm_config{

	template <typename _Type, int _Rank, typename _Device=host_tag>
	static int min_shape() {
		return 1 << (10 / _Rank);
	}

	template <typename _Type, int _Rank, typename _Device=host_tag>
	static int max_shape() {
		return  1 << ((max_memory_exponent<_Device>() - static_cast<int>(std::ceil(std::log(sizeof(_Type))/std::log(2)))) / _Rank);
	}

	template <typename _Type, int _Rank, typename _Device=host_tag>
	static int range_multiplier() {
		return  std::max(8 >> _Rank, 2);
	}

	template <typename _Device=host_tag>
	static int max_memory_exponent(){
		return 29;
	}

};

#ifdef USE_CUDA
#define HETE_TENSOR cuda::tensor
#define HETE_TAG device_tag
#endif

#ifdef USE_HOST
#define HETE_TENSOR tensor
#define HETE_TAG host_tag
#endif
