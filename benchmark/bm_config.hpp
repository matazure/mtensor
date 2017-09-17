#pragma once

#include <benchmark/benchmark.h>

struct bm_config{

	template <typename _Type, int _Rank>
	static int min_shape() {
		return 1 << (10 / _Rank);
	}

	template <typename _Type, int _Rank>
	static int max_shape() {
		return  1 << ((max_host_memory_exponent() - static_cast<int>(std::ceil(std::log2(sizeof(_Type))))) / _Rank);
	}

	static int max_host_memory_exponent(){
		return 30;
	}

	static int max_cuda_memory_exponent(){
		return 30;
	}

	static int max_cl_memory_exponent(){
		return 30;
	}
};
