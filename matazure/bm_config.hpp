#pragma once

#include <matazure/tensor>
#include <benchmark/benchmark.h>
#include <cmath>

using namespace matazure;

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

using point3b = pointb<3>;
using point4b = pointb<4>;
using point3f = pointf<3>;
using point4f = pointf<4>;

#if defined(USE_HOST)

#define HETE_TENSOR tensor
#define HETE_TAG host_tag
#define HETE_SYNCHRONIZE

#if defined(MATAZURE_SSE) 
using hete_float32x4_t = sse_vector<float, 4> ;
#elif defined(MATAZURE_NEON)
using hete_float32x4_t = neon_vector<float, 4>;
#else
using hete_float32x4_t = pointf<4>;
#endif

#elif defined(USE_CUDA)

#define HETE_TENSOR cuda::tensor
#define HETE_TAG device_tag
#define HETE_SYNCHRONIZE  cuda::device_synchronize()

using hete_float32x4_t = pointf<4>;

#else

#if defined(MATAZURE_SSE) 
using hete_float32x4_t = sse_vector<float, 4>;
#elif defined(MATAZURE_NEON)
using hete_float32x4_t = neon_vector<float, 4>;
#else
using hete_float32x4_t = pointf<4>;
#endif

#endif

// #ifdef USE_OMP
//
// #endif
