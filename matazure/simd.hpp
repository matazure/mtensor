#pragma once

#include <matazure/config.hpp>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__amd64)) && !defined(MATAZURE_DISABLE_SSE)
#define MATAZURE_SSE
#include <matazure/simd/sse.hpp>
#endif

#if defined(__ARM_NEON) && !defined(MATAZURE_DISABLE_NEON)
#define MATAZURE_NEON
#include <matazure/simd/neon.hpp>
#endif

#if defined(MATAZURE_SSE) || defined(MATAZURE_NEON)
#define MATAZUE_SIMD
#endif

namespace matazure {

#if defined(MATAZURE_SSE) && !defined(MATAZURE_NEON)
	template <typename _Type, int_t _Rank>
	using simd_vector = simd::sse_vector<_Type, _Rank>;
#endif

#if !defined(MATAZURE_SSE) && defined(MATAZURE_NEON)
	template <typename _Type, int_t _Rank>
	using simd_vector = simd::neon_vector<_Type, _Rank>;
#endif

}



