#include <gtest/gtest.h>
#include <matazure/simd/sse.hpp>

using namespace matazure;
using namespace testing;

using sse_fp32x4 = simd::sse_wrapper<point<float, 4>>;
using sse_uint8x16 = simd::sse_wrapper<point<std::uint8_t, 16>>;


TEST(ut_simd_sse, test_construct) {
	sse_fp32x4 p0, p1;

	auto tmp = p0 + p1;

	__m128 m0{}, m1{};
	auto tmp2 = _mm_add_ps(m0, m1);
	auto tmp3 = tmp2;

	printf("sse_wrapper :%d, point: %d\n", alignof(sse_fp32x4), alignof(point<float, 4>));

	int a = 0;
}
