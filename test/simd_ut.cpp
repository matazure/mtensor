#include <matazure/tensor>
#include <gtest/gtest.h>

using namespace matazure;
using namespace testing;

#ifdef MATAZURE_SSE

TEST(TestSSE, TestArithmeticOperation){
	__m128 lhs{1.0f, 1.0f, 1.0f, 1.0f};
	__m128 rhs{2.0f, 2.0f, 2.0f, 2.0f};

	lhs += rhs;

	__m128 re;
	re = lhs + rhs;
}

#endif

#ifdef MATAZURE_NEON

TEST(TestNEON, TestArithmeticOperation){
	float32x4_t lhs{0.0f, 1.0f, 2.0f, 3.0f};
	float32x4_t rhs{0.0f, 0.1f, 0.2f, 0.3f};

	lhs += rhs;
	auto re = slhs + rhs;
}

#endif
