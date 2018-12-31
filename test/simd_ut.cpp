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

	EXPECT_FLOAT_EQ(re[0], 5.0f);
	EXPECT_FLOAT_EQ(re[1], 5.0f);
	EXPECT_FLOAT_EQ(re[2], 5.0f);
	EXPECT_FLOAT_EQ(re[3], 5.0f);
}

#endif

#ifdef MATAZURE_NEON

TEST(TestNEON, TestArithmeticOperation){
	neon_vector<float, 4> lhs{ 0.0f, 1.0f, 2.0f, 3.0f};
	neon_vector<float, 4> rhs{ 0.0f, 0.1f, 0.2f, 0.3f };

	auto re = lhs + rhs;

	EXPECT_FLOAT_EQ(re[0], 0.0f);
	EXPECT_FLOAT_EQ(re[1], 1.1f);
	EXPECT_FLOAT_EQ(re[2], 2.2f);
	EXPECT_FLOAT_EQ(re[3], 3.3f);
}

#endif
