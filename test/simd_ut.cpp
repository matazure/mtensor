#include <matazure/tensor>
#include <gtest/gtest.h>

using namespace matazure;
using namespace testing;

#ifdef MATAZURE_SSE

TEST(SSE, ArithmeticOperation){
	__m128 lhs{1.0f, 1.0f, 1.0f, 1.0f};
	__m128 rhs{2.0f, 2.0f, 2.0f, 2.0f};

	lhs += rhs;
	auto re = lhs + rhs;

	auto p = new char[100];
	_mm_prefetch(p, _MM_HINT_T0);
}

#endif
