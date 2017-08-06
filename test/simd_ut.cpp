#include <matazure/tensor>
#include <gtest/gtest.h>

using namespace matazure;
using namespace testing;


TEST(SSE, ArithmeticOperation){
	__m128 lhs{1.0f, 1.0f, 1.0f, 1.0f};
	__m128 rhs{2.0f, 2.0f, 2.0f, 2.0f};

	lhs += rhs;
	auto re = lhs + rhs;

	//_mm_add_ps

	//pointi<2> p1{2, 3};
	//pointi<2> p2{1, 1};
	//p1 += 10;
}
