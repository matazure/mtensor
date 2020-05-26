#include "ut_foundation.hpp"

TEST(ZeroAndOneTests, Point) {
    auto tmp = zero<point<float, 4>>::value();
    ASSERT_FLOAT_EQ(0.0f, tmp[0]);
    ASSERT_FLOAT_EQ(0.0f, tmp[1]);
    ASSERT_FLOAT_EQ(0.0f, tmp[2]);
    ASSERT_FLOAT_EQ(0.0f, tmp[3]);
}

#ifdef __GNUC__
TEST(ZeroAndOneTests, AttributeVector) {
    typedef float simd_type __attribute__((vector_size(16)));
    auto tmp = zero<point<float, 4>>::value();
    ASSERT_FLOAT_EQ(0.0f, tmp[0]);
    ASSERT_FLOAT_EQ(0.0f, tmp[1]);
    ASSERT_FLOAT_EQ(0.0f, tmp[2]);
    ASSERT_FLOAT_EQ(0.0f, tmp[3]);
}
#endif