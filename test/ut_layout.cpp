#include "ut_foundation.hpp"

TEST(RowMajorLayoutTests, TestStride) {
    row_major_layout<2> layout(pointi<2>{5, 10});

    auto stride = layout.stride();
    EXPECT_EQ(stride[1], 1);
    EXPECT_EQ(stride[0], 10);
}
