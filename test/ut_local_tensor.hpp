#include "ut_foundation.hpp"

TEST(LocalTensorTests, ColumnLayout) {
    // mem continue initialize the local tensor
    local_tensor<float, dim<2, 3>, column_major_layout<2>> lt = {0, 1, 2, 3, 4, 5};

    ASSERT_EQ(0, lt(0, 0));
    ASSERT_EQ(2, lt(0, 1));
    ASSERT_EQ(4, lt(0, 2));
    ASSERT_EQ(1, lt(1, 0));
    ASSERT_EQ(3, lt(1, 1));
    ASSERT_EQ(5, lt(1, 2));
}

TEST(LocalTensorTests, RowLayout) {
    // mem continue initialize the local tensor
    local_tensor<float, dim<2, 3>, row_major_layout<2>> lt = {0, 1, 2, 3, 4, 5};

    ASSERT_EQ(0, lt(0, 0));
    ASSERT_EQ(1, lt(0, 1));
    ASSERT_EQ(2, lt(0, 2));
    ASSERT_EQ(3, lt(1, 0));
    ASSERT_EQ(4, lt(1, 1));
    ASSERT_EQ(5, lt(1, 2));
}