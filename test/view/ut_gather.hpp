#pragma once

#include "../ut_foundation.hpp"

TEST(ViewTests, GatherScalar) {
    // clang-format off
    tensor<float, 2> ts =
     {{0, 1, 2, 3}, 
      {4, 5, 6, 7}, 
      {8, 9, 10, 11}, 
      {12, 13, 14, 15}};
    // clang-format on

    auto ts_gather = view::gather<1>(ts, 1).persist();

    EXPECT_EQ(1, ts_gather(0));
    EXPECT_EQ(5, ts_gather(1));
    EXPECT_EQ(9, ts_gather(2));
    EXPECT_EQ(13, ts_gather(3));
}

TEST(ViewTests, GatherVector) {
    // clang-format off
    tensor<float, 2> ts =
     {{0, 1, 2, 3}, 
      {4, 5, 6, 7}, 
      {8, 9, 10, 11}, 
      {12, 13, 14, 15}};
    // clang-format on

    pointi<2> indices = {3, 1};
    auto ts_gather = view::gather<0>(ts, indices);

    EXPECT_EQ(12, ts_gather(0, 0));
    EXPECT_EQ(13, ts_gather(0, 1));
    EXPECT_EQ(14, ts_gather(0, 2));
    EXPECT_EQ(15, ts_gather(0, 3));
    EXPECT_EQ(4, ts_gather(1, 0));
    EXPECT_EQ(5, ts_gather(1, 1));
    EXPECT_EQ(6, ts_gather(1, 2));
    EXPECT_EQ(7, ts_gather(1, 3));
}
