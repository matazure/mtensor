#pragma once

#include "../ut_foundation.hpp"

TEST(ViewTests, Gather) {
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
