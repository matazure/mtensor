#pragma once

#include "../ut_foundation.hpp"

TEST(ViewTests, Slice) {
    // clang-format off
    tensor<float, 2> ts =
     {{0, 1, 2, 3}, 
      {4, 5, 6, 7}, 
      {8, 9, 10, 11}, 
      {12, 13, 14, 15}};
    // clang-format on
    auto ts_slice_view = view::slice(ts, point2i{1, 1}, point2i{2, 2});

    EXPECT_EQ(5, ts_slice_view(0, 0));
    EXPECT_EQ(6, ts_slice_view(0, 1));
    EXPECT_EQ(9, ts_slice_view(1, 0));
    EXPECT_EQ(10, ts_slice_view(1, 1));
}
