#pragma once

#include "../ut_foundation.hpp"

TEST(ViewTests, Permute) {
    // clang-format off
    tensor<float, 2> ts =
     {{0, 1, 2, 3}, 
      {4, 5, 6, 7}, 
      {8, 9, 10, 11}, 
      {12, 13, 14, 15}};
    // clang-format on

    //转置参数输在模板参数里
    auto ts_permute_view = view::permute<1, 0>(ts);
    auto ts_permute = ts_permute_view.persist();

    for_index(ts.shape(), [=](point2i idx) { EXPECT_EQ(ts(idx), ts_permute(idx[1], idx[0])); });
}
