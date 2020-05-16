#pragma once

#include "../ut_foundation.hpp"

#include <tuple>

TEST(ViewTests, Zip) {
    // clang-format off
    tensor<float, 1> ts0 = { 0, 1 ,2 ,3};
    tensor<float, 1> ts1 = { 4, 5, 6, 7};
    
    auto ts_zip = view::zip(ts0, ts1);

    ts_zip[1] = tuple<float, float>{10, 20};

    EXPECT_EQ(10, ts0[1]);
    EXPECT_EQ(20, ts1[1]);
}
