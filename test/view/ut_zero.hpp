#pragma once

#include "../ut_foundation.hpp"

TEST(ViewTests, Zero) {
    auto ts_one = view::zeros<float>(point2i{2, 3}, host_t{}).persist();

    EXPECT_EQ(ts_one(0, 0), 0);
    EXPECT_EQ(ts_one(0, 1), 0);
    EXPECT_EQ(ts_one(0, 2), 0);
    EXPECT_EQ(ts_one(1, 0), 0);
    EXPECT_EQ(ts_one(1, 1), 0);
    EXPECT_EQ(ts_one(1, 2), 0);
}
