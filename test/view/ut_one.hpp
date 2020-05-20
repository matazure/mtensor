#pragma once

#include "../ut_foundation.hpp"

TEST(ViewTests, One) {
    auto ts_one = view::ones<float>(point2i{2, 3}, host_t{}).persist();

    EXPECT_EQ(ts_one(0, 0), 1);
    EXPECT_EQ(ts_one(0, 1), 1);
    EXPECT_EQ(ts_one(0, 2), 1);
    EXPECT_EQ(ts_one(1, 0), 1);
    EXPECT_EQ(ts_one(1, 1), 1);
    EXPECT_EQ(ts_one(1, 2), 1);
}
