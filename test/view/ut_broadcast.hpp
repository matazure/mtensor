#pragma once

#include "../ut_foundation.hpp"

TEST(ViewTests, Broadcast) {
    tensor1f ts = {0, 1, 2};
    auto ts_broadcast = view::broadcast(ts, pointi<2>{2, 3});

    EXPECT_EQ(ts_broadcast(0, 0), 0);
    EXPECT_EQ(ts_broadcast(0, 1), 1);
    EXPECT_EQ(ts_broadcast(0, 2), 2);
    EXPECT_EQ(ts_broadcast(1, 0), 0);
    EXPECT_EQ(ts_broadcast(1, 1), 1);
    EXPECT_EQ(ts_broadcast(1, 2), 2);
}
