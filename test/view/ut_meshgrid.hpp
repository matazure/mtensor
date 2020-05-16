#pragma once

#include "../ut_foundation.hpp"

TEST(ViewTests, Meshgrid) {
    auto ts_meshgrid = view::meshgrid(tensor1f{0, 1}, tensor1f{2, 3, 4}).persist();

    EXPECT_TRUE(equal(ts_meshgrid(0, 0), point2f{0, 2}));
    EXPECT_TRUE(equal(ts_meshgrid(0, 1), point2f{0, 3}));
    EXPECT_TRUE(equal(ts_meshgrid(0, 2), point2f{0, 4}));
    EXPECT_TRUE(equal(ts_meshgrid(1, 0), point2f{1, 2}));
    EXPECT_TRUE(equal(ts_meshgrid(1, 1), point2f{1, 3}));
    EXPECT_TRUE(equal(ts_meshgrid(1, 2), point2f{1, 4}));
}
