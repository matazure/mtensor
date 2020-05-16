#pragma once

#include "../ut_foundation.hpp"

TEST(ViewTests, Eye) {
    auto ts_eye = view::eye<float>(point2i{2, 2}, host_t{}, row_major_layout<2>{}).persist();

    EXPECT_EQ(ts_eye(0, 0), 1);
    EXPECT_EQ(ts_eye(0, 1), 0);
    EXPECT_EQ(ts_eye(1, 0), 0);
    EXPECT_EQ(ts_eye(1, 1), 1);
}
