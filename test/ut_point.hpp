#pragma once

#include "ut_foundation.hpp"

TEST(PointTests, AssignmentOperator) {
    pointi<2> tmp;

    tmp += tmp;
    tmp -= tmp;
    tmp *= tmp;
    tmp /= tmp;
    tmp %= tmp;
}

TEST(PointTests, Reverse) {
    pointi<4> pt{0, 1, 2, 3};
    auto pt_re = reverse_point(pt);
    pointi<4> pt_gold{3, 2, 1, 0};
    EXPECT_TRUE(equal(pt_gold, pt_re));
}