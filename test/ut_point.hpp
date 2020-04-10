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