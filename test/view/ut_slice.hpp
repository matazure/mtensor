#include "../ut_foundation.hpp"

TEST(ViewTests, Slice) {
    tensor<float, 2> ts(pointi<2>{4, 4});
    ts(0, 0) = 1;
    ts(0, 1) = 2;
    ts(0, 2) = 3;
    ts(0, 3) = 4;
    ts(1, 0) = 11;
    ts(1, 1) = 12;
    ts(1, 2) = 13;
    ts(1, 3) = 14;
    ts(2, 0) = 21;
    ts(2, 1) = 22;
    ts(2, 2) = 23;
    ts(2, 3) = 24;
    ts(3, 0) = 31;
    ts(3, 1) = 32;
    ts(3, 2) = 33;
    ts(3, 3) = 34;

    auto ts_re = view::stride(ts, 2).persist();

    for_each(ts_re, [](float e) { printf("%f, ", e); });

    printf("\n");
}
