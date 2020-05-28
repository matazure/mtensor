#include "../ut_foundation.hpp"

TEST(ViewTests, Abs) {
    cuda::tensor1f ts(1000);
    fill(ts, -1.0f);
    auto ts_abs_view = view::abs(ts);
    for_each(ts_abs_view, [] MATAZURE_GENERAL(float v) {
        if ((v - 1.0f > 0.0001f) || (v - 1.0f < -0.0001f)) {
            // for triger a exception
            int* p = nullptr;
            *p = 1;
        }

    });
}
