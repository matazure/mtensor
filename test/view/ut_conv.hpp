#pragma once

#include "../ut_foundation.hpp"

TEST(ViewTests, Conv) {
    pointi<2> shape = {10, 10};
    pointi<2> padding = {1, 1};
    tensor<float, 2> ts(shape + padding * 2);
    fill(ts, 1.0f);

    auto ts_pad_view = view::slice(ts, padding, shape);
    local_tensor<float, dim<3, 3>> kernel;
    fill(kernel, 0.7f);

    auto ts_conv_view = view::conv(ts_pad_view, kernel);

    for_each(ts_conv_view, [&](float v) { ASSERT_NEAR(0.7 * kernel.size(), v, 0.01f); });
}
