#include "../ut_foundation.hpp"

#include <vector>

TEST(TensorTests, Construct) {
    {
        cuda::tensor<int, 2> ts(pointi<2>{0, 0});
        for (int i = 0; i < ts.shape().size(); ++i) {
            const auto& shape = ts.shape();
            EXPECT_EQ(shape[i], 0);
        }
    }

    {
        cuda::tensor<int, 2> ts;
        for (int i = 0; i < ts.shape().size(); ++i) {
            const auto& shape = ts.shape();
            EXPECT_EQ(shape[i], 0);
        }
    }

    {
        cuda::tensor<int, 2> ts{};
        for (int i = 0; i < ts.shape().size(); ++i) {
            const auto& shape = ts.shape();
            EXPECT_EQ(shape[i], 0);
        }
    }

    {
        cuda::tensor<int, 2> ts(1000, 1000);
        for (int i = 0; i < ts.shape().size(); ++i) {
            const auto& shape = ts.shape();
            EXPECT_EQ(shape[i], 1000);
        }
    }
}
