#include "ut_foundation.hpp"

TEST(TensorTests, ConstructByInitializerList) {
    tensor<int, 2> ts = {{1, 2, 3},  //
                         {4, 5, 6},
                         {7, 8, 9}};

    EXPECT_EQ(ts(pointi<2>{0, 0}), 1);
    EXPECT_EQ(ts(pointi<2>{1, 0}), 2);
    EXPECT_EQ(ts(pointi<2>{2, 0}), 3);
    EXPECT_EQ(ts(pointi<2>{0, 1}), 4);
    EXPECT_EQ(ts(pointi<2>{1, 1}), 5);
    EXPECT_EQ(ts(pointi<2>{2, 1}), 6);
    EXPECT_EQ(ts(pointi<2>{0, 2}), 7);
    EXPECT_EQ(ts(pointi<2>{1, 2}), 8);
    EXPECT_EQ(ts(pointi<2>{2, 2}), 9);
}

TEST(TensorTests, ConstructByZero) {
    {
        tensor<int, 2> ts(pointi<2>{0, 0});
        for (int i = 0; i < ts.shape().size(); ++i) {
            const auto& shape = ts.shape();
            EXPECT_EQ(shape[i], 0);
        }
    }

    {
        tensor<int, 2> ts;
        for (int i = 0; i < ts.shape().size(); ++i) {
            const auto& shape = ts.shape();
            EXPECT_EQ(shape[i], 0);
        }
    }

    {
        tensor<int, 2> ts{};
        for (int i = 0; i < ts.shape().size(); ++i) {
            const auto& shape = ts.shape();
            EXPECT_EQ(shape[i], 0);
        }
    }
}