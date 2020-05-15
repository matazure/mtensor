#include "ut_foundation.hpp"

TEST(TensorTests, ConstructByInitializerList) {
    tensor<int, 2> ts = {{1, 2, 3},  //
                         {4, 5, 6},
                         {7, 8, 9}};

    EXPECT_EQ(ts(pointi<2>{0, 0}), 1);
    EXPECT_EQ(ts(pointi<2>{0, 1}), 2);
    EXPECT_EQ(ts(pointi<2>{0, 2}), 3);
    EXPECT_EQ(ts(pointi<2>{1, 0}), 4);
    EXPECT_EQ(ts(pointi<2>{1, 1}), 5);
    EXPECT_EQ(ts(pointi<2>{1, 2}), 6);
    EXPECT_EQ(ts(pointi<2>{2, 0}), 7);
    EXPECT_EQ(ts(pointi<2>{2, 1}), 8);
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

TEST(TensorTests, Layout) {
    tensor<float, 2, row_major_layout<2>> ts_row{{00, 01, 02}, {10, 11, 12}};
    ASSERT_EQ(2, ts_row.shape(0));
    ASSERT_EQ(3, ts_row.shape(1));
    ASSERT_EQ(0, ts_row[0]);
    ASSERT_EQ(1, ts_row[1]);
    ASSERT_EQ(2, ts_row[2]);
    ASSERT_EQ(10, ts_row[3]);
    ASSERT_EQ(11, ts_row[4]);
    ASSERT_EQ(12, ts_row[5]);

    tensor<float, 2, column_major_layout<2>> ts_column{{00, 01, 02}, {10, 11, 12}};
    ASSERT_EQ(2, ts_column.shape(0));
    ASSERT_EQ(3, ts_column.shape(1));
    ASSERT_EQ(0, ts_column[0]);
    ASSERT_EQ(10, ts_column[1]);
    ASSERT_EQ(1, ts_column[2]);
    ASSERT_EQ(11, ts_column[3]);
    ASSERT_EQ(2, ts_column[4]);
    ASSERT_EQ(12, ts_column[5]);
}