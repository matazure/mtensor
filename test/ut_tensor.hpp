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
    std::cout << "row major layout: " << std::endl;
    for (int_t i = 0; i < ts_row.size(); ++i) {
        std::cout << ts_row[i] << ", ";
    }
    std::cout << std::endl;

    tensor<float, 2, column_major_layout<2>> ts_column{{00, 01, 02}, {10, 11, 12}};
    ASSERT_EQ(2, ts_row.shape(0));
    ASSERT_EQ(3, ts_row.shape(1));
    std::cout << "column major layout: " << std::endl;
    for (int_t i = 0; i < ts_column.size(); ++i) {
        std::cout << ts_column[i] << ", ";
    }
    std::cout << std::endl;
}