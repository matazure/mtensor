#include "ut_foundation.hpp"

TEST(LocalTensorTests, ColumnLayout) {
    // mem continue initialize the local tensor
    local_tensor<float, dim<3, 4>, column_major_layout<2>> lt = {0, 1, 2, 3, 4,  5,
                                                                 6, 7, 8, 9, 10, 11};

    for (int_t i = 0; i < lt.shape(0); ++i) {
        for (int_t j = 0; j < lt.shape(1); ++j) {
            std::cout << lt(i, j) << ", ";
        }
        std::cout << std::endl;
    }
}

TEST(LocalTensorTests, RowLayout) {
    // mem continue initialize the local tensor
    local_tensor<float, dim<3, 4>, row_major_layout<2>> lt = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    for (int_t i = 0; i < lt.shape(0); ++i) {
        for (int_t j = 0; j < lt.shape(1); ++j) {
            std::cout << lt(i, j) << ", ";
        }
        std::cout << std::endl;
    }
}