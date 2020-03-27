#include "bm_config.hpp"

constexpr int element_size = 32;

typedef float value_type __attribute__((vector_size(4 * element_size)));

void matrix_vector_multiply(tensor<value_type, 2> mat, tensor<value_type, 1> vec,
                            tensor<value_type, 1> vec_re) {
    auto col = mat.shape()[0];
    auto row = mat.shape()[1];
    // tensor<value_type, 1> vec_re(row);
    for (int r = 0; r < row; r += 1) {
        value_type re = {0};
        value_type re1 = {0};
        value_type re2 = {0};
        value_type re3 = {0};
        // value_type re4 = {0};

        auto p_mat = &mat(r, 0);
        auto p_vec = &vec(0);
        for (int c = 0; c < col; c++) {
            re += mat(c, r) * vec(c);
            // re1 += mat(c, r + 1) * vec(c);
            // re2 += mat(c, r + 2) * vec(c);
            // re3 += mat(c, r + 3) * vec(c);
            // re4 += mat(r + 4, c) * vec(c);
        }
        vec_re(r) = re;
        // vec_re(r + 1) = re1;
        // vec_re(r + 2) = re2;
        // vec_re(r + 3) = re3;
        // vec_re(r + 4) = re4;
    }
}

void bm_matrix_vector_multiply(benchmark::State& state) {
    auto col = state.range(0);
    auto row = state.range(1);

    tensor<value_type, 2> mat(col, row);
    tensor<value_type, 1> vec(col);
    tensor<value_type, 1> vec_re(row);

    while (state.KeepRunning()) {
        matrix_vector_multiply(mat, vec, vec_re);
    }

    auto byte_size = (col * row) * sizeof(value_type);
    auto item_size = col * row * 2 * element_size;
    state.SetBytesProcessed(state.iterations() * byte_size);
    state.SetItemsProcessed(state.iterations() * item_size);
}

BENCHMARK(bm_matrix_vector_multiply)->Args({8, 8});

BENCHMARK_MAIN();