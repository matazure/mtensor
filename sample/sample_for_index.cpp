#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char* argv[]) {
    pointi<2> shape{2, 3};
    tensor<float, 2> ts_a(shape);
    tensor<float, 2> ts_b(shape);
    tensor<float, 2> ts_c(shape);
    fill(ts_a, 1.0f);
    fill(ts_b, 2.0f);
    auto functor = MLAMBDA(pointi<2> idx) { ts_c(idx) = ts_a(idx) + ts_b(idx); };
    // 计算
    for_index(shape, functor);
    // 输出结果
    std::cout << ts_c << std::endl;

    return 0;
}
