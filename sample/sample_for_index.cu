#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char* argv[]) {
    pointi<2> shape{2, 3};
    cuda::tensor<float, 2> ts_a(shape);
    cuda::tensor<float, 2> ts_b(shape);
    cuda::tensor<float, 2> ts_c(shape);
    fill(ts_a, 1.0f);
    fill(ts_b, 2.0f);
    //使用cuda  lambda算子 需要申明__device__
    auto functor = MLAMBDA(pointi<2> idx) { ts_c(idx) = ts_a(idx) + ts_b(idx); };
    // 计算
    cuda::for_index(shape, functor);
    // 拷贝到主机tensor, 输出结果
    tensor<float, 2> ts_re(shape);
    mem_copy(ts_c, ts_re);
    std::cout << ts_re << std::endl;

    return 0;
}
