#include <iostream>
#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char* argv[]) {
    pointi<2> shape{5, 5};
    tensor<float, 2> ts_a(shape);
    tensor<float, 2> ts_b(shape);
    tensor<float, 2> ts_c(shape);
    fill(ts_a, 1.0f);
    fill(ts_b, 2.0f);

    //构造gpu上的tensor
    cuda::tensor<float, 2> cts_a(shape);
    cuda::tensor<float, 2> cts_b(shape);
    cuda::tensor<float, 2> cts_c(shape);

    //将cpu上的数据拷贝到gpu上
    mem_copy(ts_a, cts_a);
    mem_copy(ts_b, cts_b);

    //使用cuda  lambda算子 需要申明__device__ __host__
    auto functor = [cts_a, cts_b, cts_c] __device__ __host__(pointi<2> index) {
        cts_c(index) = cts_a(index) + cts_b(index);
    };

    cuda::for_index(shape, functor);

    //将gpu上数据拷贝会cpu
    mem_copy(cts_c, ts_c);

    //打印输出
    for_each(ts_c, [](float e) { std::cout << e << ", "; });
    std::cout << std::endl;

    return 0;
}
