#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char* argv[]) {
    //定义一个lambda算子用来描述抽象的一维数组, 其元素值等于坐标
    auto functor_a = [](int i) -> int { return i; };
    // lambda_tensor不仅需要算子， 也需要尺寸
    pointi<1> shape = {100};
    //构造lts_a， 其是一个lambda_tensor
    auto lts_a = make_lambda(shape, functor_a);

    //构造lts_b， 其元素值等于坐标的两倍
    auto functor_b = [](int i) -> int { return i * 2; };
    auto lts_b = make_lambda(shape, functor_b);

    //构造lts_a加lts_b的lambda_tensor
    auto functor_add = [lts_a, lts_b](int i) -> int { return lts_a[i] + lts_b[i]; };
    auto lts_a_add_b = make_lambda(shape, functor_add);

    //上述的定义不会执行具体的运算，当我们去获取某一个具体坐标的值时其才会真正的去调用对应的算子
    std::cout << "offset 50 value is " << lts_a_add_b[50] << std::endl;

    return 0;
}
