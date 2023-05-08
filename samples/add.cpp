#include <iostream>

#include "mtensor.hpp"

using namespace matazure;

int main(int argc, char* argv[]) {
    pointi<2> shape = {2, 4};
    tensor<int, 2> ts0(shape, runtime::host);
    tensor<int, 2> ts1(shape, runtime::host);
    auto set_fun = [=](pointi<2> idx) {
        ts0(idx) = 1;
        ts1(idx) = 2;
    };
    // for_index will call set_fun(idx), for idx in grid [(0, 0), shape);
    for_index(shape, set_fun);
    // make a lambda tensor which fun(idx) = ts0(idx) + ts1(idx);
    auto lts_add = make_lambda_tensor(shape, [=](pointi<2> idx) { return ts0(idx) + ts1(idx); });
    // mtensor has lazy evaluating binary operators +-*/
    auto lts_add2 = lts_add + ts1;
    // return a host tensor(default runtime is host) ts_re that ts_re(idx) = lts_add2(x) for idx in
    // a grid of shape
    auto ts_re = lts_add2.persist();
    for (int row = 0; row < ts_re.shape()[0]; ++row) {
        for (int col = 0; col < ts_re.shape()[1]; ++col) {
            std::cout << ts_re(row, col) << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}
