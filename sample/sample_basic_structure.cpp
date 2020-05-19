#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char* argv[]) {
    point<float, 3> pt = {0.0f, 1.0f, 2.0f};
    std::cout << "pt: " << pt << std::endl;
    std::cout << "pt offset 1 value: " << pt[1] << std::endl;

    tensor<int, 2> ts = {{0, 1}, {2, 3}, {4, 5}};
    std::cout << "ts: " << std::endl << ts << std::endl;
    std::cout << "ts linear access 3 value : " << ts[3] << std::endl;
    auto idx = pointi<2>{1, 1};
    std::cout << "ts array access " << idx << " value : " << ts(idx) << std::endl;

    return 0;
}
