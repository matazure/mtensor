#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char* argv[]) {
    //构造一个10x5的二维数组
    const int rank = 2;
    int col = 10;
    int row = 5;
    pointi<rank> shape{col, row};
    tensor<float, rank> ts(shape);

    // ts关于2维坐标的赋值函数
    auto ts_setter = [ts](pointi<rank> index) {  // ts是引用拷贝
        //将ts的元素每列递增1， 每行递增10
        ts(index) = index[0] + index[1] * 10;
    };

    //遍历shape大小的所有坐标， 默认原点是(0, 0)
    for_index(ts.shape(), ts_setter);

    //将ts的元素按行输出
    for (int j = 0; j < row; ++j) {
        for (int i = 0; i < col; ++i) {
            pointi<rank> index = {i, j};
            std::cout << ts(index) << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}
