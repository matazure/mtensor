#include "image_helper.hpp"
#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "sample5_gradient input_image" << std::endl;
        ;
        return -1;
    }

    tensor<byte, 2> img_gray = read_gray_image(argv[1]);
    pointi<2> padding{1, 1};  //需要padding1 避免越界
    tensor<byte, 2, padding_layout<2>> img_padding(img_gray.shape(), padding,
                                                   padding);  //构造有padding的图像tensor
    for_border(img_padding.shape(), padding, padding,
               [=](pointi<2> idx) { img_padding(idx) = 0; });  //将边界置零

    for_index(img_padding.shape(), [=](pointi<2> idx) { img_padding(idx) = img_gray(idx); });

    tensor<byte, 2> img_grad(img_padding.shape());
    for_index(img_padding.shape(), [=](pointi<2> idx) {
        auto grad_x = img_padding(idx + pointi<2>{1, 0}) - img_padding(idx - pointi<2>{1, 0});
        auto grad_y = img_padding(idx + pointi<2>{0, 1}) - img_padding(idx - pointi<2>{0, 1});
        img_grad(idx) = std::abs(grad_x) + std::abs(grad_y);
    });

    write_gray_png("grad_" + std::string(argv[1]), img_grad);

    return 0;
}
