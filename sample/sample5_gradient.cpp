#include <mtensor.hpp>
#include "image_helper.hpp"

using namespace matazure;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "sample5_gradient input_image" << std::endl;
        ;
        return -1;
    }

    tensor<byte, 2> img_gray = read_gray_image(argv[1]);

    pointi<2> padding{1, 1};  //需要padding避免越界
    tensor<byte, 2> img_padding_container(img_gray.shape() + 2);
    //该操作是img_padding_view具备越界访问的能力
    auto img_padding_view = view::slice(img_padding_container, padding, img_gray.shape());
    //将边界置零
    for_border(img_padding_view.shape(), padding, padding,
               [=](pointi<2> idx) { img_padding_view(idx) = 0; });
    //拷贝图像到img_padding_view
    for_index(img_padding_view.shape(),
              [=](pointi<2> idx) { img_padding_view(idx) = img_gray(idx); });

    tensor<byte, 2> img_grad(img_padding_view.shape());
    for_index(img_padding_view.shape(), [=](pointi<2> idx) {
        auto grad_x =
            img_padding_view(idx + pointi<2>{1, 0}) - img_padding_view(idx - pointi<2>{1, 0});
        auto grad_y =
            img_padding_view(idx + pointi<2>{0, 1}) - img_padding_view(idx - pointi<2>{0, 1});
        img_grad(idx) = std::abs(grad_x) + std::abs(grad_y);
    });

    write_gray_png("grad_" + std::string(argv[1]), img_grad);

    return 0;
}
