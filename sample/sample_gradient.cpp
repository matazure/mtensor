#include <mtensor.hpp>
#include "image_helper.hpp"

using namespace matazure;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "sample5_gradient input_image" << std::endl;
        return -1;
    }

    tensor<byte, 2> img_gray = read_gray_image(argv[1]);

    pointi<2> padding{1, 1};  //需要扩展一个像素，避免越界
    tensor<byte, 2> image_pad_container(img_gray.shape() + padding * 2);
    //该操作使得img具备越界一个元素访问的能力， 因为img(-1, -1)对应着image_pad_container(0, 0)
    auto img_padding_view = view::slice(image_pad_container, padding, img_gray.shape());
    copy(img_gray, img_padding_view);

    //类型转换
    auto img_float_view = view::cast<float>(img_padding_view);

    auto img_grad_view = make_lambda(img_float_view.shape(), [=](pointi<2> idx) {
        point<byte, 2> grad;
        grad[0] = img_float_view(idx + pointi<2>{1, 0}) - img_float_view(idx - pointi<2>{1, 0});
        grad[1] = img_float_view(idx + pointi<2>{0, 1}) - img_float_view(idx - pointi<2>{0, 1});
        return grad;
    });

    auto img_grad_norm1_view = make_lambda(img_grad_view.shape(), [=](pointi<2> idx) {
        auto grad = img_grad_view(idx);
        return std::abs(grad[0]) + std::abs(grad[1]);
    });

    auto img_grad_norm1_byte_view = view::cast<byte>(img_grad_norm1_view);
    tensor<byte, 2> img_grad_norm1(img_float_view.shape());
    // copy会遍历拷贝，在索引元素时会执行计算
    copy(img_grad_norm1_byte_view, img_grad_norm1);

    //写入梯度到图像，
    //因为我们未对image_pad_container的边界做适当处理，故梯度图像的边界是不合理的
    write_gray_png("grad.png", img_grad_norm1);

    return 0;
}
