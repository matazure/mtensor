#include <mtensor.hpp>
#include "image_helper.hpp"

using namespace matazure;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "sample5_gradient input_image" << std::endl;
        return -1;
    }

    tensor<byte, 2> img_gray = read_gray_image(argv[1]);
    int_t padding = 1;
    tensor<byte, 2> image_pad_container(img_gray.shape() + padding * 2);
    //该操作使得img具备越界一个元素访问的能力， 因为img(-1, -1)对应着image_pad_container(0, 0)
    auto img_padding_view = view::pad(image_pad_container, padding);
    copy(img_gray, img_padding_view);
    //使用make_lambda构建梯度视图lambda_tensor
    auto img_float_view = view::cast<float>(img_padding_view);
#if 1
    auto img_grad_view = make_lambda(img_float_view.shape(), MLAMBDA(pointi<2> idx) {
        point<float, 2> grad;
        grad[0] = img_float_view(idx + pointi<2>{1, 0}) - img_float_view(idx - pointi<2>{1, 0});
        grad[1] = img_float_view(idx + pointi<2>{0, 1}) - img_float_view(idx - pointi<2>{0, 1});
        return grad;
    });
    //将梯度转为norm1
    auto grad_norm1_view = make_lambda(img_grad_view.shape(), MLAMBDA(pointi<2> idx) {
        auto grad = img_grad_view(idx);
        return std::abs(grad[0]) + std::abs(grad[1]);
    });
#else
    auto img_grad_x_view =
        view::shift(img_float_view, point2i{0, 1}) - view::shift(img_float_view, point2i{0, -1});
    auto img_grad_y_view =
        view::shift(img_float_view, point2i{1, 0}) - view::shift(img_float_view, point2i{-1, 0});
    auto grad_norm1_view = view::abs(img_grad_x_view) + view::abs(img_grad_y_view);
#endif
    //转为byte类型并固化的tensor中, 将lambda_tensor固化到tensor结构中
    auto grad_norm1 = view::cast<byte>(grad_norm1_view).persist();
    //写入梯度到图像
    write_gray_png("grad.png", grad_norm1);

    return 0;
}
