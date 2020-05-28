#include <mtensor.hpp>
#include "image_helper.hpp"

using namespace matazure;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "sample5_gradient input_image" << std::endl;
        return -1;
    }

    tensor<point3b, 2> img_rgb = read_rgb_image(argv[1]);
    auto img_gray_view = view::map(img_rgb, [](point3b rgb) {
        float scale = 1.0f / 3.0f;
        return (rgb[0] * scale) + (rgb[1] * scale) + (rgb[2] * scale);
    });

    pointi<2> padding{1, 1};
    tensor<byte, 2> image_pad_container(img_gray_view.shape() + padding * 2);
    //该操作使得img具备越界一个元素访问的能力， 因为img(-1, -1)对应着image_pad_container(0, 0)
    auto img_padding_view = view::slice(image_pad_container, padding, img_gray_view.shape());
    copy(img_gray_view, img_padding_view);
    //使用make_lambda构建梯度视图lambda_tensor
    auto img_float_view = view::cast<float>(img_padding_view);

    auto img_grad_x_view =
        view::shift(img_float_view, point2i{0, 1}) - view::shift(img_float_view, point2i{0, -1});
    auto img_grad_y_view =
        view::shift(img_float_view, point2i{1, 0}) - view::shift(img_float_view, point2i{-1, 0});
    auto grad_norm1_view = view::abs(img_grad_x_view) + view::abs(img_grad_y_view);

    //
    auto grad_mask_view = view::binary(grad_norm1_view, [](float v) { return v > 50; });
    auto img_rgb_grad_mask = view::mask(img_rgb, grad_mask_view);
    for_index(img_rgb_grad_mask.shape(), [=](point2i idx) {
        img_rgb_grad_mask(idx) = point3b{0, 255, 0};
    });
    // fill(img_rgb_grad_mask, point3b{0, 0, 0});
    //转为byte类型并固化的tensor中, 将lambda_tensor固化到tensor结构中
    // auto img_result = img_rgb_grad_mask.persist();
    //写入梯度到图像
    write_rgb_png("grad_mask.png", img_rgb);

    return 0;
}
