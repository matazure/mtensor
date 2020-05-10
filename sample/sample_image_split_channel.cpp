#include <mtensor.hpp>
#include "image_helper.hpp"

using namespace matazure;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "sample_Image_split_channel input_image" << std::endl;
        return -1;
    }

    //读取图像
    std::string input_image = argv[1];
    tensor<point3b, 2> img_gray = read_rgb_image(input_image);

    //减均值
    auto img_mean = img_gray - point3b{128, 128, 128};
    //转为float类型，并归一化
    auto img_scale = view::cast<point3f>(img_mean) * point3f{0.0039f, 0.0039f, 0.0039f};

    auto width = img_gray.shape(0);
    auto height = img_gray.shape(1);
    tensor3f img_split(width, height, 3);
    //在img_split的基础上获取各个通道
    auto img_split_r = view::unstack<2>(img_split, 0);
    auto img_split_b = view::unstack<2>(img_split, 1);
    auto img_split_g = view::unstack<2>(img_split, 2);

    //拷贝数据
    for_index(img_gray.shape(), [=](point2i idx) {
        auto tmp = img_scale(idx);
        img_split_r(idx) = tmp[0];
        img_split_b(idx) = tmp[1];
        img_split_g(idx) = tmp[2];
    });

    //写入结果数据到文件中，只保存data数据
    write_raw_data("image_split_channel.raw", img_split);
    std::cout << "channel: " << 3 << "width: " << width << " height: " << height << std::endl;

    return 0;
}
