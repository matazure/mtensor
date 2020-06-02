#include <mtensor.hpp>
#include "image_helper.hpp"

using namespace matazure;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "sample_Image_split_channel input_image" << std::endl;
        return -1;
    }

    tensor<point3b, 2> img_gray = read_rgb_image(argv[1]);                     //读取图像
    auto img_float_view = view::cast<point3f>(img_gray);                       //转为point3f类型
    auto img_mean_view = img_float_view - point3f{128, 128, 128};              //减均值
    auto img_scale_view = img_mean_view * point3f{0.0039f, 0.0039f, 0.0039f};  //并归一化

    tensor3f img_split(3, img_gray.shape(0), img_gray.shape(1));
    //在img_split的基础上获取各个通道
    auto img_split_r_view = view::gather<2>(img_split, 0);  // red通道视图
    auto img_split_b_view = view::gather<2>(img_split, 1);  // blue通道视图
    auto img_split_g_view = view::gather<2>(img_split, 2);  // green通道视图
    //拷贝数据到img_split中
    for_index(img_gray.shape(), MLAMBDA(point2i idx) {
        auto tmp = img_scale_view(idx);
        img_split_r_view(idx) = tmp[0];
        img_split_b_view(idx) = tmp[1];
        img_split_g_view(idx) = tmp[2];
    });

    //写入结果数据到文件中，只保存data数据
    write_raw_data("image_split_channel.raw", img_split);
    std::cout << "channel: " << 3 << "width: " << img_gray.shape(1)
              << " height: " << img_gray.shape(1) << std::endl;

    return 0;
}
