#include <mtensor.hpp>
#include "image_helper.hpp"

using namespace matazure;

int main(int argc, char* argv[]) {
// {
//     //使用gcc/clang的向量化类型申明
//     typedef float value_type __attribute__((vector_size(16)));
//     pointi<2> shape{256, 256};
//     tensor<value_type, 2> image(shape);
//     local_tensor<value_type, dim<3, 3>> kernel;
//     auto img_conv_view = view::conv(image, kernel);
//     pointi<2> padding = {1, 1};
//     pointi<2> result_shape = image.shape() - kernel.shape() + 1;
//     //不计算img_conv的边界, 避免越界
//     auto img_conv_valid_view = view::slice(img_conv_view, padding, result_shape);
//     tensor<value_type, 2> img_conv(result_shape);
//     copy(img_conv_valid_view, img_conv);
// }
#ifdef MATAZURE_OPENMP
    {
        typedef float value_type;
        pointi<2> shape{256, 256};
        tensor<value_type, 2> image(shape);
        local_tensor<value_type, dim<3, 3>> kernel;
        auto img_conv_view = view::conv(image, kernel);
        pointi<2> padding = {1, 1};
        pointi<2> result_shape = image.shape() - kernel.shape() + 1;
        //不计算img_conv的边界, 避免越界
        auto img_conv_valid_view = view::slice(img_conv_view, padding, result_shape);
        tensor<value_type, 2> img_conv(result_shape);
        copy(omp_policy{}, img_conv_valid_view, img_conv);
    }
#endif
    {
        typedef float value_type;
        pointi<2> shape{256, 256};
        cuda::tensor<value_type, 2> image(shape);
        local_tensor<value_type, dim<3, 3>> kernel;
        auto img_conv_view = view::conv(image, kernel);
        pointi<2> padding = {1, 1};
        pointi<2> result_shape = image.shape() - kernel.shape() + 1;
        //不计算img_conv的边界, 避免越界
        auto img_conv_valid_view = view::slice(img_conv_view, padding, result_shape);
        cuda::tensor<value_type, 2> img_conv(result_shape);
        copy(img_conv_valid_view, img_conv);
    }

    return 0;
}