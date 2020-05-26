#include <mtensor.hpp>
#include "image_helper.hpp"

using namespace matazure;

int main(int argc, char* argv[]) {
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
        copy(img_conv_valid_view, img_conv);
    }

#ifdef __GNUC__
    {
        //使用gcc/clang的向量化类型申明
        typedef float value_type __attribute__((vector_size(16)));
        pointi<2> shape{256, 256};
        tensor<value_type, 2> image(shape);
        local_tensor<value_type, dim<3, 3>> kernel;
        auto img_conv_view = view::conv(image, kernel);
        pointi<2> padding = {1, 1};
        pointi<2> result_shape = image.shape() - kernel.shape() + 1;
        //不计算img_conv的边界, 避免越界
        auto img_conv_valid_view = view::slice(img_conv_view, padding, result_shape);
        tensor<value_type, 2> img_conv(result_shape);
        copy(img_conv_valid_view, img_conv);
    }
#endif

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
        tensor<value_type, 2> image(shape);
        local_tensor<value_type, dim<3, 3>> kernel;
        auto img_conv_view = view::conv(image, kernel);
        auto img_conv_stride_view = view::stride(img_conv_view, pointi<2>{2, 2});
        auto img_conv_stride_relu6_view = view::map(
            img_conv_stride_view, [](value_type v) { return std::min(std::max(0.0f, v), 6.0f); });

        pointi<2> padding = {1, 1};
        pointi<2> result_shape = image.shape() - kernel.shape() + 1;
        //不计算img_conv的边界, 避免越界
        auto img_conv_valid_view = view::slice(img_conv_stride_relu6_view, padding, result_shape);
        tensor<value_type, 2> img_conv(result_shape);
        copy(img_conv_valid_view, img_conv);
    }

    {
        struct line_functor {
            line_functor(float k, float b) : k_(k), b_(b) {}

            float operator()(float x) const { return k_ * x + b_; }

           private:
            float k_;
            float b_;
        };

        // int
        // main() {
        //     line_functor line(3.0f, 2.0f);
        //     auto y = line(10.0f);  // y = 32.0f
        // }
    }

    return 0;
}
