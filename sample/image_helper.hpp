#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image.h>
#include <stb_image_write.h>
#include <mtensor.hpp>
#include <string>

using namespace matazure;

inline matazure::tensor<matazure::pointb<3>, 2> read_rgb_image(const std::string& image_path) {
    using namespace matazure;

    pointi<3> shape{};
    auto data = stbi_load(image_path.c_str(), &shape[2], &shape[1], &shape[3], false);
    if (shape[3] != 3 || !data) {
        printf("need a 3 channel image");
        throw std::runtime_error("only support 3 channel rgb image");
    }

    typedef point<byte, 3> rgb;
    tensor<rgb, 2> ts_rgb(
        pointi<2>{shape[1], shape[2]},
        shared_ptr<rgb>(reinterpret_cast<rgb*>(data), [](rgb* p) { stbi_image_free(p); }));

    return ts_rgb;
}

inline matazure::tensor<byte, 2> read_gray_image(const std::string& image_path) {
    auto ts_rgb = read_rgb_image(image_path);
    tensor<byte, 2> ts_gray(ts_rgb.shape());
    transform(ts_rgb, ts_gray, [](pointb<3> rgb) {
        float scale = 1.0f / 3.0f;
        return (rgb[0] * scale) + (rgb[1] * scale) + (rgb[2] * scale);
    });

    return ts_gray;
}

void write_rgb_png(const std::string& image_path, matazure::tensor<matazure::pointb<3>, 2> ts_rgb) {
    auto re = stbi_write_png(image_path.c_str(), ts_rgb.shape()[1], ts_rgb.shape()[0], 3,
                             ts_rgb.data(), ts_rgb.shape()[1] * 3);
    if (re == 0) {
        throw std::runtime_error("failed to write ts_rgb to png image, error code: " +
                                 std::to_string(re));
    }
}

void write_gray_png(const std::string& image_path, matazure::tensor<byte, 2> ts_gray) {
    tensor<pointb<3>, 2> ts_rgb(ts_gray.shape());
    transform(ts_gray, ts_rgb, [](byte b) { return pointb<3>{b, b, b}; });
    write_rgb_png(image_path, ts_rgb);
}
