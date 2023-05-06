#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image.h>
#include <stb_image_write.h>

#include <mtensor.hpp>
#include <string>

using namespace matazure;

void write_rgb_png(const std::string& image_path, matazure::tensor<matazure::pointb<3>, 2> ts_rgb) {
    auto re = stbi_write_png(image_path.c_str(), ts_rgb.shape()[1], ts_rgb.shape()[0], 3, ts_rgb.data(),
                             ts_rgb.shape()[1] * 3);
    if (re == 0) {
        throw std::runtime_error("failed to write ts_rgb to png image, error code: " + std::to_string(re));
    }
}
