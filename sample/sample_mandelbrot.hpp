#pragma once

#include <mtensor.hpp>
#include <stdexcept>
#include "image_helper.hpp"
#include "sample_mandelbrot.hpp"

using namespace matazure;

struct mandelbrot_functor {
    mandelbrot_functor(pointi<2> shape) : shape(shape) {}

    //曼德勃罗算子， MATAZURE_GENERAL可以同时支持host和device
    MATAZURE_GENERAL point<byte, 3> operator()(pointi<2> idx) const {
        pointf<2> c = point_cast<float>(idx) / point_cast<float>(shape) * pointf<2>{3.25f, 2.5f} -
                      pointf<2>{2.0f, 1.25f};
        auto z = pointf<2>::all(0.0f);
        auto norm = 0.0f;
        int_t value = 0;
        while (norm <= 4.0f && value < max_iteration) {
            float tmp = z[0] * z[0] - z[1] * z[1] + c[0];
            z[1] = 2 * z[0] * z[1] + c[1];
            z[0] = tmp;
            ++value;
            norm = z[0] * z[0] + z[1] * z[1];
        }

        //返回rgb的像素值
        float t = float(value) / max_iteration;
        auto r = static_cast<byte>(36 * (1 - t) * t * t * t * 255);
        auto g = static_cast<byte>(60 * (1 - t) * (1 - t) * t * t * 255);
        auto b = static_cast<byte>(38 * (1 - t) * (1 - t) * (1 - t) * t * 255);
        return point<byte, 3>{r, g, b};
    }

   private:
    pointi<2> shape;
    int_t max_iteration = 256 * 16;
};

template <typename runtime_type>
auto mandelbrot(pointi<2> shape, runtime_type mem_tag)
    -> decltype(make_lambda(shape, mandelbrot_functor(shape), mem_tag)) {
    return make_lambda(shape, mandelbrot_functor(shape), mem_tag);
}