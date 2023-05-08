#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <chrono>
#include <iostream>

#include "mtensor.hpp"
#include "stb_image_write.h"

using namespace matazure;
using rgb = point<uint8_t, 3>;

int main(int argc, char* argv[]) {
    pointi<2> shape = {2048, 2048};
    int_t max_iteration = 256 * 16;
    // make a lambda tensor to evaluate the mandelbrot set.
    auto lts_mandelbrot = make_lambda_tensor(shape, [=] __general__(pointi<2> idx) -> float {
        point<float, 2> idxf;
        idxf[0] = idx[0];
        idxf[1] = idx[1];
        point<float, 2> shapef;
        shapef[0] = shape[0];
        shapef[1] = shape[1];
        point<float, 2> c =
            idxf / shapef * point<float, 2>{3.25f, 2.5f} - point<float, 2>{2.0f, 1.25f};
        auto z = point<float, 2>::all(0.0f);
        auto norm = 0.0f;
        int_t value = 0;
        while (norm <= 4.0f && value < max_iteration) {
            float tmp = z[0] * z[0] - z[1] * z[1] + c[0];
            z[1] = 2 * z[0] * z[1] + c[1];
            z[0] = tmp;
            ++value;
            norm = z[0] * z[0] + z[1] * z[1];
        }

        return value;
    });

    // convert mandelbrot value to rgb pixel
    auto lts_rgb_mandelbrot =
        make_lambda_tensor(lts_mandelbrot.shape(), [=] __general__(pointi<2> idx) {
            float t = lts_mandelbrot(idx) / max_iteration;
            auto r = static_cast<uint8_t>(36 * (1 - t) * t * t * t * 255);
            auto g = static_cast<uint8_t>(60 * (1 - t) * (1 - t) * t * t * 255);
            auto b = static_cast<uint8_t>(38 * (1 - t) * (1 - t) * (1 - t) * t * 255);
            return rgb{r, g, b};
        });

    // select runtime
    runtime rt = runtime::cuda;
    if (argc > 1 && std::string(argv[1]) == "host") {
        rt = runtime::host;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    // persist lambda tensor on cuda/host runtime
    auto ts_rgb_mandelbrot = lts_rgb_mandelbrot.persist(rt);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "render mandelbrot cost time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms"
              << std::endl;

    // sync ts_rgb_mandelbrot to host. if it's host alread, return itself
    ts_rgb_mandelbrot = ts_rgb_mandelbrot.sync(runtime::host);
    stbi_write_png("mandelbrot.png", ts_rgb_mandelbrot.shape()[1], ts_rgb_mandelbrot.shape()[0], 3,
                   ts_rgb_mandelbrot.data(), ts_rgb_mandelbrot.shape()[1] * 3);

    return 0;
}
