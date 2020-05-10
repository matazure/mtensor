#include "sample_mandelbrot.hpp"

int main(int argc, char* argv[]) {
    auto output_mandelbrot_path = argc > 1 ? argv[1] : "mandelbrot.png";

    //区域大小
    pointi<2> shape = {2048, 2048};

    auto cu_img_mandelbrot_rgb = mandelbrot(shape, device_tag{}).persist();

    auto img_mandelbrot_rgb = mem_clone(cu_img_mandelbrot_rgb, host_tag{});
    write_rgb_png(output_mandelbrot_path, img_mandelbrot_rgb);

    return 0;
}
