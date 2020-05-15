#include "sample_mandelbrot.hpp"

int main(int argc, char* argv[]) {
    auto output_mandelbrot_path = argc > 1 ? argv[1] : "mandelbrot.png";

    //区域大小
    pointi<2> shape = {2048, 2048};

    //若支持openmp， 则开启， 需要编译的时候加入-fopenmp
#ifdef MATAZURE_OPENMP
    auto img_mandelbrot_rgb = mandelbrot(shape, host_t{}).persist(omp_policy{});
#else
    auto img_mandelbrot_rgb = mandelbrot(shape, host_t{}).persist();
#endif

    write_rgb_png(output_mandelbrot_path, img_mandelbrot_rgb);

    return 0;
}
