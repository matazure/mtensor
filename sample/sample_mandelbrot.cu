#include "sample_mandelbrot.hpp"

int main(int argc, char* argv[]) {
    auto output_mandelbrot_path = argc > 1 ? argv[1] : "mandelbrot.png";

    //区域大小
    pointi<2> shape = {2048, 2048};

    // auto img_mandelbrot_rgb_view = mandelbrot(shape, device_t{});
    // cuda::tensor<pointb<3>, 2> cu_img_mandelbrot_rgb(img_mandelbrot_rgb_view.shape());
    // copy(img_mandelbrot_rgb_view, cu_img_mandelbrot_rgb);

    // 下面这行代码和上面注释掉的三行等价
    // persist相当于把lambda_tensor里的结果计算并写入到相应设备的tensor内存中
    // auto cu_img_mandelbrot_rgb = mandelbrot(shape, device_t{}).persist();

    //使用自定义的cuda资源
    cuda::execution_policy policy;
    policy.block_dim({1024, 1, 1});
    policy.grid_dim({34, 1, 1});
    auto cu_img_mandelbrot_rgb = mandelbrot(shape, device_t{}).persist(policy);

    auto img_mandelbrot_rgb = identify(cu_img_mandelbrot_rgb, host_t{});
    write_rgb_png(output_mandelbrot_path, img_mandelbrot_rgb);

    return 0;
}
