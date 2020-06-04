#include <mtensor.hpp>
#include "image_helper.hpp"

using std::is_convertible;

using namespace matazure;

int main(int argc, char* argv[]) {
    tensor<point3b, 2> img_rgb(100, 100);
    auto color_blue = point3b{0, 0, 255};
    fill(img_rgb, color_blue);

    tensor<bool, 2> img_mask(img_rgb.shape());
    fill(img_mask, false);
    fill(view::pad(img_mask, 5), true);
    auto color_red = point3b{255, 0, 0};
    // mask会阻止你往被mask的对象里写东西
    fill(view::mask(img_rgb, img_mask), color_red);

    write_rgb_png("grad_mask.png", img_rgb);

    return 0;
}
