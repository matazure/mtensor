#include <matazure/tensor>
#include <image_utility.hpp>

using namespace matazure;

typedef pointb<3> rgb;

int main(int argc, char *argv[]) {
	//加载图像
	if (argc < 2){
		printf("please input a 3 channel(rbg) image path");
		return -1;
	}

	auto ts_rgb = read_rgb_image(argv[1]);

	//选择是否使用CUDA
#ifdef USE_CUDA
	auto gts_rgb = mem_clone(ts_rgb, device_tag{});
#else
	auto &gts_rgb = ts_rgb;
#endif
	auto center_path = section(gts_rgb, gts_rgb.shape() / 4, gts_rgb.shape() / 2);
	fill(center_path, zero<rgb>::value());

#ifdef USE_CUDA
	cuda::device_synchronize();
	auto ts_re = mem_clone(gts_rgb, host_tag{});
#else
	auto &ts_re = gts_rgb;
#endif

	write_rgb_png("mask_center.png", ts_re);

	return 0;
}
