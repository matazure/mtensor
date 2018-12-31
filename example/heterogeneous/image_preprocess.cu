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
	//图像像素归一化
	auto glts_rgb_shift_zero = gts_rgb - rgb::all(128);
	auto glts_rgb_stride = stride(glts_rgb_shift_zero, 2);
	auto glts_rgb_normalized = cast<pointf<3>>(glts_rgb_stride) / pointf<3>::all(128.0f);
	//前面并未进行实质的计算，这一步将上面的运算合并处理并把结果写入到memory中, 避免了额外的内存开销
	auto gts_rgb_normalized = glts_rgb_normalized.persist();
#ifdef USE_CUDA
	cuda::device_synchronize();
	auto ts_rgb_normalized = mem_clone(gts_rgb_normalized, host_tag{});
#else
	auto &ts_rgb_normalized = gts_rgb_normalized;
#endif
	//定义三个通道的图像数据
	tensor<float, 2> ts_red(ts_rgb_normalized.shape());
	tensor<float, 2> ts_green(ts_rgb_normalized.shape());
	tensor<float, 2> ts_blue(ts_rgb_normalized.shape());
	//zip操作，就返回tuple数据，tuple的元素为上面三个通道对应元素的引用
	auto ts_zip_rgb = zip(ts_red, ts_green, ts_blue);
	//让tuple元素可以和point<byte, 3>可以相互转换
	auto ts_zip_point = point_view(ts_zip_rgb);
	//拷贝结果到ts_red, ts_green, ts_blue中，因为ts_zip_point的元素是指向这三个通道的引用
	copy(ts_rgb_normalized, ts_zip_point);

	//保存raw数据
	auto output_red_path = argc < 3 ? "red.raw_data" : argv[2];
	auto output_green_path = argc < 4 ? "green.raw_data" : argv[3];
	auto output_blue_path = argc < 5 ? "blue.raw_data" : argv[4];
	io::write_raw_data(output_red_path, ts_red);
	io::write_raw_data(output_green_path, ts_green);
	io::write_raw_data(output_blue_path, ts_blue);

	return 0;
}
