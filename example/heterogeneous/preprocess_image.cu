#include <matazure/tensor>
using namespace matazure;

#ifdef USE_CUDA
#ifndef MATAZURE_CUDA
#error "does not support cuda"
#endif
#endif

int main(int argc, char *argv[]) {
	//定义3个字节的rgb类型
	typedef point<byte, 3> rgb;
	//定义rbg图像
	tensor<rgb, 2> ts_rgb(512, 512);
	//将raw数据加载到ts_rgb中来
	io::read_raw_data("lena_rgb888_512x512.raw_data", ts_rgb);
	//选择是否使用CUDA
#ifdef USE_CUDA
	auto gts_rgb = mem_clone(ts_rgb, device{});
#else
	auto &gts_rgb = ts_rgb;
#endif
	//图像像素归一化
	auto glts_rgb_shift_zero = gts_rgb - rgb::all(128);
	auto glts_rgb_stride = stride(glts_rgb_shift_zero, 2);
	auto glts_rgb_normalized = tensor_cast<pointf<3>>(glts_rgb_stride) / pointf<3>::all(128.0f);
	//前面并未进行实质的计算，这一步将上面的运算合并处理并把结果写入到memory中, 避免了额外的内存开销
	auto gts_rgb_normalized = glts_rgb_normalized.persist();
#ifdef USE_CUDA
	cuda::device_synchronize();
	auto ts_rgb_normalized = mem_clone(gts_rgb_normalized, host{});
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
	io::write_raw_data("lena_red_float_256x256.raw_data", ts_red);
	io::write_raw_data("lena_green_float_256x256.raw_data", ts_green);
	io::write_raw_data("lena_blue_float_256x256.raw_data", ts_blue);

	return 0;
}
