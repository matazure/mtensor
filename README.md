# Tensor
tensor是matazure下的一个多维数组基础库, 具有以下的特点
* 异构计算的C++接口, 同时支持原生C++和CUDA C++
* 具备泛型(多类型,多维度)tensor，函数式lambda_tensor
* 丰富的向量化tensor操作和运算
* 延迟计算
* 内存回收
* Header Only

## 示例
下列代码分别调用CPU和GPU对输入的图像数据进行了简单的处理
```
#include <matazure/tensor>

using namespace matazure;

int main(int argc, char *argv[]) {
	tensor<point<byte, 3>, 2> ts_rgb(512, 512);
	io::read_raw_data("data/lena_rgb888_512x512.raw_data", ts_rgb);

#ifdef MATAZURE_CUDA
	auto cts_rgb = mem_clone(ts_rgb, device_t{});
	auto lcts_rgb_shift_zero = cts_rgb - point<byte, 3>{128, 128, 128};
	auto lcts_rgb_stride = stride(lcts_rgb_shift_zero, 2, 0);
	auto lcts_rgb_normalized = tensor_cast<pointf<3>>(lcts_rgb_stride) / pointf<3>{128.0f, 128.0f, 128.0f};
	auto cts_rgb_normalized = lcts_rgb_normalized.persist();
	auto ts_rgb_normalized = mem_clone(cts_rgb_normalized, host_t{});
#else
	auto lts_rgb_shift_zero = ts_rgb - point<byte, 3>{128, 128, 128};
	auto lts_rgb_stride = stride(lts_rgb_shift_zero, 2, 0);
	auto lts_rgb_normalized = tensor_cast<pointf<3>>(lts_rgb_stride) / pointf<3>{128.0f, 128.0f, 128.0f};
	auto ts_rgb_normalized = lts_rgb_normalized.persist();
#endif

	tensor<float, 2> ts_red(ts_rgb_normalized.extent());
	tensor<float, 2> ts_green(ts_rgb_normalized.extent());
	tensor<float, 2> ts_blue(ts_rgb_normalized.extent());
	auto ts_zip_point = point_view(zip(ts_red, ts_green, ts_blue));
	copy(ts_rgb_normalized, ts_zip_point);

	io::write_raw_data("data/lena_red_float_256x256.raw_data", ts_red);
	io::write_raw_data("data/lena_green_float_256x256.raw_data", ts_green);
	io::write_raw_data("data/lena_blue_float_256x256.raw_data", ts_blue);

	return 0;
}
```

## 如何使用
获取tensor项目后，将根目录加入到目标项目的头文件路径即可

## 生成项目
开关WITH_CUDA来控制是否使用CUDA  
开关WITH_BENCHMARK来控制是否使用
```
git clone https://github.com/Matazure/tensor.git
cd tensor
cd benchmark
mkdir vendors
cd vendors
git clone https://github.com/Matazure/benchmark.git
cd ../..
mkdir build
cmake ..
```

## 其他
安装[ImageMagic](http://www.imagemagick.org/)工具  
将图像转换为二进制raw data
```
convert   lena_rgb888_512x512.jpg  -depth 8 rgb:lena_rgb888_512x512.raw_data
convert   lena_rgb888_512x512.jpg  -depth 8 gray:lena_gray8_512x512.raw_data
```
将二进制raw data转换为图像
```
convert  -size 512x512 -depth 8 rgb:lena_rgb888_512x512.raw_data lena_rgb888_512x512.jpg
convert  -size 512x512 -depth 8 gray:lena_gray8_512x512.raw_data lena_gray8_512x512.jpg
```
