# [Tensor](https://github.com/Matazure/tensor)
Tensor是一个基于C++, CUDA的异构计算库，其上层接口极大地提高了高性能异构程序的开发效率。Tensor采用C++ AMP，Thrust的异构接口设计；具备类似Matlab的基本矩阵操作；将Eigen的延迟计算推广到GPU端；使用元编程技术追求扩展性和性能的极致。Tensor将致力于为众多异构应用提供底层支持。

## 特点
* 统一的异构编程接口
* 泛型设计，元编程支持
* 内存自动回收
* 向量化操作
* 延迟计算
* Header only
* 多平台支持

## 示例
 下面的程序演示如何使用Tensor库对两个tensor进行相加
``` cpp
#include <matazure/tensor>
using namespace matazure;

float main(){
	//申请设备端tensor
	cuda::tensor<float, 1> ts0(5);
	cuda::tensor<float, 1> ts1(ts0.shape());	
	//为tensor赋值
	//__matazure__关键字用于声明此lambda算子可以在cuda中运行
	cuda::for_index(0, ts0.size(), [=] __matazure__ (int_t i){
		ts0[i] = static_cast<float>(i);
		ts1[i] = i * 0.1f;
	});	
	//将ts0加ts1的结果存入ts2中
	cuda::tensor<float, 1> ts2(ts0.shape());
	cuda::for_index(0, ts0.size(), [=] __matazure__ (int_t i){
		ts2[i] = ts0[i] + ts1[i];
	});
	//打印结果
	cuda::for_index(0, ts2.size(), [=] __matazure__ (int_t i){
		printf("%d : %f\n", i, ts2[i]);
	});	
	//等待设备端的任务执行完毕
	cuda::device_synchronize();
	return 0;
}

```

图像预处理
``` cpp
#include <matazure/tensor>
using namespace matazure;

#ifdef MATAZURE_CUDA
#define WITH_CUDA
#endif

int main(int argc, char *argv[]) {
	//定义3个字节的rgb类型
	typedef point<byte, 3> rgb;
	//定义rbg图像
	tensor<rgb, 2> ts_rgb(512, 512);
	//将raw数据加载到ts_rgb中来
	io::read_raw_data("data/lena_rgb888_512x512.raw_data", ts_rgb);
	//选择是否使用CUDA
#ifdef WITH_CUDA
	auto gts_rgb = mem_clone(ts_rgb, device_t{});
#else
	auto &gts_rgb = ts_rgb;
#endif
	//图像像素归一化
	auto glts_rgb_shift_zero = gts_rgb - rgb{128, 128, 128};
	auto glts_rgb_stride = stride(glts_rgb_shift_zero, 2);
	auto glts_rgb_normalized = tensor_cast<pointf<3>>(glts_rgb_stride) / pointf<3>{128.0f, 128.0f, 128.0f};
	auto gts_rgb_normalized = glts_rgb_normalized.persist();
#ifdef WITH_CUDA
	auto ts_rgb_normalized = mem_clone(gts_rgb_normalized, host_t{});
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
	io::write_raw_data("data/lena_red_float_256x256.raw_data", ts_red);
	io::write_raw_data("data/lena_green_float_256x256.raw_data", ts_green);
	io::write_raw_data("data/lena_blue_float_256x256.raw_data", ts_blue);

	return 0;
}
```

## 开发环境
### 仅CPU支持
Tensor的代码规范遵循C++11标准， 所以只需编译器支持C++11即可。
### CUDA支持
在符合CPU支持的基础上，需要安装[CUDA 8.0](https://developer.nvidia.com/cuda-downloads)，详情可查看[CUDA官方文档](http://docs.nvidia.com/cuda/index.html#axzz4kQuxAvUe)

## 生成项目
先安装[git](https://git-scm.com/)和[CMake](https://cmake.org/),然后在命令行里执行
### Linux
``` sh
git clone --recursive https://github.com/Matazure/tensor.git
mkdir build
cd build
cmake ..
```
### Windows
``` sh
git clone --recursive https://github.com/Matazure/tensor.git
mkdir build
cd build
cmake .. -G "Visual Studio 14 2015 Win64"
```

## 使用
```
git clone https://github.com/Matazure/tensor.git
```
使用上面指令获取tensor项目后，将根目录（默认是tensor）加入到目标项目的头文件路径即可，无需其他库文件依赖。有关C++项目，或者CUDA项目的创建，可自行查阅网上众多的资源。

## 工具
为了尽可能的减少第三方库的依赖，示例会直接使用raw数据，我们可以借助[ImageMagic](http://www.imagemagick.org/)来转换图像和raw数据  
将图像转换为raw数据
```
convert   lena_rgb888_512x512.jpg  -depth 8 rgb:lena_rgb888_512x512.raw_data
convert   lena_rgb888_512x512.jpg  -depth 8 gray:lena_gray8_512x512.raw_data
```
将raw数据转换为图像
```
convert  -size 512x512 -depth 8 rgb:lena_rgb888_512x512.raw_data lena_rgb888_512x512.jpg
convert  -size 512x512 -depth 8 gray:lena_gray8_512x512.raw_data lena_gray8_512x512.jpg
```
## 许可证书
该项目使用MIT证书授权，具体可查看LICENSE文件

## 联系方式
原作者希望更多的人加入到Tensor的使用开发中来，若在使用上有迷惑的地方，可直接通过邮件联系，周末可加QQ沟通  
邮箱： p3.1415@qq.com  
QQ： 417083997
