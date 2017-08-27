# [Tensor](https://github.com/Matazure/tensor) [![Build Status](https://travis-ci.org/Matazure/tensor.svg?branch=master)](https://travis-ci.org/Matazure/tensor)  [![AppVeyor](https://img.shields.io/appveyor/ci/zhangzhimin/tensor.svg)](https://ci.appveyor.com/project/zhangzhimin/tensor)
Tensor是一个C++实现的异构计算库，其上层接口极大地提高了高性能异构程序的开发效率。Tensor采用C++ AMP，Thrust的异构接口设计；具备类似Matlab的基本矩阵操作；将Eigen的延迟计算推广到GPU端；使用元编程技术追求扩展性和性能的极致。Tensor将致力于为众多异构应用提供底层支持。

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
可以看出，使用Tensor库，异构程序的开发效率可以获得极大的提升。下面的异构程序用于rgb图像归一化并分离三个通道的数据
``` cpp
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
	auto glts_rgb_normalized = tensor_cast<pointf<3>>(glts_rgb_stride) / pointf<3>::all(128.0f);
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
```
丰富的tensor操作，向量化的接口使代码看起来清晰整洁，延迟计算的使用，避免了额外的内存读写，让程序拥有极佳的性能。
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
挑选example下的简单示例查看会是一个很好的开始。
## 使用
```
git clone https://github.com/Matazure/tensor.git
```
使用上面指令获取tensor项目后，将根目录（默认是tensor）加入到目标项目的头文件路径即可，无需编译和其他第三方库依赖。有关C++项目，或者CUDA项目的创建，可自行查阅网上众多的资源。

## 性能
Tensor编写了大量性能测试用例来确保其优异的性能，可以在目标平台上运行生成的benchmark程来评估性能情况。 直接运行tensor_benchmark, hete_host_tensor_benchmark或者hete_cu_tensor_benchmark.

## 平台支持情况
| 设备  | Windows | Linux | OSX | Android | IOS |
| --- | --- | --- | --- | --- | --- |
| C++ | 支持 | 支持 | 支持 | 支持 | 支持
| CUDA | 支持 | 支持 | 支持 |  |  |
| OpenMP | 支持 | 支持 | 支持 | 支持 | 支持 |
<!-- |向量化|SSE|SSE|SSE|SSE| | -->

## 许可证书
该项目使用MIT证书授权，具体可查看LICENSE文件

## 联系方式
原作者希望更多的人加入到Tensor的使用开发中来，若在使用上有迷惑的地方，可直接通过邮件联系，周末可加QQ沟通  
邮箱： p3.1415@qq.com  
QQ： 417083997
