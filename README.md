# [mtensor](https://github.com/matazure/tensor) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/matazure/tensor/blob/master/LICENSE) [![jenkins](http://jenkins.matazure.com/job/mtensor/job/master/badge/icon)](http://jenkins.matazure.com/blue/organizations/jenkins/mtensor/activity)

mtensor是一个支持延迟计算的多维数组计算库, 同时支持C++和CUDA平台

## 基本功能

Tensor主要用于多维数组及其计算，其可以便利高效的在CPU/GPU上实现遍历，滤波，转换等多种操作。也便于数据在CPU与GPU之间的传输交互。Tensor提供以下核心功能

* 支持CPU和GPU端的tensor，lambda_tensor等的多维数组
* 支持CPU和GPU端的延迟计算技术
* 包含基本的fill, for_each, copy, transform等算法
* 基于延迟计算，在view名字空间下实现了slice， stride， clamp， unstack等算子
<!-- * 向量化指令集 -->

## 示例

### 基本用法

Tensor会提供泛型多维数组的数据结构及相应的算法，其遵行Modern C++的编程风格

```c++
#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char *argv[]) {
    constexpr int rank = 2;
    int col = 10;
    int row = 5;
    pointi<rank> shape{col, row};
    tensor<float, rank> ts(shape);

    // ts是关于2维坐标的赋值函数
    auto ts_setter = [ts](pointi<rank> index) { //ts是引用拷贝
        //将ts的元素每列递增1， 每行递增10
        ts(index) = index[0] + index[1] * 10;
    };

    //遍历shape大小的所有坐标， 默认原点是(0, 0)
    for_index(ts.shape(), ts_setter);

    //将ts的元素按行输出
    for (int j = 0; j < row; ++j) {
        for (int i = 0; i < col; ++i) {
            pointi<rank> index = {i, j};
            std::cout << ts(index) << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

### 延迟计算

我们使用Lambda Tensor来延迟计算技术，Lambda Tensor是一个抽象的多维数组，该数组不会指向具体的存储而是通过一个关于坐标的函数（算子）来描述。

```c++
#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char *argv[]) {
    //定义一个lambda算子用来描述抽象的一维数组, 其元素值等于坐标
    auto functor_a = [](int i) -> int {
        return i;
    };
    //lambda_tensor不仅需要算子， 也需要尺寸
    pointi<1> shape = { 100 };
    //构造lts_a， 其是一个lambda_tensor
    auto lts_a = make_lambda(shape, functor_a);

    //构造lts_b， 其元素值等于坐标的两倍
    auto functor_b = [] (int i) -> int{
        return i * 2;
    };
    auto lts_b = make_lambda(shape, functor_b);

    //构造lts_a加lts_b的lambda_tensor
    auto functor_add = [lts_a, lts_b] (int i) -> int {
        return lts_a[i] + lts_b[i];
    };
    auto lts_a_add_b = make_lambda(shape, functor_add);

    //上述的定义不会执行具体的运算，当我们去获取某一个具体坐标的值时其才会真正的去调用对应的算子
    std::cout << "offset 50 value is " << lts_a_add_b[50] << std::endl;
}
```

### 基于GPU的并行计算

```c++
#include <iostream>
#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char *argv[]) {
    pointi<2> shape {5, 5};
    tensor<float, 2> ts_a(shape);
    tensor<float, 2> ts_b(shape);
    tensor<float, 2> ts_c(shape);
    fill(ts_a, 1.0f);
    fill(ts_b, 2.0f);

    //构造gpu上的tensor
    cuda::tensor<float, 2> cts_a(shape);
    cuda::tensor<float, 2> cts_b(shape);
    cuda::tensor<float, 2> cts_c(shape);

    //将cpu上的数据拷贝到gpu上
    mem_copy(ts_a, cts_a);
    mem_copy(ts_b, cts_b);

    //在gpu上执行加法操作，这里使用了__device__ lambda, 需要加上nvcc的编译参数--expt-extended-lambda，
    cuda::for_index(shape, [cts_a, cts_b, cts_c] __device__ (pointi<2> index) {
        cts_c(index) = cts_a(index) + cts_b(index);
    });

    //将gpu上数据拷贝会cpu
    mem_copy(cts_c, ts_c);

    //打印输出
    for_each(ts_c, [](float e) {
        printf("%f, ", e);
    });
    printf("/n");

    return 0;
}
```

### GPU的分块计算block_for_index

下面是一个卷积实现,该实现仅为了展示block_for_index,其每个块的边界是无效值

```c++
#include <mtensor.hpp>
#include "image_helper.hpp"

using namespace matazure;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "sample6_convolution input_image" << std::endl;
        return -1;
    }

    //读取图像
    tensor<pointb<3>, 2> img_rgb = read_rgb_image(argv[1]);
    //使用均值滤波器
    tensor<pointf<3>, 2> kernel_mean(pointi<2>{3, 3});
    fill(kernel_mean, pointf<3>{0.111f, 0.111f, 0.111f});

    //向GPU拷贝数据
    cuda::tensor<pointb<3>, 2> cimg_rgb(img_rgb.shape());
    mem_copy(img_rgb, cimg_rgb);
    cuda::tensor<pointf<3>, 2> ckernel_mean(kernel_mean.shape());
    mem_copy(kernel_mean, ckernel_mean);

    //结果图像
    cuda::tensor<pointf<3>, 2> cimg_mean(img_rgb.shape());

    typedef dim<16, 16> BLOCK_DIM;
    pointi<2> block_dim = BLOCK_DIM::value();
    auto grid_dim = (img_rgb.shape() + block_dim - pointi<2>{1, 1}) / block_dim;
    auto padding = kernel_mean.shape() / 2;

    cuda::block_for_index<BLOCK_DIM>(grid_dim, [=] __device__(
                                                   cuda::block_index<BLOCK_DIM> block_idx) {
        //使用shared memory以获取更好的速度
        __shared__ local_tensor<pointf<3>, BLOCK_DIM> sh_ts_block;
        //若是无效区域则填充0
        if (inside_rect(block_idx.global, pointi<2>{0, 0}, cimg_rgb.shape())) {
            sh_ts_block(block_idx.local) = point_cast<float>(cimg_rgb(block_idx.global));
        } else {
            sh_ts_block(block_idx.local) = pointf<3>{0, 0, 0};
        }

        cuda::syncthreads();

        if (inside_rect(block_idx.local, padding, block_idx.block_dim - ckernel_mean.shape() + 1) &&
            inside_rect(block_idx.global, pointi<2>{0, 0}, cimg_rgb.shape())) {
            auto sum = pointf<3>{0, 0, 0};
            //在__device__ lambda算子里，一样可以使用matazure::for_index操作
            for_index(pointi<2>{0, 0}, ckernel_mean.shape(), [&](const pointi<2>& idx) {
                sum += sh_ts_block(block_idx.local + idx - padding) * ckernel_mean(idx);
            });
            cimg_mean(block_idx.global) = sum;
        }
    });

    //转换float类型到byte类型
    cuda::tensor<pointb<3>, 2> cimg_mean_byte(cimg_mean.shape());
    cuda::transform(cimg_mean, cimg_mean_byte,
                    [] __device__(pointf<3> pixel) { return point_cast<byte>(pixel); });
    //向主机写入图像
    tensor<pointb<3>, 2> img_mean(cimg_mean_byte.shape());
    mem_copy(cimg_mean_byte, img_mean);
    write_rgb_png("mean_" + std::string(argv[1]), img_mean);

    return 0;
}

```

[sample](sample)下有更多的示例可供参考

## 编译

需先安装[git](https://git-scm.com/)和[CMake](https://cmake.org/)及相应的编译工具g++/clang

也可以构建本项目的docker环境作为开发环境

```bash
docker build . -f dockerfile/tensor-dev-ubuntu18.04.dockerfile -t tensor-dev
```

从github上克隆代码

```bash
git clone https://github.com/matazure/tensor.git
```

获取submodule依赖

```bash
git submodule update --init -f third_party
```

使用编译脚本编译相关代码

* ./script/build_windows.bat编译windows版本
* ./script/build_native.sh编译unix版本(linux及mac)
* ./script/build_android.sh可以在linux主机编译android版本。

还可以添加参数来选择是否编译CUDA版本

```bash
./script/build_native.sh -DWITH_CUDA=ON
```

### 环境要求

* 需要编译器支持C++11的支持, g++4.8及其以上版本
* 需要CMAKE3.8以上版本
* 需要git支持submodule特性,较新版本均支持

目前CUDA的tensor编译还有几个关于主机设备函数调用的warning， 主要是std::shared_ptr和std::allocator产生， 可以忽略

## 如何在你的项目中集成

在你的项目的头文件路径中包含include目录路径即可，无第三方库和动态库依赖。

除了需要你的C++编译器支持C++11外, 对于CUDA项目，需要nvcc加入编译参数"--expt-extended-lambda"和"-std=c++11", CUDA的官方文档有nvcc编译参数设置的详细说明, 若使用CMake构建项目, 也可参考本项目的CMakeLists.txt。

```cmake
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
```

## Advanced

在tensor中，并不强调row和column，故column_major_layout仅表示第一维坐标是内存连续的, row_major_layout表示最后一维坐标是内存联系的

## 和其他开源库的比较

* mtensor的延迟计算技术比Eigen，xtensor等更加简洁，lambda_tensor的函数式实现比模板表达式更加通用简介
* mtensor支持CUDA版本的tensor和延迟计算，对于CUDA这种经常卡在显存带宽的计算设备来说是效果很好的
* mtensor围绕着多维坐标来设计算法，Nvidia的thrust是围绕着迭代器来实现的，迭代器是有顺序依赖的，其并不适用于并行计算
* mtensor是一个标准的现代C++风格的计算库，其在内存管理，接口设计等多方面都吸取了现代C++的优点

## 联系方式

有建议或者问题可以联系

* 邮箱 p3.1415@qq.com
* 微信 zhangzhimin-tju
