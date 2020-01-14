# [Tensor](https://github.com/Matazure/tensor) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Matazure/tensor/blob/master/LICENSE)

Tensor是一个C++实现的<!--异构-->多维数组计算库

## 介绍

Tensor主要用于多维数组及其计算，其可以便利高效的在CPU/GPU上实现遍历，滤波，转换等多种操作。也便于数据在CPU与GPU之间的传输交互。Tensor提供以下核心功能

* 泛型多维数组
* 延迟计算
* 并行计算
<!-- * 向量化指令集 -->

## 示例

### 基本用法

Tensor会提供泛型多维数组的数据结构及相应的算法，其遵行Modern C++的编程风格

```c++
#include <matazure/tensor>

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
#include <matazure/tensor>

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
#include <matazure/tensor>

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
    //阻塞等待执行完毕， 这是必须的
    cuda::device_synchronize();

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

[sample](sample)下有更多的示例可供参考

## 编译

需先安装[git](https://git-scm.com/)和[CMake](https://cmake.org/)及相应的编译工具，然后运行script下对应的编译脚本即可。  

```bash
git clone https://github.com/Matazure/tensor.git
```

* build_windows.bat编译windows版本
* build_native.sh编译unix版本(linux及mac)
* build_android.sh可以在linux主机编译android版本。

## 如何在你的项目中集成

在你的项目的头文件路径中包含include目录路径即可，无第三方库和动态库依赖。

对于CUDA项目，需要nvcc加入编译参数"--expt-extended-lambda"和"-std=c++11"。 CUDA的官方文档有nvcc编译参数设置的详细说明，也可参考本项目的CMakeLists.txt。

```cmake
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
```

## 环境要求

* 需要编译器开启C++11的支持\AAA
* 需要CUDA的编译器（a，需CUDA10.0及其以上版本

<!-- Tensor的代码规范遵循C++14标准， 所以只需编译器支持C++14即可, 推荐使用g++-7I
在符合CPU支持的基础上，需要安装[至少CUDA 10.1](https://developer.nvidia.com/cuda-downloads)，详情可查看[CUDA官方文档](http://docs.nvidia.com/cuda/index.html#axzz4kQuxAvUe) -->

<!-- ## 性能

Tensor编写了大量性能测试用例来确保其优异的性能，可以在目标平台上运行生成的benchmark程来评估性能情况。 直接运行tensor_benchmark, hete_host_tensor_benchmark或者hete_cu_tensor_benchmark. -->

<!-- ## 平台支持情况

| 设备  | Windows | Linux | OSX | Android | IOS |
| --- | --- | --- | --- | --- | --- |
| C++ | 支持 | 支持 | 支持 | 支持 | 支持
| CUDA | 支持 | 支持 | 支持 |  |  |
| OpenMP | 支持 | 支持 | 支持 | 支持 | 支持 |
|向量化|支持|支持|支持|支持|支持 | -->
