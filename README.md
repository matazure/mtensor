# [mtensor](https://github.com/matazure/mtensor) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/matazure/mtensor/blob/master/LICENSE) [![jenkins](http://jenkins.matazure.com/job/mtensor/job/master/badge/icon)](http://jenkins.matazure.com/blue/organizations/jenkins/mtensor/activity)

mtensor是一个支持延迟计算的多维数组计算库, 同时支持C++和CUDA/C++

## 基本功能

![image](doc/architecture.jpg)

mtensor主要用于多维数组及其计算, 其可以结构化高效地在CPU/GPU上实现遍历, 滤波, 转换等多种操作。也便于数据在CPU与GPU之间的传输交互。mtensor提供以下核心功能

* 同时支持CPU和CUDA两种计算框架
* 支持point, mtensor, local_mtensor等多种易用的泛型数据结构
* 通过lambda_mtensor实现了比模板表达式更强更简洁的延迟计算
* 实现了stl类似的fill, for_each, copy, transform等常用算法(algorithm).
* 基于延迟计算, 在view名字空间下实现了map, slice, stride, unstack等视图(view), 其返回的结果为相应的lambda_tensor.
* 算法和视图的接口为对于cuda和c++来说是统一的

除此之外, mtensor是一个遵循c++11的项目, 其是header only的, 我们可以很方便的在先用项目中集成使用它.

## 用法

### 基本数据结构

下面是常用的基本数据结构, mtensor拥有丰富的泛型多维数据结构和相应的操作.

```c++
#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char* argv[]) {
    point<float, 3> pt = {0.0f, 1.0f, 2.0f};
    std::cout << "pt: " << pt << std::endl;
    std::cout << "pt offset 1 value: " << pt[1] << std::endl;

    tensor<int, 2> ts = {{0, 1}, {2, 3}, {4, 5}};
    std::cout << "ts: " << std::endl << ts << std::endl;
    std::cout << "ts linear access 3 value : " << ts[3] << std::endl;
    auto idx = pointi<2>{1, 1};
    std::cout << "ts array access " << idx << " value : " << ts(idx) << std::endl;
}
```

输出

```console
pt: {0, 1, 2}
pt offset 1 value: 1
ts: 
{{0, 1}, 
{2, 3}, 
{4, 5}}
ts linear access 3 value : 3
ts array access {1, 1} value : 3
```

### for_index

for_index和cuda::for_index是最基本的计算接口, mtensor中的大部分计算都是由for_index来驱动的, 下面的例子是在gpu设备上， 
使用cuda::for_index来实现cuda::tensor的加法运算. 可以看出我们不必花过多精力在内存的申请释放上, 并且也无需计算thread坐标,
只需实现关于坐标的算子即可. 完整示例可查看[sample/sample_for_index.cu](sample/sample_for_index.cu)

```c++
pointi<2> shape{2, 3};
cuda::tensor<float, 2> ts_a(shape);
cuda::tensor<float, 2> ts_b(shape);
cuda::tensor<float, 2> ts_c(shape);
fill(ts_a, 1.0f);
fill(ts_b, 2.0f);
//使用cuda  lambda算子 需要申明__device__
auto functor = [=] MATAZURE_DEVICE(pointi<2> idx) { ts_c(idx) = ts_a(idx) + ts_b(idx); };
// 计算
cuda::for_index(shape, functor);
// 拷贝到主机tensor, 输出结果
tensor<float, 2> ts_re(shape);
mem_copy(ts_c, ts_re);
std::cout << ts_re << std::endl;
```

输出

```console
{{3, 3, 3}, 
{3, 3, 3}}
```

可以和[sample/sample_for_index.cpp](sample/sample_for_index.cpp)中的cpu端示例对比，看一下cpu和gpu的实现由什么区别

### 一个更复杂的例子

下述代码片段是[sample/sample_gradient.cpp](sample/sample_gradient.cpp)中的片段, 该例子中使用了view名字空间下的slice和cast视图, 
也通过了make_lambda来自定义了梯度视图和norm1视图, 并最终通过这些操作计算了图像的梯度强度. 
上以上过程中, 每个视图运算和make_lambda并不会真的去计算结果而只是把计算的方式存下来, 在最后的persist函数调用时才会申请内存,并遍历计算结果写入中.

```c++
tensor<byte, 2> img_gray = read_gray_image(argv[1]);
pointi<2> padding{1, 1};
tensor<byte, 2> image_pad_container(img_gray.shape() + padding * 2);
//该操作使得img具备越界一个元素访问的能力， 因为img(-1, -1)对应着image_pad_container(0, 0)
auto img_padding_view = view::slice(image_pad_container, padding, img_gray.shape());
copy(img_gray, img_padding_view);
//使用make_lambda构建梯度视图lambda_tensor
auto img_float_view = view::cast<float>(img_padding_view);
auto img_grad_view = make_lambda(img_float_view.shape(), [=](pointi<2> idx) {
    point<byte, 2> grad;
    grad[0] = img_float_view(idx + pointi<2>{1, 0}) - img_float_view(idx - pointi<2>{1, 0});
    grad[1] = img_float_view(idx + pointi<2>{0, 1}) - img_float_view(idx - pointi<2>{0, 1});
    return grad;
});
//将梯度转为norm1
auto grad_norm1_view = make_lambda(img_grad_view.shape(), [=](pointi<2> idx) {
    auto grad = img_grad_view(idx);
    return std::abs(grad[0]) + std::abs(grad[1]);
});
//转为byte类型并固化的tensor中, 将lambda_tensor固化到tensor结构中
auto grad_norm1 = view::cast<byte>(grad_norm1_view).persist();
//写入梯度到图像
write_gray_png("grad.png", grad_norm1);
```

### gpu的分块计算block_for_index

下面的示例展示了如何在block_for_index中使用shared内存来实现矩阵乘法,完整示例[sample/sample_matrix_mul.cu](sample/sample_matrix_mul.cu)

```c++
const int BLOCK_SIZE = 16;                      // block尺寸位16x16
typedef dim<BLOCK_SIZE, BLOCK_SIZE> BLOCK_DIM;  // 需要用一个dim<16, 16>来表示编译时block尺寸
point2i block_dim = BLOCK_DIM::value();  //将编译时的block尺寸转换为运行时point2i类型
point2i grid_dim{8, 8};                  // grid的尺寸，决定block的数目，布局
point2i global_dim = block_dim * grid_dim;  // 全局尺寸
int M = global_dim[0];
int N = global_dim[1];
int K = BLOCK_SIZE * 4;
cuda::tensor<float, 2> cmat_a(point2i{M, K});
cuda::tensor<float, 2> cmat_b(point2i{K, N});
cuda::tensor<float, 2> cmat_c(point2i{M, N});
// block_for_index需要给一个编译时的block尺寸， grid_dim是运行时的grid尺寸
cuda::block_for_index<BLOCK_DIM>(grid_dim,
    [=] __device__(cuda::block_index<BLOCK_DIM> block_idx) {
        auto row = block_idx.local[0];
        auto col = block_idx.local[1];
        auto global_row = block_idx.global[0];
        auto global_col = block_idx.global[1];
        //位于shared内存的分块矩阵
        __shared__ local_tensor<float, BLOCK_DIM> local_a;
        __shared__ local_tensor<float, BLOCK_DIM> local_b;
        float sum = 0.0f;
        for (int_t i = 0; i < K; i += BLOCK_SIZE) {
            //拷贝局部矩阵块
            local_a(row, col) = cmat_a(global_row, col + i);
            local_b(row, col) = cmat_b(row + i, global_col);
            cuda::syncthreads();
            //矩阵块乘法
            for (int_t N = 0; N < BLOCK_SIZE; N++) {
                sum += local_a(row, N) * local_b(N, col);
            }
            cuda::syncthreads();
        }
        cmat_c(block_idx.global) = sum;
    });
```

### c++和cuda通用代码实现

![img](doc/common_implement.jpg)

*一个通用实现阶段*

大部分需要同时支持cuda和c++的程序可以由若干个由上图所示的阶段构成, 在该阶段中会把tensor的数据拷贝的cuda::tensor, 
然后cuda和c++端均可以执行一个通用的实现, 在将cuda的数据拷贝会tensor. 这样cuda的运算结果最终和c++的结果是一致的. 
在上图中, 每个阶段的"common implement"是用模板泛型实现的, 其调用的函数需要申明_\_device\_\_ \_\_host\_\_
. 更多的细节看参考示例[smaple/sample_mandelbrot.hpp](sample/sample_mandelbrot.hpp). 
除此之外[include/matazure/view](include/matazure/view)下的实现都是cpu和gpu通用的(同一份代码实现), 
sample下的levelset分割算法是一个更复杂的泛型多维度异构通用实现.

### 其他

除此之外mandelbrot的例子还向我们展示了如何在mtensor中及其方便的使用openmp和使用特定尺寸的gpu资源

## mtensor的性能是否高效

mtensor在绝大部分场景下都不会带来额外的性能开销, 并且方便开发人员编写出高效的代码

* mtensor的延迟计算可以有效的避免内存频繁拷贝, 可以带来性能上的加成
* mtensor的泛型实现, 可以很容易的获得simd, fp16等带来的性能提升, 你只需将其设置为相应的模板类型就好
* mtensor的计算由for_index驱动, 你可以很方便的拓展for_index来获取特定的效果, 比如在c++端已实现了openmp的并行for_index

除此之外, mtensor还编写了大量的benchmark来确保性能指标, 
可以看出原生cuda的copy性能和mtensor封装后是一致, 甚至我们通过使用长字节的类型还很容易获取性能的提升

```console
bm_cuda_raw1f_for_copy/1000000000                    154.62GB/s    38.655G items/s
bm_cuda_tensor1f_for_array_index_copy/1000000000     154.782GB/s   38.6955G items/s
bm_cuda_tensor2f_for_array_index_copy/32000          151.656GB/s    37.914G items/s
bm_cuda_tensor1f_copy/1000000000                     154.465GB/s   38.6163G items/s
bm_cuda_tensor2f_copy/32000                          155.446GB/s   38.8614G items/s
bm_cuda_tensor2p4f_copy/8000                         240.697GB/s   15.0436G items/s
bm_cuda_tensor2a4f_copy/8000                         240.725GB/s   15.0453G items/s
```

## 如何在你的项目中集成

在你的项目的头文件路径中包含include目录路径即可, 无第三方库和动态库依赖。

除了需要你的C++编译器支持C++11外, 对于CUDA项目, 需要nvcc加入编译参数"--expt-extended-lambda"和"-std=c++11",
 CUDA的官方文档有nvcc编译参数设置的详细说明, 若使用CMake构建项目, 也可参考本项目的CMakeLists.txt。

```cmake
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
```

## 编译测试

### 环境要求

* 需要编译器支持C++11的支持, g++4.8及其以上版本
* 需要CMAKE3.8以上版本
* 需要git支持submodule特性,较新版本均支持
* 若需cuda支持,则需要cuda10.0及其以上版本, cuda9.0未测试, 9.0以下是无法支持的

也可以构建本项目的docker环境作为开发环境

```bash
docker build . -f dockerfile/mtensor-dev-ubuntu18.04.dockerfile -t mtensor-dev
```

### 编译

从github上克隆代码

```bash
git clone https://github.com/matazure/mtensor.git
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
./script/build_native.sh -DWITH_CUDA=ON -DWITH_OPENMP=ON -DWITH_SSE=ON
```

目前CUDA的mtensor编译还有几个关于主机设备函数调用的warning, 主要是std::shared_ptr和std::allocator产生, 可以忽略

### 测试

单元测试以ut开头, host表示主机cpu测试, cuda表示需要cuda环境的gpu测试

```bash
./build/bin/ut_host_mtensor
./build/bin/ut_cuda_mtensor
```

性能测试以bm开头, 示例以sample开头, 便不在一一举例

## 和其他开源库的比较

* mtensor的延迟计算技术比Eigen, xmtensor等更加简洁, lambda_mtensor的函数式实现比模板表达式更加通用简介
* mtensor支持CUDA版本的mtensor和延迟计算, 对于CUDA这种经常卡在显存带宽的计算设备来说是效果很好的
* mtensor围绕着多维坐标来设计算法, Nvidia的thrust是围绕着迭代器来实现的, 迭代器是有顺序依赖的, 其并不适用于并行计算
* mtensor是一个标准的现代C++风格的计算库, 其在内存管理, 接口设计等多方面都吸取了现代C++的优点

## 交流合作

有建议或者问题可以随时联系

* 邮箱 p3.1415@qq.com
* 微信 zhangzhimin-tju
