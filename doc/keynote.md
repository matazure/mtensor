# mtensor

mtensor是一个tensor计算库, 支持cuda的延迟计算, 项目地址为<https://github.com/matazure/mtensor>.

## 背景

延迟计算具有避免额外内存读写的优点, 该技术广泛应用于数值计算， 图像处理等领域. 目前绝大部分支持延迟计算的库都没支持cuda, 而对于gpu这种计算能里远强于内存带宽的设备来说, 延迟计算尤为重要, cuda 9版本以来, cuda c++逐渐完善了对c++11和c++14的支持,
使得cuda的延迟计算可以得到简洁的实现.

## 简介

![image](./images/architecture.jpg)

mtensor主要用于多维数组及其计算, 其可以结构化高效地在CPU/GPU上实现遍历, 滤波, 转换等多种操作。也便于数据在CPU与GPU之间的传输交互。mtensor提供以下核心功能

* 同时支持CPU和CUDA两种计算框架
* 支持point, tensor, local_mtensor等多种易用的泛型数据结构
* 通过lambda_tensor实现了比模板表达式更强更简洁的延迟计算
* 实现了fill, for_each, copy, transform等常用算法.
* 实现了丰富的map, slice, stride, gather等视图运算.
* 视图和算法均在c++和cuda上接口是统一的
* 使用方便, 仅需包含头文件和添加必要的cuda编译参数即可

### 延迟计算实现lambda_tensor

延迟计算有多种实现方式, 最为常见的是eigen所采用的模板表达式, 但该种方式每实现一种新的运算就要实现一个完整的模板表达式class, 过程非常繁琐. 不便于拓展新的运算. mtensor自研的基于[闭包算子](#闭包算子)的lambda tensor是一种更为通用简洁的延迟计算实现. 下面两张图一个是坐标空间, 另一个是值空间

|     |     |
|:---:|:---:|
| ![img](./images/idx_tensor.svg) | ![img](./images/f_tensor.svg) |
| array index | lambda tensor |

上图的lambda tensor, 可以通过如下代码定义. 其中f是一个自定义的闭包算子

```c++
auto lambda_ts = make_lambda(pointi{3, 3}, f);
```

事实上, 对于任何一个tensor, 我们都可以定义一个关于数组(线性)坐标的闭包算子来获得, mtensor的官方闭包算子在view名字空间下面, 这样的一个关于闭包的lambda tensor, 我们将其称为视图. mtensor的常见运算均是以视图的形式来定义的, 这样我们便可以方便地将其组合使用, 且不会带来额外的性能开销.

## 一个卷积的实现

### 卷积算子

下面是一段在mtensor中的卷积运算实现

```c++
typedef float value_type;
pointi<2> shape{256, 256};
tensor<value_type, 2> image(shape);
local_tensor<value_type, dim<3, 3>> kernel;
auto img_conv_view = view::conv(image, kernel);
pointi<2> padding = {1, 1};
pointi<2> result_shape = image.shape() - kernel.shape() + 1;
//不计算img_conv的边界, 避免越界
auto img_conv_valid_view = view::slice(img_conv_view, padding, result_shape);
tensor<value_type, 2> img_conv(result_shape);
copy(img_conv_valid_view, img_conv);
```

其中view::conv会调用卷积闭包算子, 其实现如下所以

```c++
template <typename _Tensor, typename _Kernel>
struct conv_functor<_Tensor, _Kernel, false> {
   private:
    typedef typename _Tensor::value_type value_type;
    static const int_t rank = _Tensor::rank;

    _Tensor ts_;
    _Kernel kernel_;
    pointi<rank> kernel_shape_;

   public:
    //我们通过构造函数将所需的图像和卷积核传入
    conv_functor(_Tensor ts, _Kernel kernel)
        : ts_(ts),
          kernel_(kernel),
          kernel_shape_(kernel.shape()){}

    //MATAZURE_GENNERAL等价于__host__ __device__关键字, 申明这个函数是一个c++/cuda均可使用
    //我们所有的tensor均可认为是一个关于坐标的函数, 故我们的参数传入一个数组坐标即可
    __host__ __device__ value_type operator()(pointi<_Tensor::rank> idx) const {
        //零初始化
        auto re = matazure::zero<value_type>::value();
        //遍历卷积核尺寸的坐标
        for_index(kernel_shape_, [&](pointi<rank> neigbor_idx) {
            //之所以减kernel_shape_/2是为了让卷积核中心和idx对齐
            re += kernel_(neigbor_idx) * ts_(idx + neigbor_idx - kernel_shape_ / 2);
        });

        return re;
    }
};
```

### 使用向量化指令集

在卷积神经网络里卷积运算是多通道的, 我们可以通过向量化指令集来加速他们. 其在mtensor中的实现非常简单, 将tensor的value_type特化为simd类型就好, 只需添加一行代码.

```c++
//使用gcc/clang的向量化类型申明
typedef float value_type __attribute__((vector_size(16)));
pointi<2> shape{256, 256};
tensor<value_type, 2> image(shape);
local_tensor<value_type, dim<3, 3>> kernel;
auto img_conv_view = view::conv(image, kernel);
pointi<2> padding = {1, 1};
pointi<2> result_shape = image.shape() - kernel.shape() + 1;
//不计算img_conv的边界, 避免越界
auto img_conv_valid_view = view::slice(img_conv_view, padding, result_shape);
tensor<value_type, 2> img_conv(result_shape);
copy(img_conv_valid_view, img_conv);
```

### 使用openmp加速

我们的conv运算的结果是没有前后依赖, 所以很容易实现其并行化. 在copy后面加一个omp_policy就好了, 这样copy操作就会调用openmp的并行循环来遍历计算卷积结果. mtensor之所以强调闭包算子, 是因为其可以解耦程序的描述(算子)和运行(copy). 我们之所以使用关于坐标的闭包算子, 而不是像stl一样关于迭代器的算子, 是因为我们关注的是强并行性的问题, 不能像迭代器一样有前后依赖.(迭代器这个名字就已经说明它有前后依赖关系了)

```c++
typedef float value_type;
pointi<2> shape{256, 256};
tensor<value_type, 2> image(shape);
local_tensor<value_type, dim<3, 3>> kernel;
auto img_conv_view = view::conv(image, kernel);
pointi<2> padding = {1, 1};
pointi<2> result_shape = image.shape() - kernel.shape() + 1;
//不计算img_conv的边界, 避免越界
auto img_conv_valid_view = view::slice(img_conv_view, padding, result_shape);
tensor<value_type, 2> img_conv(result_shape);
copy(omp_policy{}, img_conv_valid_view, img_conv);
```

### cuda本版实现

除了在cpu, 很多时候我们希望程序运行在gpu上, mtensor中只需要使用cuda::tensor就可以切换到cuda了.

```c++
typedef float value_type;
pointi<2> shape{256, 256};
cuda::tensor<value_type, 2> image(shape);
local_tensor<value_type, dim<3, 3>> kernel;
auto img_conv_view = view::conv(image, kernel);
pointi<2> padding = {1, 1};
pointi<2> result_shape = image.shape() - kernel.shape() + 1;
//不计算img_conv的边界, 避免越界
auto img_conv_valid_view = view::slice(img_conv_view, padding, result_shape);
cuda::tensor<value_type, 2> img_conv(result_shape);
copy(img_conv_valid_view, img_conv);
```

### 延迟计算所带来的代码融合

前面的小节阐述了卷积运算如何快速向量化和并行化, 下面看下卷积如何后接stride和relu6视图

```c++
typedef float value_type;
pointi<2> shape{256, 256};
tensor<value_type, 2> image(shape);
local_tensor<value_type, dim<3, 3>> kernel;
auto img_conv_view = view::conv(image, kernel);
auto img_conv_stride_view = view::stride(img_conv_view, pointi<2>{2, 2});
auto img_conv_stride_relu6_view = view::map(
    img_conv_stride_view, [](value_type v) { return std::min(std::max(0.0f, v), 6.0f); })
pointi<2> padding = {1, 1};
pointi<2> result_shape = image.shape() - kernel.shape() + 1;
//不计算img_conv的边界, 避免越界
auto img_conv_valid_view = view::slice(img_conv_stride_relu6_view, padding, result_shape);
tensor<value_type, 2> img_conv(result_shape);
copy(img_conv_valid_view, img_conv);
```

view::conv|stride|map并不会执行真正的计算, 而只是把算子记录下来, 当copy的时候才会去遍历计算结果, 并将其写入到内存中.

|   |   |   |
|:-:|:-:|:-:|
| ![img](./images/map_seq.svg) | ![img](./images/seq_map.svg) | ![img](./images/seq_map_opt.svg) |
| 计算的序列 | 算子序列的计算 | 算子融合后的计算 |

上图计算的序列所示, 一旦我们把计算的结果写入内存(上图用圆表示), 则c++编译器就无法穿透优化他们, 因为内存有可能在其他的地方被修改. 而我们延迟计算所表示的闭包算子系列(上图中的小矩形)的计算对于编译器来说是典型的局部优化问题, 编译器可以很容易地优化它们.

通过使用mtensor中丰富的数据结构, 算法和视图我们可以快速地实现一个高效的并行程序. 不仅仅是研究阶段, mtensor的结构化高效程序实现在产品落地阶段也能带来维护和性能上的便利. 在我们的项目中有上述代码的相关测试, 可按照readme编译并执行

```c++
./build/bin/bm_host_mtensor --benchmark_filter=conv
```

## 闭包算子

这里的闭包算子在很多编程语言中叫做闭包(javascript等)或者算子(c++的stl/boost等), 我们这里直接叫闭包算子, 顾名思义它有两个核心性质.

* 闭包, 指的是可以脱离上下文独立存在, 它会捕获它所需要的变量到其内部
* 算子, 指的是可以像一个函数一样执行, 在c++中它可以是函数指针, lambda表达式或者拥有operator()的class实例

这是一个常见的一次函数y=kx+b, 我们应该怎么实现它呢? 先看第一种方式

```c++
float line(float k, float b, float x) {
    return k * x + b;
}
```

这种方式没法体现出k, b和x的区别, 因为只有x才是真正的自变量, k和b都应该是常数, 那我们换种方法

```c++
float line(float x) {
    return 3.0f * x + 2.0f;
}
```

如果像上面一样实现, 那不同的k和b都需要定义一个函数, 并且很多时候k和b需要程序执行时才能确定, 所以第二种方法也是不可行的. 其实我们可以通过闭包算子来实现

```c++
struct line_functor {
    line_functor(float k, float b) : k_(k), b_(b) {}

    float operator()(float x) const {
        return k_ * x + b_;
    }

private:
    float k_;
    float b_;
};

int main () {
    line_functor line(3.0f, 2.0f);
    auto y = line(10.0f); // y = 32.0f
}
```

通过闭包算子我们实现了y=kx+b, 区分了x才是真正的自变量而k和b则通过构造函数传入, line_functor才真正和我们数学上的函数定义是一致的, 面向对象编程的核心之一就是解决面向过程中的函数和数学上的函数无法良好对应的问题. 很多时候闭包算子并不需要如此繁琐, lamba表达式使用起来更为简洁

```c+++
float k = 3.0f;
float b = 2.0f;
auto line = [k, b](float x) {
    return k * x + b;
};
```

这样的实现也是一个闭包算子, 其将k和b捕获, 而x则作为自变量传进去

## 和其他开源项目比较

mtensor从技术上借鉴了很多其他开源库的优点, 但也有下面的一些明显区别

* mtensor的延迟计算技术比eigen, xtensor等更加简洁, lambda_tensor的函数式实现比模板表达式更加通用简洁
* mtensor的cuda延迟计算是其他库目前还未具备的
* mtensor的闭包算子是关于数组/线性坐标的, thrust和stl是围绕着迭代器来实现的, 笔者认为并行计算中不应以迭代器作为数据的访问方式, 迭代器的前后依赖和并行性有剧烈冲突

和一些以具体的图像处理, 矩阵或者数值分析等行业领域为核心问题不同的是, mtensor以这些计算问题中的共性为研究对象,
致力于成为**计算模式**的最佳实践

## 参考资料

本项目借鉴了下面资料和项目的很多设计思想

* 结构化并行程序设计<高效计算模式>
* 微软amp
* cuda-samples和thrust
* <https://devblogs.nvidia.com/power-cpp11-cuda-7/>
* xtensor
* itk
* boost::ublas/mlp/compute/gil

## 交流合作

有建议或者问题可以随时联系

* 邮箱 p3.1415@qq.com
* 微信 zhangzhimin-tju
