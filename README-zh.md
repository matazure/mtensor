# mtensor

mtensor是一个c++/cuda模板库, 其支持tensor的延迟计算. 和Eigen不同的是, mtensor以张量为核心数据结构, 并且支持cuda的延迟计算. mtensor不提供线性代数和数值计算的功能.

## 如何使用


### tensor

mtensor由模板类**tensor**来管理基本的数据结构, 可以这样使用它

```c++
pointi<2> shape = {10, 20};
tensor<int, 2> cts(shape, runtime::cuda);
tensor<int, 2> ts(shape, runtime::host);
```

tensor可以通过模板参数来定义value_type和维度, 通过cuda/host来可以指定tensor的内存类型.

### lambda_tensor

lambda_tensor是一个函数式的tensor结构, 其有着很强的表达能力.
我们可以通过shape和"index->value"的函数来定义lambda_tensor.

```c++
pointi<2> shape = {10, 10};
tensor<int, 2> ts(shape);
auto fun = [=](pointi<2> idx) {
    return ts(idx) + ts(idx);
};
lambda_tensor<decltype(fun)> lts(shape, fun);
// will evaluate fun(pointi<2>{2,2}), and return it;
auto value = lts(2, 2);
```

### for_index

for_index是一个函数式的for循环, 原则上我们认为其是并行的. for_index支持host和cuda两种执行方式, 我们需要注意在cuda运行时里, 我们只能访问对应运行时的tensor.

```c++
pointi<2> shape = {10, 10};
tensor<int, 2> cts(shape, runtime::cuda);
auto lts = make_lambda_tensor(shape, [](pointi<2> idx)->int{
    return idx[0] * 10 + idx[1];
});

for_index(shape, [=](pointi<2> idx){
    cts(idx) = lts(idx);
}, cts.runtime()); // in cuda runtime, we can only access cuda runtime tensor
```

我们可以拓展for_index实现不同的加速引擎, 比如我们可以拓展一个omp的for_index来使用cpu的多核.

上述三个功能是mtensor库的核心构成, 我们可以为绕它们快速实现很多功能. 详细可以参考代码用例[samples](samples)

## 用例编译

```bash
git submodule update --init .
cmake -B build
```

## 如何使用

下载[mtensor.hpp](mtensor.hpp)单一头文件, 将其包含在项目中.

```
#include "mtensor.hpp"
```

在使用mtensor中遇到问题, 可以自行修复下然后提一个pull request.
